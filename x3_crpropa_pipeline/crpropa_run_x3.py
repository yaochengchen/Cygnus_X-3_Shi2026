#!/usr/bin/env python3
"""Run continuous X-3 proton injection in the prepared periodic MHD field."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from x3_common import (
    KYR_S,
    PC_CM,
    canonical_hash,
    crpropa_diffusion_scale,
    diffusion_cm2_s,
    load_config,
    particle_weights,
    resolve_path,
)


def load_json(path: Path, schema: str) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if value.get("schema") != schema:
        raise ValueError(f"Expected {schema} in {path}, got {value.get('schema')!r}")
    return value


def sample_energies_gev(rng: np.random.Generator, count: int, source: dict) -> np.ndarray:
    emin = float(source["sample_emin_tev"]) * 1e3
    emax = float(source["sample_emax_pev"]) * 1e6
    return np.exp(rng.uniform(math.log(emin), math.log(emax), count))


def random_directions(rng: np.random.Generator, count: int) -> np.ndarray:
    directions = rng.normal(size=(count, 3))
    return directions / np.linalg.norm(directions, axis=1)[:, None]


def slice_counts(total: int, slices: int) -> list[int]:
    if total < slices:
        raise ValueError("particles_per_age must be >= injection_time_slices")
    quotient, remainder = divmod(total, slices)
    return [quotient + (index < remainder) for index in range(slices)]


def shi_coefficients(config: dict) -> dict:
    baseline = config["transport"]["shi_baseline"]
    return {
        "d0_cm2_s_at_reference": float(baseline["d0_cm2_s_at_reference"]),
        "reference_energy_pev": float(baseline["reference_energy_pev"]),
        "delta": float(baseline["delta"]),
        "sde_epsilon": 1.0,
    }


def user_sde_coefficients(config: dict) -> dict:
    """Return the diffusion law explicitly supplied by the user in YAML."""
    supplied = config["transport"]["user_diffusion"]
    alpha = float(supplied.get("alpha", 1.0 / 3.0))
    if not math.isclose(alpha, 1.0 / 3.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(
            "This X-3 production setup requires transport.user_diffusion.alpha=1/3"
        )
    return {
        "d0_cm2_s_at_reference": float(supplied["d0_cm2_s_at_reference"]),
        "reference_energy_pev": float(supplied["reference_energy_pev"]),
        "delta": alpha,
        "sde_epsilon": float(supplied["sde_epsilon"]),
    }


def run_analytic_age(config: dict, age_kyr: float, rng: np.random.Generator) -> dict[str, np.ndarray]:
    transport = config["transport"]
    coefficient = shi_coefficients(config)
    slices = int(transport["injection_time_slices"])
    counts = slice_counts(int(transport["particles_per_age"]), slices)
    delta_t_kyr = age_kyr / slices
    arrays: dict[str, list[np.ndarray]] = {
        key: [] for key in ("energy_gev", "weight", "age_kyr", "x_pc", "y_pc", "z_pc")
    }
    d0 = coefficient["d0_cm2_s_at_reference"]
    e0 = coefficient["reference_energy_pev"] * 1e6
    delta = coefficient["delta"]
    for index, count in enumerate(counts):
        propagation_age = (index + 0.5) * delta_t_kyr
        energy = sample_energies_gev(rng, count, config["source"])
        diffusion = diffusion_cm2_s(energy, d0, e0, delta)
        sigma_pc = np.sqrt(2.0 * diffusion * propagation_age * KYR_S) / PC_CM
        positions = rng.normal(size=(count, 3)) * sigma_pc[:, None]
        weights = particle_weights(energy, delta_t_kyr, count, config["source"])
        arrays["energy_gev"].append(energy)
        arrays["weight"].append(weights)
        arrays["age_kyr"].append(np.full(count, propagation_age))
        arrays["x_pc"].append(positions[:, 0])
        arrays["y_pc"].append(positions[:, 1])
        arrays["z_pc"].append(positions[:, 2])
    return {key: np.concatenate(parts) for key, parts in arrays.items()}


def make_crpropa_mhd_field(background: dict, background_path: Path):
    """Create a periodic MagneticFieldGrid and verify binary axis ordering."""
    import crpropa as crp

    shape = tuple(int(value) for value in background["grid_shape_xyz"])
    if len(set(shape)) != 1:
        raise ValueError("CRPropa GridProperties export currently requires a cubic grid")
    size_pc = float(background["box_size_pc"])
    spacing = size_pc / shape[0]
    origin = crp.Vector3d(*([-0.5 * size_pc * crp.pc] * 3))
    properties = crp.GridProperties(origin, shape[0], spacing * crp.pc)
    grid = crp.Grid3f(properties)
    binary = Path(background["crpropa_magnetic_binary"])
    if not binary.is_absolute():
        binary = (background_path.parent / binary).resolve()
    if not binary.exists():
        raise FileNotFoundError(f"Missing prepared CRPropa magnetic grid: {binary}")
    crp.loadGrid(grid, str(binary), crp.microgauss)
    field = crp.MagneticFieldGrid(grid)

    # Fail early if a future CRPropa build changes raw Grid3f ordering.
    raw = np.memmap(binary, mode="r", dtype="<f4", shape=shape + (3,), order="C")
    for index in ((0, 0, 0), tuple(value // 2 for value in shape)):
        point_pc = [-0.5 * size_pc + (index[axis] + 0.5) * spacing for axis in range(3)]
        actual = field.getField(crp.Vector3d(*(np.asarray(point_pc) * crp.pc)))
        actual_uG = np.array([actual.x, actual.y, actual.z]) / crp.microgauss
        expected_uG = np.asarray(raw[index], dtype=float)
        if not np.allclose(actual_uG, expected_uG, rtol=3e-5, atol=1e-6):
            raise RuntimeError(
                "CRPropa Grid3f binary order self-check failed. Check CRPropa version and "
                f"crpropa_binary_order. Expected {expected_uG}, obtained {actual_uG}."
            )
    return field, str(binary)


def run_crpropa_age(
    config: dict,
    background: dict,
    background_path: Path,
    coefficient: dict,
    age_kyr: float,
    rng: np.random.Generator,
) -> tuple[dict[str, np.ndarray], dict]:
    try:
        import crpropa as crp
    except ImportError as exc:
        raise RuntimeError("crpropa_sde_mhd requires the CRPropa Python package") from exc

    transport = config["transport"]
    field, binary = make_crpropa_mhd_field(background, background_path)
    d0 = float(coefficient["d0_cm2_s_at_reference"])
    e0_gev = float(coefficient["reference_energy_pev"]) * 1e6
    delta = float(coefficient["delta"])
    epsilon = float(coefficient["sde_epsilon"])
    scale = crpropa_diffusion_scale(d0, e0_gev, delta)
    slices = int(transport["injection_time_slices"])
    counts = slice_counts(int(transport["particles_per_age"]), slices)
    delta_t_kyr = age_kyr / slices

    seed = int(config["random"]["transport_seed"])
    crp.Random.instance().seed(seed)
    crp.Random.seedThreads(seed)
    arrays: dict[str, list[np.ndarray]] = {
        key: []
        for key in (
            "energy_gev",
            "weight",
            "age_kyr",
            "x_pc",
            "y_pc",
            "z_pc",
            "trajectory_pc",
        )
    }

    propagator = crp.DiffusionSDE(
        field,
        float(transport["sde_tolerance"]),
        float(transport["sde_min_step_pc"]) * crp.pc,
        float(transport["sde_max_step_pc"]) * crp.pc,
        epsilon,
    )
    propagator.setAlpha(delta)
    propagator.setScale(scale)

    for slice_index, count in enumerate(counts):
        propagation_age = (slice_index + 0.5) * delta_t_kyr
        energy = sample_energies_gev(rng, count, config["source"])
        weights = particle_weights(energy, delta_t_kyr, count, config["source"])
        directions = random_directions(rng, count)
        candidates = crp.CandidateVector()
        for particle_index in range(count):
            candidate = crp.Candidate(
                crp.nucleusId(1, 1),
                energy[particle_index] * crp.GeV,
                crp.Vector3d(0.0),
                crp.Vector3d(*directions[particle_index]),
                0.0,
                float(weights[particle_index]),
                "X3",
            )
            candidates.push_back(candidate)

        modules = crp.ModuleList()
        modules.add(propagator)
        modules.add(crp.MaximumTrajectoryLength(propagation_age * crp.kyr * crp.c_light))
        modules.run(candidates, False)

        positions = np.empty((count, 3))
        trajectories = np.empty(count)
        for particle_index, candidate in enumerate(candidates):
            position = candidate.current.getPosition()
            positions[particle_index] = [
                position.x / crp.pc,
                position.y / crp.pc,
                position.z / crp.pc,
            ]
            trajectories[particle_index] = candidate.getTrajectoryLength() / crp.pc
        arrays["energy_gev"].append(energy)
        arrays["weight"].append(weights)
        arrays["age_kyr"].append(np.full(count, propagation_age))
        arrays["x_pc"].append(positions[:, 0])
        arrays["y_pc"].append(positions[:, 1])
        arrays["z_pc"].append(positions[:, 2])
        arrays["trajectory_pc"].append(trajectories)
        print(f"  age={age_kyr:g} kyr, injection slice {slice_index + 1}/{slices} complete")

    result = {key: np.concatenate(parts) for key, parts in arrays.items()}
    details = {
        "diffusion_coefficient_source": "transport.user_diffusion in x3_config.yaml",
        "diffusion_d0_cm2_s_at_reference": d0,
        "diffusion_reference_energy_pev": e0_gev / 1e6,
        "diffusion_alpha": delta,
        "crpropa_version": getattr(crp, "__version__", "unknown"),
        "crpropa_diffusion_scale": scale,
        "sde_epsilon": epsilon,
        "magnetic_binary": binary,
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS", "runtime default"),
    }
    return result, details


def save_age(
    output_dir: Path,
    arrays: dict[str, np.ndarray],
    config: dict,
    background_path: Path,
    background: dict,
    transport_path: Path | None,
    transport_summary: dict | None,
    engine: str,
    age_kyr: float,
    engine_details: dict,
) -> Path:
    label = f"{age_kyr:g}".replace(".", "p")
    output = output_dir / f"protons_age_{label}kyr_{engine}.npz"
    science_config = {key: value for key, value in config.items() if not key.startswith("_")}
    metadata = {
        "schema": "x3-proton-endpoints-v2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "engine": engine,
        "age_kyr": age_kyr,
        "continuous_injection": True,
        "injection_time_quadrature": "equal-width midpoint slices",
        "particle_count": int(len(arrays["energy_gev"])),
        "config_path": config["_config_path"],
        "config_sha256": canonical_hash(science_config),
        "background_path": str(background_path),
        "background_sha256": canonical_hash(background),
        "transport_path": str(transport_path) if transport_path else None,
        "transport_sha256": canonical_hash(transport_summary) if transport_summary else None,
        "six_degree_aperture_used_as_boundary": False,
        "periodic_mhd_box_used_as_particle_boundary": False,
        "proton_interactions_during_transport": False,
        "weight_meaning": "number of protons represented by this present-day pseudo-particle",
        "source_normalization_is_placeholder": bool(config["source"]["normalization_is_placeholder"]),
        "python": platform.python_version(),
        **engine_details,
    }
    np.savez_compressed(output, **arrays, metadata_json=json.dumps(metadata, sort_keys=True))
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="x3_config.yaml")
    parser.add_argument("--background", default=None)
    parser.add_argument("--engine", choices=("analytic_shi", "crpropa_sde_mhd"), default=None)
    parser.add_argument("--age-kyr", type=float, action="append", dest="ages")
    parser.add_argument("--n-particles", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.n_particles is not None:
        config["transport"]["particles_per_age"] = args.n_particles
    engine = args.engine or config["transport"]["engine"]
    ages = args.ages or [float(value) for value in config["transport"]["ages_kyr"]]
    background_path = resolve_path(config, args.background or config["project"]["background_file"])
    background = load_json(background_path, "x3-mhd-background-v2")
    transport_path = None
    transport_summary = None
    if engine == "crpropa_sde_mhd":
        coefficient = user_sde_coefficients(config)
    else:
        coefficient = shi_coefficients(config)

    output_dir = resolve_path(config, config["project"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(int(config["random"]["transport_seed"]))
    scale = crpropa_diffusion_scale(
        float(coefficient["d0_cm2_s_at_reference"]),
        float(coefficient["reference_energy_pev"]) * 1e6,
        float(coefficient["delta"]),
    )
    print(
        "Engine={}; user D0={:.3e} cm^2/s, alpha={:.4f}, CRPropa scale={:.8g}".format(
            engine, coefficient["d0_cm2_s_at_reference"], coefficient["delta"], scale
        )
    )
    for age in ages:
        if engine == "analytic_shi":
            arrays = run_analytic_age(config, age, rng)
            details = {"crpropa_diffusion_scale_equivalent": scale}
        else:
            arrays, details = run_crpropa_age(
                config, background, background_path, coefficient, age, rng
            )
        output = save_age(
            output_dir,
            arrays,
            config,
            background_path,
            background,
            transport_path,
            transport_summary,
            engine,
            age,
            details,
        )
        print(f"Wrote {output}")


if __name__ == "__main__":
    main()
