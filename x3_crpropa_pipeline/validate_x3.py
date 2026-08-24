#!/usr/bin/env python3
"""Validate MHD scaling, SDE steps, thin-target limits, and endpoint moments."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from x3_common import (
    C_CM_S,
    KYR_S,
    PC_CM,
    angular_radius_to_pc,
    crpropa_diffusion_scale,
    diffusion_cm2_s,
    json_from_npz,
    load_config,
    resolve_path,
)


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    return float(np.sum(values * weights) / np.sum(weights))


def load_json(path: Path, schema: str) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if value.get("schema") != schema:
        raise ValueError(f"Expected {schema} in {path}")
    return value


def coefficients(config: dict, engine: str) -> dict:
    if engine == "analytic_shi":
        baseline = config["transport"]["shi_baseline"]
        return {
            "d0_cm2_s_at_reference": float(baseline["d0_cm2_s_at_reference"]),
            "reference_energy_pev": float(baseline["reference_energy_pev"]),
            "delta": float(baseline["delta"]),
            "sde_epsilon": 1.0,
        }
    supplied = config["transport"]["user_diffusion"]
    alpha = float(supplied.get("alpha", 1.0 / 3.0))
    if not math.isclose(alpha, 1.0 / 3.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("transport.user_diffusion.alpha must equal 1/3")
    return {
        "d0_cm2_s_at_reference": float(supplied["d0_cm2_s_at_reference"]),
        "reference_energy_pev": float(supplied["reference_energy_pev"]),
        "delta": alpha,
        "sde_epsilon": float(supplied["sde_epsilon"]),
    }


def validate_config(
    config: dict, background: dict, optional_transport_summary: dict | None = None
) -> dict:
    transport = config["transport"]
    source = config["source"]
    obs = config["observation"]
    coefficient = coefficients(config, "crpropa_sde_mhd")
    d0 = float(coefficient["d0_cm2_s_at_reference"])
    e0_gev = float(coefficient["reference_energy_pev"]) * 1e6
    delta = float(coefficient["delta"])
    scale = crpropa_diffusion_scale(d0, e0_gev, delta)
    max_energy_gev = float(source["sample_emax_pev"]) * 1e6
    cutoff_gev = float(source["cutoff_pev"]) * 1e6
    dmax = float(diffusion_cm2_s(max_energy_gev, d0, e0_gev, delta))
    dt_s = float(transport["sde_max_step_pc"]) * PC_CM / C_CM_S
    rms_parallel_pc = math.sqrt(2.0 * dmax * dt_s) / PC_CM
    cell_pc = float(background["cell_size_pc"])
    injection_pc = float(background["magnetic_field"]["injection_scale_pc"])
    aperture_pc = angular_radius_to_pc(float(obs["distance_kpc"]), float(obs["aperture_deg"]))
    oldest_kyr = max(float(value) for value in transport["ages_kyr"])
    mean_density = float(background["density"]["mean_cm3"])
    pp_probability = mean_density * 50e-27 * C_CM_S * oldest_kyr * KYR_S

    warnings = []
    if max_energy_gev < 5.0 * cutoff_gev:
        warnings.append("sample_emax is below 5 times the exponential cutoff")
    if rms_parallel_pc > 0.5 * injection_pc:
        warnings.append("maximum SDE step produces >0.5 injection-scale RMS displacement at sample_emax")
    if float(transport["sde_max_step_pc"]) > injection_pc / 5.0:
        warnings.append("sde_max_step_pc is larger than one fifth of the turbulence injection scale")
    if float(transport["sde_max_step_pc"]) < cell_pc:
        warnings.append("sde_max_step_pc is below one MHD cell; interpolation cost may dominate")
    if abs(aperture_pc - 1000.0) > 100.0:
        warnings.append("configured 6-degree aperture differs from 1 kpc by more than 10 percent")
    if background["magnetic_field"]["dimensionless_rms_divB_over_B_per_cell"] > 0.1:
        warnings.append("prepared magnetic grid has a large finite-difference divergence diagnostic")
    result = {
        "production_diffusion": {
            "coefficient_source": "transport.user_diffusion in x3_config.yaml",
            "d0_cm2_s": d0,
            "reference_energy_gev": e0_gev,
            "delta": delta,
            "sde_epsilon": float(coefficient["sde_epsilon"]),
            "crpropa_scale": scale,
        },
        "mhd": {
            "grid_shape_xyz": background["grid_shape_xyz"],
            "box_size_pc": background["box_size_pc"],
            "cell_size_pc": cell_pc,
            "injection_scale_pc": injection_pc,
            "estimated_correlation_length_pc": background["magnetic_field"][
                "estimated_correlation_length_pc"
            ],
            "mean_density_cm3": mean_density,
            "B_rms_uG": background["magnetic_field"]["rms_strength"],
            "Ms": background["turbulence"]["sonic_mach"],
            "MA": background["turbulence"]["alfvenic_mach"],
            "periodic": background["periodic"],
        },
        "sde_step": {
            "max_step_pc_over_c": float(transport["sde_max_step_pc"]),
            "rms_parallel_displacement_at_sample_emax_pc": rms_parallel_pc,
            "fraction_of_injection_scale": rms_parallel_pc / injection_pc,
        },
        "geometry": {"aperture_radius_pc": aperture_pc, "aperture_is_boundary": False},
        "thin_target": {
            "conservative_pp_interaction_probability_at_oldest_age": pp_probability,
            "weighted_emissivity_recommended": pp_probability < 0.1,
        },
        "warnings": warnings,
    }
    if optional_transport_summary is not None and optional_transport_summary.get(
        "not_used_by_crpropa_run"
    ):
        estimated = optional_transport_summary["crpropa_powerlaw"]
        result["optional_test_particle_comparison"] = {
            "used_by_crpropa": False,
            "d0_cm2_s": estimated["d0_cm2_s_at_reference"],
            "reference_energy_pev": estimated["reference_energy_pev"],
            "delta": estimated["delta"],
        }
    elif optional_transport_summary is not None:
        result["warnings"].append(
            "Ignored a legacy transport diagnostic; regenerate it with estimate_transport.py"
        )
    return result


def validate_endpoints(input_path: Path, config: dict, background: dict) -> dict:
    with np.load(input_path, allow_pickle=False) as data:
        metadata = json_from_npz(data)
        coefficient = coefficients(config, metadata["engine"])
        weights = data["weight"]
        time_s = data["age_kyr"] * KYR_S
        energy = data["energy_gev"]
        d_parallel = diffusion_cm2_s(
            energy,
            float(coefficient["d0_cm2_s_at_reference"]),
            float(coefficient["reference_energy_pev"]) * 1e6,
            float(coefficient["delta"]),
        )
        epsilon = float(coefficient["sde_epsilon"])
        positions = np.column_stack((data["x_pc"], data["y_pc"], data["z_pc"])) * PC_CM
        mean_b = np.asarray(background["magnetic_field"]["mean_vector"], dtype=float)
        if np.linalg.norm(mean_b) == 0:
            mean_b = np.array([0.0, 0.0, 1.0])
        bhat = mean_b / np.linalg.norm(mean_b)
        parallel = positions @ bhat
        perpendicular_sq = np.sum(positions**2, axis=1) - parallel**2
        parallel_ratio = parallel**2 / (2.0 * time_s * d_parallel)
        perpendicular_ratio = perpendicular_sq / (4.0 * time_s * d_parallel * epsilon)
        trace_expected = 2.0 * time_s * d_parallel * (1.0 + 2.0 * epsilon)
        trace_ratio = np.sum(positions**2, axis=1) / trace_expected
        return {
            "input": str(input_path),
            "engine": metadata["engine"],
            "age_kyr": metadata["age_kyr"],
            "particle_count": len(weights),
            "weighted_moment_ratios": {
                "r2_over_expected_tensor_trace": weighted_mean(trace_ratio, weights),
                "mean_field_parallel_x2_over_2Dparallel_t": weighted_mean(parallel_ratio, weights),
                "mean_field_perpendicular_r2_over_4Dperp_t": weighted_mean(perpendicular_ratio, weights),
            },
            "interpretation": (
                "Ratios near one are expected for a uniform field and large statistics. A spatially "
                "varying periodic MHD field can rotate the local diffusion tensor, so mean-field "
                "parallel/perpendicular ratios are diagnostics rather than exact invariants."
            ),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="x3_config.yaml")
    parser.add_argument("--background", default=None)
    parser.add_argument("--transport", default=None)
    parser.add_argument("--input", action="append", default=[])
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    background_path = resolve_path(config, args.background or config["project"]["background_file"])
    transport_path = resolve_path(config, args.transport or config["project"]["transport_file"])
    background = load_json(background_path, "x3-mhd-background-v2")
    transport_summary = None
    if transport_path.exists():
        transport_summary = load_json(transport_path, "x3-mirror-scattering-transport-v2")
    report = {
        "schema": "x3-validation-v2",
        "configuration": validate_config(config, background, transport_summary),
        "endpoints": [
            validate_endpoints(Path(value).resolve(), config, background)
            for value in args.input
        ],
    }
    output = resolve_path(
        config, args.output or str(Path(config["project"]["output_dir"]) / "validation.json")
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(report, indent=2))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
