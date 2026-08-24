#!/usr/bin/env python3
"""Build an optional mirror+scattering test-particle comparison table."""

from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from x3_common import C_CM_S, KYR_S, PC_CM, load_config, resolve_path


# Complete magnetic-field distribution, Table 3 of Barreto-Mota et al. 2025.
# D_tilde is their saturated <(z-z0)^2>/t in code units. It already contains
# magnetic mirroring and pitch-angle scattering in the same MHD realization.
LUCAS_RL_OVER_L0 = np.array([0.03, 0.06, 0.10], dtype=float)
LUCAS_D_TILDE = np.array([0.005, 0.006, 0.008], dtype=float)


def log_powerlaw_fit(energy_tev: np.ndarray, diffusion: np.ndarray, reference_pev: float) -> dict:
    slope, intercept = np.polyfit(np.log(energy_tev), np.log(diffusion), 1)
    reference_tev = reference_pev * 1e3
    d0 = float(np.exp(intercept + slope * np.log(reference_tev)))
    fitted = d0 * (energy_tev / reference_tev) ** slope
    fractional = fitted / diffusion - 1.0
    return {
        "d0_cm2_s_at_reference": d0,
        "reference_energy_pev": reference_pev,
        "delta": float(slope),
        "maximum_absolute_fractional_fit_residual": float(np.max(np.abs(fractional))),
    }


def lucas_model(background: dict, config: dict) -> tuple[np.ndarray, np.ndarray, dict]:
    transport = config["transport"]
    energy_tev = np.geomspace(
        float(transport["energy_grid_min_tev"]),
        float(transport["energy_grid_max_pev"]) * 1e3,
        int(transport["energy_grid_bins"]),
    )
    box_pc = float(background["box_size_pc"])
    # Equation 9 of the paper: rL[pc] = 1.084 E[PeV] / B[uG] for v_perp~c.
    b_uG = float(background["magnetic_field"]["rms_strength"])
    rl_over_l0 = 1.084 * (energy_tev / 1e3) / b_uG / box_pc
    log_d = np.interp(
        np.log(rl_over_l0),
        np.log(LUCAS_RL_OVER_L0),
        np.log(LUCAS_D_TILDE),
    )
    # np.interp clamps; replace both tails with the nearest log-log slope.
    low_slope = math.log(LUCAS_D_TILDE[1] / LUCAS_D_TILDE[0]) / math.log(
        LUCAS_RL_OVER_L0[1] / LUCAS_RL_OVER_L0[0]
    )
    high_slope = math.log(LUCAS_D_TILDE[-1] / LUCAS_D_TILDE[-2]) / math.log(
        LUCAS_RL_OVER_L0[-1] / LUCAS_RL_OVER_L0[-2]
    )
    below = rl_over_l0 < LUCAS_RL_OVER_L0[0]
    above = rl_over_l0 > LUCAS_RL_OVER_L0[-1]
    log_d[below] = math.log(LUCAS_D_TILDE[0]) + low_slope * np.log(
        rl_over_l0[below] / LUCAS_RL_OVER_L0[0]
    )
    log_d[above] = math.log(LUCAS_D_TILDE[-1]) + high_slope * np.log(
        rl_over_l0[above] / LUCAS_RL_OVER_L0[-1]
    )
    d_parallel = np.exp(log_d) * box_pc * PC_CM * C_CM_S
    calibration_energy_tev = LUCAS_RL_OVER_L0 * box_pc * b_uG / 1.084 * 1e3
    calibration_diffusion = LUCAS_D_TILDE * box_pc * PC_CM * C_CM_S
    details = {
        "paper_calibration": {
            "rl_over_l0": LUCAS_RL_OVER_L0.tolist(),
            "dimensionless_D_parallel": LUCAS_D_TILDE.tolist(),
            "energy_TeV_after_current_scaling": calibration_energy_tev.tolist(),
            "D_parallel_cm2_s_after_current_scaling": calibration_diffusion.tolist(),
            "box_size_pc": box_pc,
            "magnetic_strength_used_uG": b_uG,
        },
        "powerlaw_fit_points_energy_TeV": calibration_energy_tev.tolist(),
        "powerlaw_fit_points_D_parallel_cm2_s": calibration_diffusion.tolist(),
        "fraction_of_output_grid_inside_direct_calibration": float(
            np.mean((rl_over_l0 >= LUCAS_RL_OVER_L0[0]) & (rl_over_l0 <= LUCAS_RL_OVER_L0[-1]))
        ),
        "extrapolation_note": (
            "Only 0.03<=rL/L0<=0.10 is directly calibrated. Values outside use endpoint "
            "log-log slopes; the CRPropa SDE run uses one fitted power law over all energies."
        ),
    }
    return energy_tev, d_parallel, {**details, "rl_over_l0": rl_over_l0}


def external_csv_model(config: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, dict]:
    path = resolve_path(config, config["transport"]["external_csv"])
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(row)
    if not rows:
        raise ValueError(f"No rows in {path}")
    energy = np.array([float(row["energy_TeV"]) for row in rows])
    parallel = np.array([float(row["D_parallel_cm2_s"]) for row in rows])
    perpendicular = None
    if "D_perp_cm2_s" in rows[0] and rows[0]["D_perp_cm2_s"]:
        perpendicular = np.array([float(row["D_perp_cm2_s"]) for row in rows])
    return energy, parallel, perpendicular, {"external_csv": str(path)}


def build_transport(background: dict, config: dict) -> tuple[dict, np.ndarray]:
    transport = config["transport"]
    model = str(transport["coefficient_model"])
    external_perp = None
    if model == "lucas2025_complete_field":
        energy, parallel, details = lucas_model(background, config)
        fit_energy = np.asarray(details.pop("powerlaw_fit_points_energy_TeV"))
        fit_diffusion = np.asarray(details.pop("powerlaw_fit_points_D_parallel_cm2_s"))
        rl_over_l0 = np.asarray(details.pop("rl_over_l0"))
    elif model == "external_csv":
        energy, parallel, external_perp, details = external_csv_model(config)
        fit_energy, fit_diffusion = energy, parallel
        box_pc = float(background["box_size_pc"])
        b_uG = float(background["magnetic_field"]["rms_strength"])
        rl_over_l0 = 1.084 * (energy / 1e3) / b_uG / box_pc
    else:
        raise ValueError(f"Unsupported coefficient_model: {model}")

    ma = float(background["turbulence"]["alfvenic_mach"])
    perpendicular_model = str(transport["perpendicular_model"])
    if external_perp is not None:
        perpendicular = external_perp
        epsilon = float(np.exp(np.mean(np.log(perpendicular / parallel))))
        perpendicular_note = "read from external CSV; CRPropa uses its geometric-mean constant ratio"
    elif perpendicular_model == "ma4":
        epsilon = ma**4
        perpendicular = epsilon * parallel
        perpendicular_note = "D_perp/D_parallel=MA^4 closure; not measured by Table 3"
    elif perpendicular_model == "constant":
        epsilon = float(transport["perpendicular_epsilon"])
        perpendicular = epsilon * parallel
        perpendicular_note = "user-supplied constant D_perp/D_parallel"
    else:
        raise ValueError(f"Unsupported perpendicular_model: {perpendicular_model}")
    if not 0 < epsilon <= 1:
        raise ValueError("DiffusionSDE requires 0 < D_perp/D_parallel <= 1")

    fit = log_powerlaw_fit(
        fit_energy, fit_diffusion, float(transport["reference_energy_pev"])
    )
    ages = [float(value) for value in transport["ages_kyr"]]
    rms = {}
    for age in ages:
        trace_diffusion = fit["d0_cm2_s_at_reference"] * (1.0 + 2.0 * epsilon)
        rms[str(age)] = math.sqrt(2.0 * trace_diffusion * age * KYR_S) / PC_CM

    summary = {
        "schema": "x3-mirror-scattering-transport-v2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "coefficient_model": model,
        "physical_interpretation": (
            "Optional test-particle comparison containing mirror diffusion and pitch-angle "
            "scattering. It is not the coefficient source for the CRPropa production run."
        ),
        "crpropa_powerlaw": {**fit, "sde_epsilon": epsilon},
        "perpendicular_note": perpendicular_note,
        "not_used_by_crpropa_run": True,
        "rms_displacement_pc_at_reference_energy": rms,
        "background_box_size_pc": float(background["box_size_pc"]),
        "background_Ms": float(background["turbulence"]["sonic_mach"]),
        "background_MA": ma,
        "background_B_rms_uG": float(background["magnetic_field"]["rms_strength"]),
        "details": details,
        "warnings": [
            "This diagnostic file is not read by crpropa_run_x3.py.",
            "DiffusionSDE does not calculate mirror/scattering coefficients from B(x).",
            "The Lucas calibration has only three rL/L0 points and measures D_parallel only.",
            "Applying one periodic realization to 1 kpc assumes statistical homogeneity over that region.",
        ],
    }
    table = np.column_stack((energy, rl_over_l0, parallel, perpendicular))
    return summary, table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="x3_config.yaml")
    parser.add_argument("--background", default=None)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    background_path = resolve_path(config, args.background or config["project"]["background_file"])
    with background_path.open("r", encoding="utf-8") as handle:
        background = json.load(handle)
    if background.get("schema") != "x3-mhd-background-v2":
        raise ValueError(f"Unsupported background schema in {background_path}")
    output = resolve_path(config, args.output or config["project"]["transport_file"])
    output.parent.mkdir(parents=True, exist_ok=True)
    summary, table = build_transport(background, config)
    table_path = output.with_suffix(".csv")
    np.savetxt(
        table_path,
        table,
        delimiter=",",
        header="energy_TeV,rL_over_L0,D_parallel_cm2_s,D_perp_cm2_s",
        comments="",
    )
    summary["table_csv"] = table_path.name
    with output.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    fit = summary["crpropa_powerlaw"]
    print(f"Wrote {output} and {table_path}")
    print(
        "D_parallel={:.3e}(E/{:g} PeV)^{:.3f} cm^2/s; epsilon={:.4f}".format(
            fit["d0_cm2_s_at_reference"], fit["reference_energy_pev"], fit["delta"], fit["sde_epsilon"]
        )
    )
    print("Diagnostic only: crpropa_run_x3.py does not read this fitted coefficient.")


if __name__ == "__main__":
    main()
