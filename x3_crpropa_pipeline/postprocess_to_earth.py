#!/usr/bin/env python3
"""Convert proton endpoints to intrinsic pp gamma rays and the flux at Earth."""

from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mhd_io import load_prepared_background, sample_periodic_scalar

from x3_common import (
    C_CM_S,
    KPC_CM,
    angular_radius_to_pc,
    galactocentric_line,
    gas_density_cm3,
    json_from_npz,
    load_config,
    resolve_path,
    trapezoid,
)


MBARN_CM2 = 1.0e-27


def effective_proton_spectrum(
    data: np.lib.npyio.NpzFile,
    density_cm3: np.ndarray,
    mask: np.ndarray,
    energy_edges_gev: np.ndarray,
) -> np.ndarray:
    """Return d/dEp sum_i[w_i nH(x_i)] in particles cm^-3 GeV^-1."""
    weighted_counts, _ = np.histogram(
        data["energy_gev"][mask],
        bins=energy_edges_gev,
        weights=data["weight"][mask] * density_cm3[mask],
    )
    return weighted_counts / np.diff(energy_edges_gev)


def endpoint_density(
    data: np.lib.npyio.NpzFile,
    config: dict,
    background: dict | None,
    background_arrays: dict[str, np.ndarray] | None,
) -> np.ndarray:
    if config["gas"]["model"] != "mhd_periodic":
        return gas_density_cm3(data["z_pc"], config["gas"])
    if background is None or background_arrays is None:
        raise ValueError("mhd_periodic gas requires --background or project.background_file")
    positions = np.column_stack((data["x_pc"], data["y_pc"], data["z_pc"]))
    return sample_periodic_scalar(
        background_arrays["density_cm3"], positions, float(background["box_size_pc"])
    )


def hadronic_cross_section(config: dict, ep_gev: np.ndarray, egamma_gev: np.ndarray) -> np.ndarray:
    try:
        from aafragpy import get_cross_section, get_cross_section_Kafexhiu2014
    except ImportError as exc:
        raise RuntimeError("Post-processing requires aafragpy>=2.0.3") from exc

    model = config["hadronic"]["model"]
    if model == "aafrag_qgsjet":
        matrix, _, _ = get_cross_section(
            "gam", "p-p", E_primaries=ep_gev, E_secondaries=egamma_gev, outside_bounds="zeros"
        )
    elif model == "kafexhiu2014":
        matrix, _, _ = get_cross_section_Kafexhiu2014(
            E_primaries=ep_gev, E_secondaries=egamma_gev
        )
    else:
        raise ValueError(f"Unsupported hadronic model: {model}")
    matrix = np.asarray(matrix, dtype=float)
    return np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)


def gamma_flux_from_effective_protons(
    ep_gev: np.ndarray,
    egamma_gev: np.ndarray,
    cross_section_mb_per_gev: np.ndarray,
    effective_spectrum: np.ndarray,
    config: dict,
) -> np.ndarray:
    try:
        from aafragpy import get_spectrum
    except ImportError as exc:
        raise RuntimeError("Post-processing requires aafragpy>=2.0.3") from exc
    production_mb_per_gev_cm3 = get_spectrum(
        ep_gev, egamma_gev, cross_section_mb_per_gev, effective_spectrum
    )
    luminosity_per_gev_s = (
        production_mb_per_gev_cm3
        * MBARN_CM2
        * C_CM_S
        * float(config["hadronic"]["nuclear_enhancement_factor"])
    )
    distance_cm = float(config["observation"]["distance_kpc"]) * KPC_CM
    return luminosity_per_gev_s / (4.0 * math.pi * distance_cm**2)


def attenuation_tau(egamma_gev: np.ndarray, config: dict) -> tuple[np.ndarray, dict]:
    attenuation = config["attenuation"]
    if not attenuation["enabled"]:
        return np.zeros_like(egamma_gev), {"enabled": False, "fields_used": []}
    if attenuation.get("include_secondary_cascade", False):
        raise NotImplementedError(
            "This post-processor implements absorption only; secondary cascades need a separate photon/electron run"
        )
    try:
        import crpropa as crp
    except ImportError as exc:
        raise RuntimeError("Enabled gamma-gamma attenuation requires CRPropa") from exc

    processes: list[tuple[str, object]] = []
    if attenuation["use_cmb"]:
        processes.append(("CMB", crp.EMPairProduction(crp.CMB(), False)))
    isrf_error = None
    if attenuation["use_isrf_robitaille12"]:
        try:
            isrf = crp.ISRF_Robitaille12(None)
            processes.append(("ISRF_Robitaille12", crp.EMPairProduction(isrf, False)))
        except Exception as exc:  # CRPropa throws runtime_error when tables are absent.
            isrf_error = f"{type(exc).__name__}: {exc}"
            if attenuation["strict_isrf"]:
                raise RuntimeError("Could not initialize CRPropa Robitaille12 ISRF tables") from exc
    if not processes:
        raise RuntimeError("Attenuation is enabled but no photon field could be initialized")

    path_kpc = galactocentric_line(config, int(attenuation["path_steps"]))
    positions = [crp.Vector3d(*(point * crp.kpc)) for point in path_kpc]
    segment_m = np.linalg.norm(path_kpc[1] - path_kpc[0]) * crp.kpc
    tau_by_field: dict[str, np.ndarray] = {}
    for name, process in processes:
        values = np.empty_like(egamma_gev)
        for energy_index, energy in enumerate(egamma_gev):
            rates = np.array(
                [max(0.0, process.getRate(float(energy) * crp.GeV, position, 0.0)) for position in positions]
            )
            values[energy_index] = trapezoid(rates, dx=segment_m)
        tau_by_field[name] = values
    total = np.sum(np.stack(list(tau_by_field.values())), axis=0)
    details = {
        "enabled": True,
        "fields_used": list(tau_by_field),
        "isrf_initialization_error": isrf_error,
        "path_steps": len(path_kpc),
        "secondary_cascade_included": False,
        "tau_by_field": {name: values.tolist() for name, values in tau_by_field.items()},
    }
    return total, details


def fit_observation_scale(
    energy_tev: np.ndarray, flux_per_tev: np.ndarray, config: dict
) -> tuple[float, dict | None]:
    value = str(config["comparison"].get("observations_csv", "")).strip()
    if not value:
        return 1.0, None
    path = resolve_path(config, value)
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(
                (
                    float(row["energy_TeV"]),
                    float(row["flux_per_cm2_s_TeV"]),
                    float(row["sigma_per_cm2_s_TeV"]),
                )
            )
    observed = np.asarray(rows)
    fit_mask = observed[:, 0] >= float(config["comparison"]["fit_min_tev"])
    observed = observed[fit_mask]
    if len(observed) == 0:
        raise ValueError("No observation rows remain after fit_min_tev")
    model = np.exp(
        np.interp(
            np.log(observed[:, 0]),
            np.log(energy_tev),
            np.log(np.maximum(flux_per_tev, np.finfo(float).tiny)),
        )
    )
    inverse_variance = 1.0 / observed[:, 2] ** 2
    scale = float(np.sum(model * observed[:, 1] * inverse_variance) / np.sum(model**2 * inverse_variance))
    chi2 = float(np.sum(((observed[:, 1] - scale * model) / observed[:, 2]) ** 2))
    return scale, {"observations_csv": str(path), "points": len(observed), "chi2": chi2}


def interpolate_at(energy_tev: np.ndarray, values: np.ndarray, target_tev: float) -> float:
    return float(np.interp(math.log(target_tev), np.log(energy_tev), values))


def process_file(
    input_path: Path,
    config: dict,
    output_dir: Path,
    background: dict | None,
    background_arrays: dict[str, np.ndarray] | None,
) -> tuple[Path, Path, Path]:
    with np.load(input_path, allow_pickle=False) as data:
        proton_metadata = json_from_npz(data)
        source = config["source"]
        hadronic = config["hadronic"]
        obs = config["observation"]
        ep_edges = np.geomspace(
            float(source["sample_emin_tev"]) * 1e3,
            float(source["sample_emax_pev"]) * 1e6,
            int(hadronic["proton_energy_bins"]) + 1,
        )
        ep = np.sqrt(ep_edges[:-1] * ep_edges[1:])
        egamma = np.geomspace(
            float(hadronic["gamma_energy_min_tev"]) * 1e3,
            float(hadronic["gamma_energy_max_pev"]) * 1e6,
            int(hadronic["gamma_energy_bins"]),
        )
        cross_section = hadronic_cross_section(config, ep, egamma)

        density = endpoint_density(data, config, background, background_arrays)

        projected_pc = np.hypot(data["y_pc"], data["z_pc"])
        aperture_pc = angular_radius_to_pc(float(obs["distance_kpc"]), float(obs["aperture_deg"]))
        aperture_mask = projected_pc <= aperture_pc
        effective = effective_proton_spectrum(data, density, aperture_mask, ep_edges)
        intrinsic = gamma_flux_from_effective_protons(ep, egamma, cross_section, effective, config)

        theta_deg = np.degrees(np.arctan2(projected_pc, float(obs["distance_kpc"]) * 1e3))
        theta_edges = np.linspace(0.0, float(obs["aperture_deg"]), int(hadronic["radial_bins"]) + 1)
        radial_intrinsic = np.empty((len(theta_edges) - 1, len(egamma)))
        for radial_index, (lower, upper) in enumerate(zip(theta_edges[:-1], theta_edges[1:])):
            annulus = (theta_deg >= lower) & (theta_deg < upper)
            annular_effective = effective_proton_spectrum(data, density, annulus, ep_edges)
            radial_intrinsic[radial_index] = gamma_flux_from_effective_protons(
                ep, egamma, cross_section, annular_effective, config
            )

        weighted_mean_density = float(
            np.sum(data["weight"][aperture_mask] * density[aperture_mask])
            / np.sum(data["weight"][aperture_mask])
        )
        fraction_in_aperture = float(np.sum(aperture_mask) / len(aperture_mask))

    tau, attenuation_details = attenuation_tau(egamma, config)
    transmission = np.exp(-tau)
    earth_flux_per_gev = intrinsic * transmission
    radial_earth_per_gev = radial_intrinsic * transmission[None, :]
    energy_tev = egamma / 1e3
    flux_per_tev = earth_flux_per_gev * 1e3
    radial_per_tev = radial_earth_per_gev * 1e3
    scale, fit_details = fit_observation_scale(energy_tev, flux_per_tev, config)
    flux_per_tev *= scale
    intrinsic_per_tev = intrinsic * 1e3 * scale
    radial_per_tev *= scale

    band_min = float(config["hadronic"]["radial_band_min_tev"])
    band_max = float(config["hadronic"]["radial_band_max_pev"]) * 1e3
    band_mask = (energy_tev >= band_min) & (energy_tev <= band_max)
    solid_angle = 2.0 * math.pi * (
        np.cos(np.radians(theta_edges[:-1])) - np.cos(np.radians(theta_edges[1:]))
    )
    band_flux_annulus = trapezoid(
        radial_per_tev[:, band_mask], x=energy_tev[band_mask], axis=1
    )
    intensity = band_flux_annulus / solid_angle
    theta_center = 0.5 * (theta_edges[:-1] + theta_edges[1:])

    stem = input_path.stem.replace("protons_", "earth_")
    spectrum_csv = output_dir / f"{stem}_spectrum.csv"
    radial_csv = output_dir / f"{stem}_radial_profile.csv"
    summary_json = output_dir / f"{stem}_summary.json"
    figure_png = output_dir / f"{stem}.png"
    np.savetxt(
        spectrum_csv,
        np.column_stack((energy_tev, intrinsic_per_tev, tau, transmission, flux_per_tev)),
        delimiter=",",
        header="energy_TeV,intrinsic_flux_per_cm2_s_TeV,tau_gamma_gamma,transmission,earth_flux_per_cm2_s_TeV",
        comments="",
    )
    np.savetxt(
        radial_csv,
        np.column_stack((theta_center, theta_edges[:-1], theta_edges[1:], band_flux_annulus, intensity)),
        delimiter=",",
        header="theta_center_deg,theta_min_deg,theta_max_deg,band_flux_per_cm2_s,band_intensity_per_cm2_s_sr",
        comments="",
    )

    required_power = float(config["source"]["proton_power_erg_s"]) * scale
    summary = {
        "schema": "x3-earth-products-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "input": str(input_path),
        "proton_metadata": proton_metadata,
        "hadronic_model": config["hadronic"]["model"],
        "nuclear_enhancement_factor": float(config["hadronic"]["nuclear_enhancement_factor"]),
        "aperture_radius_pc": aperture_pc,
        "fraction_of_mc_endpoints_inside_projected_aperture": fraction_in_aperture,
        "weighted_mean_target_density_inside_aperture_cm3": weighted_mean_density,
        "target_density_model": config["gas"]["model"],
        "normalization_scale": scale,
        "required_proton_power_erg_s": required_power,
        "fit": fit_details,
        "attenuation": attenuation_details,
        "tau_400_TeV": interpolate_at(energy_tev, tau, 400.0),
        "tau_1_PeV": interpolate_at(energy_tev, tau, 1000.0),
        "radial_band_TeV": [band_min, band_max],
        "caveats": [
            "gamma-gamma absorption is integrated along the central X-3 sightline",
            "secondary electromagnetic cascades are not included",
            "proton pp losses are treated in the optically thin weighted-emissivity limit",
        ],
    }
    with summary_json.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")

    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    axes[0].loglog(energy_tev, energy_tev**2 * intrinsic_per_tev, label="intrinsic")
    axes[0].loglog(energy_tev, energy_tev**2 * flux_per_tev, label="at Earth")
    axes[0].axvline(400.0, color="0.5", ls="--", lw=1)
    axes[0].set(xlabel="Gamma-ray energy [TeV]", ylabel=r"$E^2 d\Phi/dE$ [TeV cm$^{-2}$ s$^{-1}$]")
    axes[0].legend(frameon=False)
    axes[0].grid(alpha=0.2, which="both")
    axes[1].plot(theta_center, intensity, marker="o")
    axes[1].set(
        xlabel="Angular offset [deg]",
        ylabel=r">400 TeV intensity [cm$^{-2}$ s$^{-1}$ sr$^{-1}$]",
        yscale="log",
    )
    axes[1].grid(alpha=0.2, which="both")
    figure.suptitle(f"X-3, age={proton_metadata['age_kyr']:g} kyr, {proton_metadata['engine']}")
    figure.tight_layout()
    figure.savefig(figure_png, dpi=180)
    plt.close(figure)
    return spectrum_csv, radial_csv, summary_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", help="protons_*.npz file(s)")
    parser.add_argument("--config", default="x3_config.yaml")
    parser.add_argument("--background", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--no-attenuation",
        action="store_true",
        help="debug only: skip gamma-gamma attenuation even if enabled in the configuration",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.no_attenuation:
        config["attenuation"]["enabled"] = False
    output_dir = resolve_path(config, args.output_dir or config["project"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    background = None
    background_arrays = None
    if config["gas"]["model"] == "mhd_periodic":
        background_path = resolve_path(
            config, args.background or config["project"]["background_file"]
        )
        background, background_arrays = load_prepared_background(background_path)
    for value in args.inputs:
        input_path = Path(value).resolve()
        products = process_file(input_path, config, output_dir, background, background_arrays)
        print("Wrote " + ", ".join(str(path) for path in products))


if __name__ == "__main__":
    main()
