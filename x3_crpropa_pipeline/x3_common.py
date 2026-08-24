#!/usr/bin/env python3
"""Shared numerical helpers for the X-3 / Cygnus Bubble pipeline."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import yaml


PC_CM = 3.0856775814913673e18
KPC_CM = 1.0e3 * PC_CM
KYR_S = 1.0e3 * 365.25 * 86400.0
GEV_ERG = 1.602176634e-3
C_CM_S = 2.99792458e10


def trapezoid(y, x=None, dx: float = 1.0, axis: int = -1):
    """NumPy 1.x/2.x compatible trapezoidal integration."""
    function = getattr(np, "trapezoid", None)
    if function is None:
        function = np.trapz
    return function(y, x=x, dx=dx, axis=axis)


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path).resolve()
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must be a mapping: {path}")
    config["_config_path"] = str(path)
    config["_config_dir"] = str(path.parent)
    return config


def resolve_path(config: dict[str, Any], value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = Path(config["_config_dir"]) / path
    return path.resolve()


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def turbulence_correlation_length_pc(lmin_pc: float, lmax_pc: float, s_index: float) -> float:
    """CRPropa SimpleTurbulenceSpectrum correlation length, in pc."""
    ratio = lmin_pc / lmax_pc
    numerator = 1.0 - ratio**s_index
    denominator = 1.0 - ratio ** (s_index - 1.0)
    return 0.5 * lmax_pc * (s_index - 1.0) / s_index * numerator / denominator


def solve_lmax_pc(lmin_pc: float, target_lc_pc: float, s_index: float) -> float:
    """Solve for lmax such that CRPropa reports the requested correlation length."""
    if lmin_pc <= 0 or target_lc_pc <= 0 or s_index <= 1:
        raise ValueError("lmin, target correlation length, and s_index must be positive; s_index > 1")
    low = max(lmin_pc * (1.0 + 1e-9), target_lc_pc)
    high = max(10.0 * target_lc_pc, 2.0 * low)
    while turbulence_correlation_length_pc(lmin_pc, high, s_index) < target_lc_pc:
        high *= 2.0
    for _ in range(100):
        middle = 0.5 * (low + high)
        if turbulence_correlation_length_pc(lmin_pc, middle, s_index) < target_lc_pc:
            low = middle
        else:
            high = middle
    return 0.5 * (low + high)


def diffusion_cm2_s(energy_gev: np.ndarray | float, d0_cm2_s: float, e0_gev: float, delta: float):
    return d0_cm2_s * (np.asarray(energy_gev) / e0_gev) ** delta


def crpropa_diffusion_scale(d0_cm2_s: float, e0_gev: float, delta: float) -> float:
    """Scale used by DiffusionSDE: D=scale*6.1e24 m^2/s*(R/4 GV)^alpha."""
    d0_m2_s = d0_cm2_s * 1e-4
    return d0_m2_s / (6.1e24 * (e0_gev / 4.0) ** delta)


def gas_density_cm3(z_pc: np.ndarray | float, gas: dict[str, Any]):
    model = gas["model"]
    z = np.asarray(z_pc)
    if model in {"uniform", "analytic_uniform"}:
        return np.full_like(z, float(gas["uniform_density_cm3"]), dtype=float)
    if model == "exponential_vertical":
        return float(gas["midplane_density_cm3"]) * np.exp(
            -np.abs(z) / float(gas["scale_height_pc"])
        )
    if model == "mhd_periodic":
        raise ValueError("mhd_periodic density requires the prepared background grid and xyz positions")
    raise ValueError(f"Unsupported gas model: {model}")


def source_shape(energy_erg: np.ndarray | float, source: dict[str, Any]):
    energy = np.asarray(energy_erg)
    reference = float(source["reference_energy_tev"]) * 1e3 * GEV_ERG
    cutoff = float(source["cutoff_pev"]) * 1e6 * GEV_ERG
    return (energy / reference) ** (-float(source["index"])) * np.exp(-energy / cutoff)


def source_q0_per_erg_s(source: dict[str, Any]) -> float:
    """Normalize Q(E)=Q0*shape(E) to the configured proton power."""
    emin = float(source["normalization_emin_tev"]) * 1e3 * GEV_ERG
    emax = float(source["normalization_emax_pev"]) * 1e6 * GEV_ERG
    grid = np.geomspace(emin, emax, 32768)
    integral = trapezoid(grid * source_shape(grid, source), x=grid)
    return float(source["proton_power_erg_s"]) / integral


def particle_weights(
    energy_gev: np.ndarray,
    delta_t_kyr: float,
    n_samples: int,
    source: dict[str, Any],
) -> np.ndarray:
    """Importance weights for log-uniform energy samples and one injection-time slice."""
    energy_erg = np.asarray(energy_gev) * GEV_ERG
    emin = float(source["sample_emin_tev"]) * 1e3
    emax = float(source["sample_emax_pev"]) * 1e6
    log_width = math.log(emax / emin)
    q0 = source_q0_per_erg_s(source)
    q_per_erg_s = q0 * source_shape(energy_erg, source)
    sampling_pdf_per_erg = 1.0 / (energy_erg * log_width)
    return q_per_erg_s / sampling_pdf_per_erg * (delta_t_kyr * KYR_S) / n_samples


def angular_radius_to_pc(distance_kpc: float, angle_deg: float) -> float:
    return 1.0e3 * distance_kpc * math.tan(math.radians(angle_deg))


def galactocentric_line(config: dict[str, Any], n_steps: int) -> np.ndarray:
    """Earth-to-source path in a right-handed Galactic Cartesian frame, in kpc."""
    obs = config["observation"]
    distance = float(obs["distance_kpc"])
    longitude = math.radians(float(obs["galactic_longitude_deg"]))
    latitude = math.radians(float(obs["galactic_latitude_deg"]))
    earth = np.array(
        [-float(obs["sun_galactocentric_radius_kpc"]), 0.0, float(obs.get("sun_height_kpc", 0.0))]
    )
    direction = np.array(
        [math.cos(latitude) * math.cos(longitude), math.cos(latitude) * math.sin(longitude), math.sin(latitude)]
    )
    source = earth + distance * direction
    fraction = np.linspace(0.0, 1.0, n_steps)
    return earth[None, :] + fraction[:, None] * (source - earth)[None, :]


def json_from_npz(data: np.lib.npyio.NpzFile, key: str = "metadata_json") -> dict[str, Any]:
    raw = data[key]
    if isinstance(raw, np.ndarray):
        raw = raw.item()
    return json.loads(str(raw))
