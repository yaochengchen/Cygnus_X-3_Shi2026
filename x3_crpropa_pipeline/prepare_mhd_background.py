#!/usr/bin/env python3
"""Read, rescale, validate, and export a real periodic MHD snapshot."""

from __future__ import annotations

import argparse
import json
import math
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from mhd_io import (
    approximate_correlation_length_pc,
    detect_format,
    downsample,
    normalize_density,
    normalize_magnetic,
    read_field,
    reorder_spatial,
    vector_statistics,
    write_crpropa_binary,
)
from x3_common import angular_radius_to_pc, canonical_hash, load_config, resolve_path


PROTON_MASS_G = 1.67262192369e-24


def periodic_divergence_diagnostic(field: np.ndarray, box_size_pc: float, max_axis: int) -> dict:
    stride = max(1, math.ceil(max(field.shape[:3]) / max_axis))
    sample = np.asarray(field[::stride, ::stride, ::stride], dtype=np.float64)
    spacing = box_size_pc / np.asarray(sample.shape[:3], dtype=float)
    divergence = np.zeros(sample.shape[:3], dtype=np.float64)
    for axis in range(3):
        derivative = (np.roll(sample[..., axis], -1, axis=axis) - np.roll(sample[..., axis], 1, axis=axis))
        divergence += derivative / (2.0 * spacing[axis])
    brms = float(np.sqrt(np.mean(np.sum(sample**2, axis=-1))))
    characteristic = brms / float(np.mean(spacing))
    return {
        "diagnostic_stride": stride,
        "rms_divB_uG_per_pc": float(np.sqrt(np.mean(divergence**2))),
        "dimensionless_rms_divB_over_B_per_cell": float(np.sqrt(np.mean(divergence**2)) / characteristic),
    }


def prepare(config: dict, input_override: str | None = None) -> tuple[dict, dict[str, np.ndarray]]:
    mhd = config["mhd_input"]
    source = resolve_path(config, input_override or mhd["path"])
    if not source.exists():
        raise FileNotFoundError(
            f"MHD file not found: {source}. Edit mhd_input.path or pass --input."
        )
    fmt = detect_format(source, str(mhd.get("format", "auto")))
    fields = mhd["fields"]
    order = str(mhd["spatial_axis_order"])
    stride = int(mhd.get("downsample_stride", 1))

    density_raw = read_field(source, fmt, fields["density"], vector=False)
    magnetic_raw = read_field(source, fmt, fields["magnetic"], vector=True)
    assert density_raw is not None and magnetic_raw is not None
    density_raw = downsample(reorder_spatial(density_raw, order, vector=False), stride)
    magnetic_raw = downsample(reorder_spatial(magnetic_raw, order, vector=True), stride)
    assert density_raw is not None and magnetic_raw is not None
    if density_raw.shape != magnetic_raw.shape[:3]:
        raise ValueError(f"Density shape {density_raw.shape} != magnetic shape {magnetic_raw.shape}")
    if len(set(density_raw.shape)) != 1:
        raise ValueError(
            "Current CRPropa export expects a cubic grid. Use a cubic snapshot or crop it before this step."
        )

    scaling = mhd["physical_scaling"]
    box_size_pc = float(scaling["box_size_pc"])
    density, density_factor = normalize_density(
        density_raw,
        float(scaling["mean_target_density_cm3"]),
        float(scaling["density_floor_cm3"]),
    )
    magnetic, magnetic_factor, magnetic_stats = normalize_magnetic(
        magnetic_raw,
        float(scaling["target_magnetic_uG"]),
        str(scaling["magnetic_normalization"]),
    )
    del density_raw, magnetic_raw

    # Load the optional velocity only after the large raw density/B arrays have
    # been released. Velocity is used for diagnostics and is not exported.
    velocity_raw = read_field(
        source,
        fmt,
        fields.get("velocity", {"dataset": "", "optional": True}),
        vector=True,
    )
    if velocity_raw is not None:
        velocity_raw = downsample(reorder_spatial(velocity_raw, order, vector=True), stride)

    fft_max = int(mhd.get("diagnostics_fft_max_axis", 128))
    correlation = approximate_correlation_length_pc(magnetic, box_size_pc, fft_max)
    divergence = periodic_divergence_diagnostic(magnetic, box_size_pc, fft_max)
    injection_scale_pc = box_size_pc * float(scaling["injection_scale_fraction_of_box"])

    ma = float(scaling["alfvenic_mach"])
    ms = float(scaling["sonic_mach"])
    mean_mass_density = float(density.mean(dtype=np.float64)) * 1.4 * PROTON_MASS_G
    brms_gauss = float(magnetic_stats["rms_strength"]) * 1e-6
    alfven_speed_kms = brms_gauss / math.sqrt(4.0 * math.pi * mean_mass_density) / 1e5
    target_velocity_rms_kms = ma * alfven_speed_kms
    inferred_sound_speed_kms = target_velocity_rms_kms / ms

    velocity_details = {"present": velocity_raw is not None}
    if velocity_raw is not None:
        raw_stats = vector_statistics(velocity_raw)
        raw_fluctuation_rms = float(raw_stats["fluctuation_rms_strength"])
        if raw_fluctuation_rms <= 0:
            raise ValueError("Velocity field has zero fluctuation RMS")
        velocity_factor = target_velocity_rms_kms / raw_fluctuation_rms
        velocity_details.update(
            {
                "raw_statistics": raw_stats,
                "scale_factor_to_km_s": velocity_factor,
                "target_fluctuation_rms_km_s": target_velocity_rms_kms,
            }
        )

    obs = config["observation"]
    aperture_pc = angular_radius_to_pc(float(obs["distance_kpc"]), float(obs["aperture_deg"]))
    science_config = {key: value for key, value in config.items() if not key.startswith("_")}
    metadata = {
        "schema": "x3-mhd-background-v2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_file": str(source),
        "source_format": fmt,
        "source_fields": fields,
        "source_spatial_axis_order": order,
        "downsample_stride": stride,
        "grid_shape_xyz": list(density.shape),
        "box_size_pc": box_size_pc,
        "cell_size_pc": box_size_pc / density.shape[0],
        "periodic": True,
        "periodic_evaluation_note": (
            "The one MHD cube is evaluated periodically at arbitrary particle coordinates; "
            "no 1-kpc magnetic array is materialized."
        ),
        "density": {
            "mean_cm3": float(density.mean(dtype=np.float64)),
            "minimum_cm3": float(density.min()),
            "maximum_cm3": float(density.max()),
            "raw_to_physical_factor": density_factor,
            "target_is_total_pp_nucleon_density": True,
        },
        "magnetic_field": {
            **magnetic_stats,
            "units": "microgauss",
            "raw_to_physical_factor": magnetic_factor,
            "normalization": str(scaling["magnetic_normalization"]),
            "estimated_correlation_length_pc": correlation,
            "injection_scale_pc": injection_scale_pc,
            **divergence,
        },
        "turbulence": {
            "sonic_mach": ms,
            "alfvenic_mach": ma,
            "alfven_speed_km_s": alfven_speed_kms,
            "velocity_rms_km_s_required_to_preserve_MA": target_velocity_rms_kms,
            "sound_speed_km_s_required_to_preserve_Ms_and_MA": inferred_sound_speed_kms,
            "velocity_field": velocity_details,
        },
        "geometry": {
            "source_distance_kpc": float(obs["distance_kpc"]),
            "angular_aperture_deg": float(obs["aperture_deg"]),
            "projected_aperture_radius_pc": aperture_pc,
            "aperture_is_not_a_transport_boundary": True,
        },
        "config_sha256": canonical_hash(science_config),
        "generator": {"python": platform.python_version(), "numpy": np.__version__},
        "units_contract": {
            "density": "cm^-3 total target nucleons",
            "magnetic_binary_values": "microgauss float32",
            "length": "pc",
            "CRPropa_conversion": "binary values multiplied by crpropa.microgauss",
        },
    }
    return metadata, {"density_cm3": density, "magnetic_uG": magnetic}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="x3_config.yaml")
    parser.add_argument("--input", default=None, help="override mhd_input.path")
    parser.add_argument("--background", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    background_path = resolve_path(config, args.background or config["project"]["background_file"])
    background_path.parent.mkdir(parents=True, exist_ok=True)
    archive_path = background_path.with_suffix(".npz")
    binary_path = background_path.with_name(background_path.stem + "_B_uG.bin")

    metadata, arrays = prepare(config, args.input)
    np.savez_compressed(archive_path, density_cm3=arrays["density_cm3"])
    write_crpropa_binary(binary_path, arrays["magnetic_uG"])
    metadata["prepared_archive"] = archive_path.name
    metadata["crpropa_magnetic_binary"] = binary_path.name
    metadata["crpropa_binary_order"] = "C order over (x,y,z,component), component fastest"
    with background_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(f"Wrote {background_path}")
    print(f"Wrote {archive_path} (periodic gas grid)")
    print(f"Wrote {binary_path} (periodic CRPropa Grid3f values)")
    print(
        "shape={}, L0={:.3g} pc, Linj={:.3g} pc, <n>={:.3g} cm^-3, Brms={:.3g} uG".format(
            metadata["grid_shape_xyz"],
            metadata["box_size_pc"],
            metadata["magnetic_field"]["injection_scale_pc"],
            metadata["density"]["mean_cm3"],
            metadata["magnetic_field"]["rms_strength"],
        )
    )


if __name__ == "__main__":
    main()
