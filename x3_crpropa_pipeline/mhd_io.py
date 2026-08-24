#!/usr/bin/env python3
"""MHD snapshot adapters and periodic-grid helpers.

All prepared arrays use ``(x, y, z)`` spatial order. Vector components are the
last axis and are also ordered ``(x, y, z)``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def detect_format(path: Path, requested: str = "auto") -> str:
    if requested != "auto":
        return requested.lower()
    suffix = path.suffix.lower()
    if suffix in {".h5", ".hdf5", ".hdf"}:
        return "hdf5"
    if suffix == ".npz":
        return "npz"
    if suffix == ".npy":
        return "npy"
    raise ValueError(f"Cannot infer MHD format from {path}; set mhd_input.format")


def inspect_container(path: Path, requested_format: str = "auto") -> dict[str, Any]:
    """Return array names, shapes, dtypes, and HDF5 attributes without editing."""
    fmt = detect_format(path, requested_format)
    result: dict[str, Any] = {"path": str(path), "format": fmt, "arrays": []}
    if fmt == "hdf5":
        try:
            import h5py
        except ImportError as exc:
            raise RuntimeError("Reading HDF5 requires h5py") from exc
        with h5py.File(path, "r") as handle:
            result["root_attributes"] = {key: _jsonable(value) for key, value in handle.attrs.items()}

            def visitor(name: str, obj: Any) -> None:
                if isinstance(obj, h5py.Dataset):
                    result["arrays"].append(
                        {
                            "name": name,
                            "shape": list(obj.shape),
                            "dtype": str(obj.dtype),
                            "attributes": {key: _jsonable(value) for key, value in obj.attrs.items()},
                        }
                    )

            handle.visititems(visitor)
    elif fmt == "npz":
        with np.load(path, allow_pickle=False) as handle:
            for name in handle.files:
                array = handle[name]
                result["arrays"].append(
                    {"name": name, "shape": list(array.shape), "dtype": str(array.dtype)}
                )
    elif fmt == "npy":
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        result["arrays"].append({"name": "array", "shape": list(array.shape), "dtype": str(array.dtype)})
    else:
        raise ValueError(f"Unsupported MHD format: {fmt}")
    return result


def _jsonable(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _all_finite(array: np.ndarray, chunk_size: int = 4_000_000) -> bool:
    iterator = np.nditer(
        np.asarray(array),
        flags=["external_loop", "buffered", "zerosize_ok"],
        op_flags=["readonly"],
        buffersize=chunk_size,
        order="K",
    )
    for chunk in iterator:
        if not np.isfinite(chunk).all():
            return False
    return True


def _read_named(path: Path, fmt: str, name: str) -> np.ndarray:
    if fmt == "hdf5":
        try:
            import h5py
        except ImportError as exc:
            raise RuntimeError("Reading HDF5 requires h5py") from exc
        with h5py.File(path, "r") as handle:
            if name not in handle:
                raise KeyError(f"Dataset {name!r} not found in {path}")
            return np.asarray(handle[name])
    if fmt == "npz":
        with np.load(path, allow_pickle=False) as handle:
            if name not in handle.files:
                raise KeyError(f"Array {name!r} not found in {path}")
            return np.asarray(handle[name])
    if fmt == "npy":
        if name not in {"array", ""}:
            raise KeyError("A .npy file contains only one array; set dataset: array")
        return np.asarray(np.load(path, allow_pickle=False))
    raise ValueError(f"Unsupported MHD format: {fmt}")


def read_field(path: Path, fmt: str, specification: dict[str, Any], vector: bool) -> np.ndarray | None:
    """Read a scalar or vector using either a dataset or component datasets."""
    optional = bool(specification.get("optional", False))
    try:
        if "components" in specification:
            names = specification["components"]
            if len(names) != 3:
                raise ValueError("components must contain exactly three dataset names")
            array = np.stack([_read_named(path, fmt, str(name)) for name in names], axis=-1)
        else:
            array = _read_named(path, fmt, str(specification["dataset"]))
            if vector:
                axis = int(specification.get("vector_axis", -1))
                array = np.moveaxis(array, axis, -1)
    except (KeyError, OSError):
        if optional:
            return None
        raise
    expected_ndim = 4 if vector else 3
    if array.ndim != expected_ndim:
        raise ValueError(f"Expected {expected_ndim} dimensions, got {array.shape}")
    if vector and array.shape[-1] != 3:
        raise ValueError(f"Vector field must have three components, got {array.shape}")
    return np.asarray(array)


def reorder_spatial(array: np.ndarray, source_order: str, vector: bool) -> np.ndarray:
    order = source_order.lower().replace(" ", "")
    if sorted(order) != ["x", "y", "z"] or len(order) != 3:
        raise ValueError("mhd_input.spatial_axis_order must be a permutation of xyz")
    axes = [order.index(label) for label in "xyz"]
    if vector:
        axes.append(3)
    return np.transpose(array, axes)


def downsample(array: np.ndarray | None, stride: int) -> np.ndarray | None:
    if array is None:
        return None
    if stride < 1:
        raise ValueError("downsample_stride must be >= 1")
    slices = (slice(None, None, stride),) * 3
    if array.ndim == 4:
        slices += (slice(None),)
    # Keep this as a view. The physical-scaling step performs the one required
    # C-contiguous copy, avoiding an extra multi-GB copy for a 512^3 vector grid.
    return array[slices]


def vector_statistics(field: np.ndarray) -> dict[str, Any]:
    array = np.asarray(field)
    if array.ndim != 4 or array.shape[-1] != 3:
        raise ValueError("Expected vector field with shape (Nx,Ny,Nz,3)")
    count = int(np.prod(array.shape[:3]))
    component_sum = np.zeros(3, dtype=np.float64)
    for ix in range(array.shape[0]):
        component_sum += np.sum(array[ix], axis=(0, 1), dtype=np.float64)
    mean_vector = component_sum / count
    sum_magnitude = 0.0
    sum_magnitude_sq = 0.0
    sum_fluctuation_sq = 0.0
    for ix in range(array.shape[0]):
        chunk = np.asarray(array[ix], dtype=np.float64).reshape(-1, 3)
        magnitude_sq = np.sum(chunk * chunk, axis=1)
        sum_magnitude += float(np.sum(np.sqrt(magnitude_sq), dtype=np.float64))
        sum_magnitude_sq += float(np.sum(magnitude_sq, dtype=np.float64))
        difference = chunk - mean_vector
        sum_fluctuation_sq += float(np.sum(difference * difference, dtype=np.float64))
    return {
        "mean_vector": mean_vector.tolist(),
        "mean_vector_strength": float(np.linalg.norm(mean_vector)),
        "mean_strength": sum_magnitude / count,
        "rms_strength": math.sqrt(sum_magnitude_sq / count),
        "fluctuation_rms_strength": math.sqrt(sum_fluctuation_sq / count),
    }


def normalize_density(raw: np.ndarray, target_mean: float, floor: float) -> tuple[np.ndarray, float]:
    raw = np.asarray(raw)
    if not _all_finite(raw):
        raise ValueError("Density contains NaN or infinity")
    raw_mean = float(raw.mean(dtype=np.float64))
    if raw_mean <= 0:
        raise ValueError("Mean raw density must be positive")
    result = np.empty(raw.shape, dtype=np.float32, order="C")
    np.multiply(raw, np.float32(target_mean / raw_mean), out=result, casting="unsafe")
    np.maximum(result, np.float32(floor), out=result)
    # Re-normalize after applying a floor so the configured volume mean is exact.
    result *= np.float32(target_mean / float(result.mean(dtype=np.float64)))
    return result, target_mean / raw_mean


def normalize_magnetic(
    raw: np.ndarray, target_uG: float, normalization: str
) -> tuple[np.ndarray, float, dict[str, Any]]:
    raw = np.asarray(raw)
    if not _all_finite(raw):
        raise ValueError("Magnetic field contains NaN or infinity")
    stats = vector_statistics(raw)
    key_by_mode = {
        "rms_strength": "rms_strength",
        "mean_strength": "mean_strength",
        "mean_vector": "mean_vector_strength",
    }
    if normalization not in key_by_mode:
        raise ValueError(f"Unsupported magnetic_normalization: {normalization}")
    denominator = float(stats[key_by_mode[normalization]])
    if denominator <= 0:
        raise ValueError(f"Cannot normalize a zero magnetic field using {normalization}")
    factor = target_uG / denominator
    result = np.empty(raw.shape, dtype=np.float32, order="C")
    np.multiply(raw, np.float32(factor), out=result, casting="unsafe")
    return result, factor, vector_statistics(result)


def approximate_correlation_length_pc(field_uG: np.ndarray, box_size_pc: float, max_axis: int) -> float:
    """Estimate a 1D integral scale from the 3D periodic autocorrelation.

    The diagnostic copy may be strided for memory control. The production grid
    remains untouched. This statistic is a check, not the transport closure.
    """
    field = np.asarray(field_uG, dtype=np.float64)
    stride = max(1, math.ceil(max(field.shape[:3]) / max_axis))
    sample = field[::stride, ::stride, ::stride]
    sample = sample - sample.reshape(-1, 3).mean(axis=0)
    power = np.zeros(sample.shape[:3], dtype=np.float64)
    for component in range(3):
        transform = np.fft.fftn(sample[..., component])
        power += np.abs(transform) ** 2
    corr = np.fft.ifftn(power).real
    if corr.flat[0] <= 0:
        return float("nan")
    corr /= corr.flat[0]
    dx = box_size_pc / np.asarray(sample.shape[:3], dtype=float)
    lengths = []
    for axis in range(3):
        index = [0, 0, 0]
        line = []
        for offset in range(sample.shape[axis] // 2 + 1):
            index[axis] = offset
            line.append(corr[tuple(index)])
        line = np.asarray(line)
        nonpositive = np.flatnonzero(line <= 0)
        stop = int(nonpositive[0]) if len(nonpositive) else len(line)
        stop = max(stop, 2)
        lengths.append(float(np.trapz(np.maximum(line[:stop], 0.0), dx=dx[axis])))
    return float(np.mean(lengths))


def write_crpropa_binary(path: Path, magnetic_uG: np.ndarray) -> None:
    """Write raw float32 Vector3 values in CRPropa Grid3f index order."""
    array = np.asarray(magnetic_uG, dtype="<f4", order="C")
    if array.ndim != 4 or array.shape[-1] != 3:
        raise ValueError("magnetic_uG must have shape (Nx,Ny,Nz,3)")
    array.tofile(path)


def load_prepared_background(background_json: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    with background_json.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    if metadata.get("schema") != "x3-mhd-background-v2":
        raise ValueError(f"Unsupported background schema in {background_json}")
    archive = Path(metadata["prepared_archive"])
    if not archive.is_absolute():
        archive = (background_json.parent / archive).resolve()
    with np.load(archive, allow_pickle=False) as handle:
        arrays = {name: np.asarray(handle[name]) for name in handle.files if name != "metadata_json"}
    return metadata, arrays


def sample_periodic_scalar(grid: np.ndarray, xyz_pc: np.ndarray, box_size_pc: float) -> np.ndarray:
    """Periodic trilinear interpolation at source-centred coordinates."""
    values = np.asarray(grid)
    points = np.asarray(xyz_pc, dtype=np.float64)
    if values.ndim != 3 or points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Expected scalar grid (Nx,Ny,Nz) and points (N,3)")
    shape = np.asarray(values.shape, dtype=np.int64)
    # Grid samples represent cell centres: x_i=-L/2+(i+1/2)L/N.
    coordinate = ((points + 0.5 * box_size_pc) / box_size_pc * shape - 0.5) % shape
    lower = np.floor(coordinate).astype(np.int64)
    fraction = coordinate - lower
    upper = (lower + 1) % shape
    lower %= shape
    output = np.zeros(len(points), dtype=np.float64)
    for bx in (0, 1):
        ix = np.where(bx, upper[:, 0], lower[:, 0])
        wx = fraction[:, 0] if bx else 1.0 - fraction[:, 0]
        for by in (0, 1):
            iy = np.where(by, upper[:, 1], lower[:, 1])
            wy = fraction[:, 1] if by else 1.0 - fraction[:, 1]
            for bz in (0, 1):
                iz = np.where(bz, upper[:, 2], lower[:, 2])
                wz = fraction[:, 2] if bz else 1.0 - fraction[:, 2]
                output += wx * wy * wz * values[ix, iy, iz]
    return output
