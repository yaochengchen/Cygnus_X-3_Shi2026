#!/usr/bin/env python3
"""Create a tiny dimensionless periodic file for plumbing tests only."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/demo_mhd.npz")
    parser.add_argument("--size", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    n = int(args.size)
    coordinate = (np.arange(n) + 0.5) / n * 2.0 * np.pi
    x, y, z = np.meshgrid(coordinate, coordinate, coordinate, indexing="ij")
    density_xyz = np.exp(0.45 * (np.sin(x) + np.cos(y) + 0.5 * np.sin(z)))
    # Curl of a periodic vector potential plus a z guide field.
    bx = 0.35 * np.cos(y) * np.sin(z)
    by = 0.35 * np.cos(z) * np.sin(x)
    bz = 1.0 + 0.35 * np.cos(x) * np.sin(y)
    vx = np.sin(y) + 0.2 * np.cos(z)
    vy = np.sin(z) + 0.2 * np.cos(x)
    vz = np.sin(x) + 0.2 * np.cos(y)
    # Match the default incoming layout: (component,z,y,x), scalar (z,y,x).
    magnetic = np.transpose(np.stack((bx, by, bz), axis=0), (0, 3, 2, 1)).astype(np.float32)
    velocity = np.transpose(np.stack((vx, vy, vz), axis=0), (0, 3, 2, 1)).astype(np.float32)
    density = np.transpose(density_xyz, (2, 1, 0)).astype(np.float32)
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, density=density, magnetic_field=magnetic, velocity=velocity)
    print(f"Wrote {output}; this file is for I/O tests, not scientific runs")


if __name__ == "__main__":
    main()
