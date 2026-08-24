#!/usr/bin/env python3
"""List datasets, shapes, dtypes, and attributes in an incoming MHD file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mhd_io import inspect_container


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file")
    parser.add_argument("--format", default="auto", choices=("auto", "hdf5", "npz", "npy"))
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = inspect_container(Path(args.file).resolve(), args.format)
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        output = Path(args.output).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
        print(f"Wrote {output}")


if __name__ == "__main__":
    main()
