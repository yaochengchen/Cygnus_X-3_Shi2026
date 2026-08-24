#!/usr/bin/env python3
"""Run MHD preparation, optional transport comparison, CRPropa, and post-processing."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from x3_common import load_config, resolve_path


def run(command: list[str], cwd: Path) -> None:
    print("\n$ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="x3_config.yaml")
    parser.add_argument("--input", default=None, help="override mhd_input.path")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument(
        "--skip-transport-estimate",
        action="store_true",
        help="skip the optional test-particle mirror+scattering comparison",
    )
    parser.add_argument("--n-particles", type=int, default=None)
    parser.add_argument("--age-kyr", type=float, action="append", dest="ages")
    parser.add_argument("--no-postprocess", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.config).resolve().parent
    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    python = sys.executable

    prepare = [python, "prepare_mhd_background.py", "--config", str(config_path)]
    if args.input:
        prepare += ["--input", str(Path(args.input).resolve())]
    run(prepare, root)
    if not args.skip_transport_estimate:
        run([python, "estimate_transport.py", "--config", str(config_path)], root)
    run([python, "validate_x3.py", "--config", str(config_path)], root)
    if args.prepare_only:
        return

    transport = [python, "crpropa_run_x3.py", "--config", str(config_path)]
    if args.n_particles is not None:
        transport += ["--n-particles", str(args.n_particles)]
    for age in args.ages or []:
        transport += ["--age-kyr", str(age)]
    run(transport, root)

    output_dir = resolve_path(config, config["project"]["output_dir"])
    engine = config["transport"]["engine"]
    ages = args.ages or [float(value) for value in config["transport"]["ages_kyr"]]
    endpoints = [
        output_dir / f"protons_age_{f'{age:g}'.replace('.', 'p')}kyr_{engine}.npz"
        for age in ages
    ]
    validate = [python, "validate_x3.py", "--config", str(config_path)]
    for endpoint in endpoints:
        validate += ["--input", str(endpoint)]
    run(validate, root)
    if not args.no_postprocess:
        run(
            [python, "postprocess_to_earth.py", "--config", str(config_path)]
            + [str(path) for path in endpoints],
            root,
        )


if __name__ == "__main__":
    main()
