#!/usr/bin/env python3
"""Generate slurm/commands.txt: one main.py invocation per round (array task)."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Write commands.txt for sbatch array jobs.")
    parser.add_argument(
        "--rounds",
        type=int,
        required=True,
        help="Number of rounds; creates lines for --round 1 .. N.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output path (default: slurm/commands.txt next to this script).",
    )
    args = parser.parse_args()

    slurm_dir = Path(__file__).resolve().parent
    project_root = slurm_dir.parent
    main_py = project_root / "main.py"
    out = args.output if args.output is not None else slurm_dir / "commands.txt"

    lines = []
    for r in range(1, args.rounds + 1):
        lines.append(
            f"python3 {main_py} --round {r} --slurm-task\n",
        )

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(lines), encoding="utf-8")
    print(out.resolve())


if __name__ == "__main__":
    main()
