"""Dispatch any EXP-* experiment by id (skeleton).

This is a thin CLI wrapper. It parses arguments and will dispatch into
``fnb.experiments.*`` once those modules are implemented (Milestones 5-7). No
research logic lives here.

Examples (once implemented):
    python scripts/run_experiment.py --exp EXP-1 --seed 13
    python scripts/run_experiment.py --exp EXP-A1 --seed 42
"""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fnb-run-experiment",
        description="Dispatch an fnb experiment by id (EXP-*).",
    )
    parser.add_argument(
        "--exp",
        required=True,
        help="Experiment id to run, e.g. EXP-1, EXP-P6, EXP-A1, EXP-C1, EXP-F0.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Global seed for this run (default: 42).",
    )
    parser.add_argument(
        "--config-dir",
        default="configs",
        help="Directory holding the frozen YAML config mirrors.",
    )
    parser.add_argument(
        "--output-dir",
        default="/kaggle/working",
        help="Where to write outputs on Kaggle (default: /kaggle/working).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code."""
    args = build_parser().parse_args(argv)
    # TODO(M5-M7): route args.exp to the matching fnb.experiments driver.
    raise NotImplementedError(
        f"Experiment '{args.exp}' is not implemented yet (Milestones 5-7)."
    )


if __name__ == "__main__":
    raise SystemExit(main())
