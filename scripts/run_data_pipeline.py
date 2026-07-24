"""Run the data pipeline stages EXP-P0 -> EXP-P5 (skeleton).

This is a thin CLI wrapper. It parses arguments and will dispatch into
``fnb.data.*`` once those modules are implemented (Milestone 2). No research
logic lives here.

Examples (once implemented):
    python scripts/run_data_pipeline.py --stage all
    python scripts/run_data_pipeline.py --stage P1a --config configs/datasets.yaml
"""

from __future__ import annotations

import argparse

# Ordered data-pipeline stages (see MASTER_IMPLEMENTATION_GUIDE.md, Milestone 2).
STAGES = ["P0", "P1a", "P1b", "P2", "P3", "P4", "P5"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fnb-data-pipeline",
        description="Run the fnb data pipeline (EXP-P0 -> EXP-P5).",
    )
    parser.add_argument(
        "--stage",
        default="all",
        choices=[*STAGES, "all"],
        help="Which pipeline stage to run (default: all).",
    )
    parser.add_argument(
        "--config-dir",
        default="configs",
        help="Directory holding the frozen YAML config mirrors.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Global seed for any stochastic step (default: 42).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code."""
    args = build_parser().parse_args(argv)
    # TODO(M2): dispatch into fnb.data.{acquire,binarize,preprocess,dedup,splits,stats,xdedup}.
    raise NotImplementedError(
        f"Data pipeline stage '{args.stage}' is not implemented yet (Milestone 2)."
    )


if __name__ == "__main__":
    raise SystemExit(main())
