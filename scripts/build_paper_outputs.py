"""Build paper tables and figures from frozen result CSVs (skeleton).

This is a thin CLI wrapper. It parses arguments and will dispatch into
``fnb.analysis.figures`` and the evaluation stack once implemented (Milestone
8). No research logic lives here.

Examples (once implemented):
    python scripts/build_paper_outputs.py --results-dir results --out-dir results/paper
"""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fnb-build-paper",
        description="Assemble paper tables and figures for the fnb project.",
    )
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Directory holding the frozen result CSVs.",
    )
    parser.add_argument(
        "--out-dir",
        default="results/paper",
        help="Where to write assembled tables and figures.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code."""
    args = build_parser().parse_args(argv)
    # TODO(M8): build tables + figures from args.results_dir into args.out_dir.
    raise NotImplementedError(
        "Paper output assembly is not implemented yet (Milestone 8)."
    )


if __name__ == "__main__":
    raise SystemExit(main())
