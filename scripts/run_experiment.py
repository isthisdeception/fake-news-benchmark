"""Dispatch any EXP-* experiment by id.

Thin CLI wrapper. Research logic lives in ``fnb.experiments.*``.

Examples:
    python scripts/run_experiment.py --exp EXP-1 --seed 13
    python scripts/run_experiment.py --exp EXP-1 --model BERT --seed 13
    python scripts/run_experiment.py --exp EXP-1 --all-seeds
    python scripts/run_experiment.py --exp EXP-1 --finalize-only
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fnb-run-experiment",
        description="Dispatch an fnb experiment by id (EXP-*).",
    )
    parser.add_argument(
        "--exp",
        required=True,
        help="Experiment id, e.g. EXP-1, EXP-P6, EXP-A1.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Single seed for this run.",
    )
    parser.add_argument(
        "--all-seeds",
        action="store_true",
        help="Run all primary seeds from configs/protocol.yaml.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Single model id (e.g. LR, BERT).",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="List of model ids to run.",
    )
    parser.add_argument(
        "--config-dir",
        default="configs",
        help="Directory holding the frozen YAML config mirrors.",
    )
    parser.add_argument(
        "--output-dir",
        default="/kaggle/working",
        help="Where to write outputs (results/, artifacts/, logs/).",
    )
    parser.add_argument(
        "--processed-dir",
        default="data/processed",
        help="Directory with {DSx}_vDEDUP / vCLEAN-N parquet files.",
    )
    parser.add_argument(
        "--splits-dir",
        default="data/splits",
        help="Directory with S-RAND *.idx split files.",
    )
    parser.add_argument(
        "--finalize-only",
        action="store_true",
        help="Only compute pairwise stats + best_encoder (no training).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run cells even if already present in the results CSV.",
    )
    parser.add_argument(
        "--no-finalize",
        action="store_true",
        help="Skip auto-finalize after training cells.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device for encoders (default: cuda if available).",
    )
    return parser


def _resolve_models(args: argparse.Namespace) -> list[str] | None:
    if args.models:
        return [str(m).upper() for m in args.models]
    if args.model:
        return [str(args.model).upper()]
    return None


def _resolve_seeds(args: argparse.Namespace, config_dir: str) -> list[int]:
    from fnb.experiments.exp1_indist import primary_seeds

    if args.all_seeds:
        return primary_seeds(config_dir)
    if args.seed is not None:
        return [int(args.seed)]
    return primary_seeds(config_dir)


def _run_exp1(args: argparse.Namespace) -> int:
    from fnb.experiments.exp1_indist import finalize_exp1, run_exp1

    out = Path(args.output_dir)
    results_dir = out / "results"
    artifacts_dir = out / "artifacts"

    if args.finalize_only:
        paths = finalize_exp1(
            results_dir=results_dir,
            artifacts_dir=artifacts_dir,
            config_dir=args.config_dir,
        )
        best = Path(paths["best_encoder"]).read_text(encoding="utf-8").strip()
        print(json.dumps({
            "finalized": True,
            "best_encoder": best,
            "paths": {k: str(v) for k, v in paths.items()},
        }, indent=2))
        return 0

    summary = run_exp1(
        models=_resolve_models(args),
        seeds=_resolve_seeds(args, args.config_dir),
        processed_dir=args.processed_dir,
        splits_dir=args.splits_dir,
        output_dir=out,
        config_dir=args.config_dir,
        force=bool(args.force),
        finalize=not bool(args.no_finalize),
        device=args.device,
    )
    printable = {
        "ran": [f"{m}:seed{s}" for m, s in summary.get("ran", [])],
        "skipped": [f"{m}:seed{s}" for m, s in summary.get("skipped", [])],
        "progress": summary.get("progress"),
        "finalized": summary.get("finalized"),
        "best_encoder": summary.get("best_encoder"),
        "results_csv": summary.get("results_csv"),
    }
    print(json.dumps(printable, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code."""
    args = build_parser().parse_args(argv)
    exp = str(args.exp).upper().replace("_", "-")
    if exp in {"EXP-1", "EXP1"}:
        return _run_exp1(args)
    raise NotImplementedError(
        f"Experiment '{args.exp}' is not implemented yet (Milestones 5-7)."
    )


if __name__ == "__main__":
    raise SystemExit(main())
