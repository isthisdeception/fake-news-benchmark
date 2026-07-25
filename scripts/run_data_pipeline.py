"""Run the data pipeline stages EXP-P0 -> EXP-P5.

Thin CLI wrapper. Research logic lives in ``fnb.data.*``.

Examples:
    python scripts/run_data_pipeline.py --stage P0
    python scripts/run_data_pipeline.py --stage P0 --discover
    python scripts/run_data_pipeline.py --stage P1a
    python scripts/run_data_pipeline.py --stage all
"""

from __future__ import annotations

import argparse

STAGES = ["P0", "P1a", "P1b", "P2", "P3", "P4", "P5"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fnb-data-pipeline",
        description="Run the fnb data pipeline (EXP-P0 -> EXP-P5).",
    )
    parser.add_argument(
        "--stage",
        default="P0",
        choices=[*STAGES, "all"],
        help="Which pipeline stage to run (default: P0).",
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
    parser.add_argument(
        "--input-root",
        default="/kaggle/input",
        help="Kaggle input root for attached datasets (default: /kaggle/input).",
    )
    parser.add_argument(
        "--hashes-path",
        default="data/SNAPSHOT_HASHES.txt",
        help="Where to write the snapshot hash ledger.",
    )
    parser.add_argument(
        "--manifest-path",
        default="results/kaggle_dataset_manifest.json",
        help="Where to write the resolved Kaggle dataset manifest.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed",
        help="Directory for processed parquet outputs (vBIN, vCLEAN-*, …).",
    )
    parser.add_argument(
        "--report-path",
        default="results/label_mapping_report.csv",
        help="Where to write the EXP-P1a label-mapping report CSV.",
    )
    parser.add_argument(
        "--discover",
        action="store_true",
        help="Only list attached /kaggle/input datasets (no hashing).",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Optional subset of dataset ids (e.g. DS1 DS2). Default: all.",
    )
    return parser


def _run_p0(args: argparse.Namespace) -> int:
    from fnb.data.acquire import acquire_snapshots, discover_report

    print(discover_report(args.input_root))
    if args.discover:
        return 0

    snapshots = acquire_snapshots(
        input_root=args.input_root,
        hashes_path=args.hashes_path,
        manifest_path=args.manifest_path,
        config_dir=args.config_dir,
        dataset_ids=args.datasets,
    )
    ok = [s for s in snapshots if s.status == "ok"]
    missing = [s for s in snapshots if s.status != "ok"]
    print(f"\nHashed {len(ok)}/{len(snapshots)} datasets.")
    for s in ok:
        print(f"  {s.dataset_id}: {s.sha256[:16]}...  ({s.n_files} files, {s.total_bytes} bytes)")
        print(f"           path={s.input_path}")
    for s in missing:
        print(f"  {s.dataset_id}: MISSING — {s.notes}")
    print(f"\nWrote: {args.hashes_path}")
    print(f"Wrote: {args.manifest_path}")
    # Partial success is useful on Kaggle while datasets are being attached.
    return 0 if ok and not missing else (0 if ok else 1)


def _run_p1a(args: argparse.Namespace) -> int:
    from fnb.data.binarize import binarize_all

    results = binarize_all(
        input_root=args.input_root,
        output_dir=args.output_dir,
        report_path=args.report_path,
        config_dir=args.config_dir,
        dataset_ids=args.datasets,
    )
    print(f"Binarized {len(results)} dataset(s) → vBIN")
    for r in results:
        rep = r.report
        dropped = rep.n_dropped_ambiguous + rep.n_dropped_missing_label
        print(
            f"  {rep.dataset_id}: kept={rep.n_kept}/{rep.n_raw} "
            f"(dropped={dropped}: ambiguous={rep.n_dropped_ambiguous}, "
            f"missing={rep.n_dropped_missing_label}); "
            f"real={rep.n_real} fake={rep.n_fake}"
        )
        if r.output_path is not None:
            print(f"           → {r.output_path}")
    print(f"\nWrote: {args.report_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code."""
    args = build_parser().parse_args(argv)

    if args.stage in ("P0", "all"):
        rc = _run_p0(args)
        if args.stage == "P0":
            return rc
        if rc != 0:
            return rc

    if args.stage in ("P1a", "all"):
        rc = _run_p1a(args)
        if args.stage == "P1a":
            return rc
        if rc != 0:
            return rc

    remaining = STAGES[2:] if args.stage == "all" else [args.stage]
    for stage in remaining:
        if stage in ("P0", "P1a"):
            continue
        raise NotImplementedError(
            f"Data pipeline stage '{stage}' is not implemented yet (Milestone 2, steps after S7)."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
