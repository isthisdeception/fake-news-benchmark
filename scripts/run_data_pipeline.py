"""Run the data pipeline stages EXP-P0 -> EXP-P5.

Thin CLI wrapper. Research logic lives in ``fnb.data.*``.

Examples:
    python scripts/run_data_pipeline.py --stage P0
    python scripts/run_data_pipeline.py --stage P0 --discover
    python scripts/run_data_pipeline.py --stage P1a
    python scripts/run_data_pipeline.py --stage P1b
    python scripts/run_data_pipeline.py --stage P2
    python scripts/run_data_pipeline.py --stage P3
    python scripts/run_data_pipeline.py --stage P4
    python scripts/run_data_pipeline.py --stage P5
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
        "--dedup-report-path",
        default="results/dedup_counts.csv",
        help="Where to write the EXP-P2 within-dataset dedup counts CSV.",
    )
    parser.add_argument(
        "--splits-dir",
        default="data/splits",
        help="Directory for EXP-P3 split index files (*.idx).",
    )
    parser.add_argument(
        "--split-applicability-path",
        default="results/split_applicability.csv",
        help="Where to write S-SRC / S-TEMP applicability notes.",
    )
    parser.add_argument(
        "--dataset-stats-path",
        default="results/dataset_stats.csv",
        help="Where to write EXP-P4 dataset_stats.csv.",
    )
    parser.add_argument(
        "--overlap-report-path",
        default="results/interdataset_overlap.csv",
        help="Where to write EXP-P5 interdataset_overlap.csv.",
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


def _run_p1b(args: argparse.Namespace) -> int:
    from fnb.data.preprocess import preprocess_all

    results = preprocess_all(
        processed_dir=args.output_dir,
        output_dir=args.output_dir,
        config_dir=args.config_dir,
        dataset_ids=args.datasets,
        nltk_report_path="results/nltk_resource_versions.json",
        environment_md="ENVIRONMENT.md",
    )
    # Pair classical/neural per dataset for readable logging
    by_ds: dict[str, list] = {}
    for r in results:
        by_ds.setdefault(r.dataset_id, []).append(r)
    print(f"Preprocessed {len(by_ds)} dataset(s) → vCLEAN-C + vCLEAN-N")
    for ds_id, tracks in by_ds.items():
        for r in tracks:
            path = r.output_path if r.output_path is not None else "?"
            print(f"  {ds_id} {r.version_tag}: n={r.n_rows} → {path}")
    print("\nWrote: results/nltk_resource_versions.json")
    return 0


def _run_p2(args: argparse.Namespace) -> int:
    from fnb.data.dedup import dedup_all

    results = dedup_all(
        processed_dir=args.output_dir,
        output_dir=args.output_dir,
        report_path=args.dedup_report_path,
        config_dir=args.config_dir,
        dataset_ids=args.datasets,
    )
    print(f"Deduped {len(results)} dataset(s) → vDEDUP")
    for r in results:
        c = r.counts
        print(
            f"  {c.dataset_id}: kept={c.n_kept}/{c.n_input} "
            f"(exact-{c.n_removed_exact}, near-{c.n_removed_near}, "
            f"{c.pct_removed:.2f}% removed; thr={c.near_duplicate_threshold})"
        )
        if r.output_path is not None:
            print(f"           → {r.output_path}")
    print(f"\nWrote: {args.dedup_report_path}")
    return 0


def _run_p3(args: argparse.Namespace) -> int:
    from fnb.data.splits import create_all_splits

    results, applicability = create_all_splits(
        processed_dir=args.output_dir,
        splits_dir=args.splits_dir,
        applicability_path=args.split_applicability_path,
        config_dir=args.config_dir,
        dataset_ids=args.datasets,
    )
    print(f"Wrote {len(results)} regime×seed split sets → {args.splits_dir}")
    for a in applicability:
        print(f"  {a.dataset_id} {a.regime}: {a.status} — {a.note}")
    print(f"\nWrote: {args.split_applicability_path}")
    return 0


def _run_p4(args: argparse.Namespace) -> int:
    from fnb.data.stats import compute_dataset_stats

    df = compute_dataset_stats(
        processed_dir=args.output_dir,
        splits_dir=args.splits_dir,
        report_path=args.dataset_stats_path,
        config_dir=args.config_dir,
        dataset_ids=args.datasets,
    )
    print(f"Wrote dataset stats: {len(df)} rows → {args.dataset_stats_path}")
    # Compact preview: full-corpus rows only
    full = df[df["split"] == "full"][
        ["dataset_id", "dataset_version", "n", "n_real", "n_fake", "fake_ratio", "mean_words"]
    ]
    print(full.to_string(index=False))
    return 0


def _run_p5(args: argparse.Namespace) -> int:
    from fnb.data.xdedup import xdedup_all

    results = xdedup_all(
        processed_dir=args.output_dir,
        splits_dir=args.splits_dir,
        report_path=args.overlap_report_path,
        config_dir=args.config_dir,
    )
    print(f"Cross-deduped {len(results)} directed pair(s) → vXDEDUP test indices")
    for r in results:
        c = r.counts
        flag = " [same-source]" if c.same_source_control else ""
        print(
            f"  {c.train_dataset_id}→{c.test_dataset_id}: "
            f"kept={c.n_test_after}/{c.n_test_before} "
            f"(removed={c.n_overlap_removed}, {c.pct_removed:.2f}%; "
            f"fake_ratio {c.fake_ratio_before:.4f}→{c.fake_ratio_after:.4f})"
            f"{flag}"
        )
        if r.index_path is not None:
            print(f"           → {r.index_path}")
    print(f"\nWrote: {args.overlap_report_path}")
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

    if args.stage in ("P1b", "all"):
        rc = _run_p1b(args)
        if args.stage == "P1b":
            return rc
        if rc != 0:
            return rc

    if args.stage in ("P2", "all"):
        rc = _run_p2(args)
        if args.stage == "P2":
            return rc
        if rc != 0:
            return rc

    if args.stage in ("P3", "all"):
        rc = _run_p3(args)
        if args.stage == "P3":
            return rc
        if rc != 0:
            return rc

    if args.stage in ("P4", "all"):
        rc = _run_p4(args)
        if args.stage == "P4":
            return rc
        if rc != 0:
            return rc

    if args.stage in ("P5", "all"):
        rc = _run_p5(args)
        if args.stage == "P5":
            return rc
        if rc != 0:
            return rc

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
