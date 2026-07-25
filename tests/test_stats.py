"""Tests for EXP-P4 dataset statistics."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

from fnb.data.splits import stratified_random_split, write_split_files
from fnb.data.stats import compute_dataset_stats, summarize_subset


def test_summarize_subset_counts_and_lengths():
    df = pd.DataFrame(
        {
            "title": ["A", "BB"],
            "text": ["one two", "three"],
            "label": [0, 1],
        }
    )
    s = summarize_subset(df)
    assert s["n"] == 2
    assert s["n_real"] == 1
    assert s["n_fake"] == 1
    assert s["fake_ratio"] == 0.5
    assert s["mean_words"] > 0
    assert s["mean_chars"] > 0


def test_compute_dataset_stats_matches_split_sizes(tmp_path: Path):
    def _parquet_ok() -> bool:
        try:
            p = tmp_path / "_p.parquet"
            pd.DataFrame({"x": [1]}).to_parquet(p)
            return True
        except Exception:
            return False

    if not _parquet_ok():
        pytest.skip("parquet engine unavailable (Kaggle has pyarrow)")

    n = 40
    labels = [0, 1] * 20
    vbin = pd.DataFrame(
        {
            "uid": [f"u{i}" for i in range(n)],
            "title": [f"T{i}" for i in range(n)],
            "text": [f"body words {i} " * 3 for i in range(n)],
            "label": labels,
        }
    )
    # Simulate light dedup: drop 4 exact rows → 36 survivors with pre_dedup_index
    keep = list(range(0, n, 1))
    keep = [i for i in keep if i % 10 != 0]  # drop 0,10,20,30 → 36 rows
    vdedup = vbin.iloc[keep].reset_index(drop=True).copy()
    vdedup.insert(0, "pre_dedup_index", keep)

    processed = tmp_path / "processed"
    processed.mkdir()
    vbin.to_parquet(processed / "DS1_vBIN.parquet")
    vdedup.to_parquet(processed / "DS1_vDEDUP.parquet")

    splits_dir = tmp_path / "splits"
    splits = stratified_random_split(vdedup["label"].to_numpy(), seed=13)
    write_split_files("DS1", "S-RAND", 13, splits, splits_dir=splits_dir)

    cfg_dir = tmp_path / "configs"
    cfg_dir.mkdir()
    protocol = {
        "protocol_version": "v1.0",
        "status": "FROZEN",
        "frozen_date": "2026-07-25",
        "primary_endpoint": "x",
        "research_questions": {"primary": {"RQ-A": "a"}, "secondary": {"RQ-D": "d"}},
        "compute_budget": {
            "total_gpu_hours": 120,
            "rq_f_cap_gpu_hours": 20,
            "accelerator": "T4",
            "cut_order_if_over_budget": ["RQ-F"],
        },
        "seeds": {"primary": [13], "secondary": [13], "reporting": "mean"},
        "determinism": {
            "use_deterministic_algorithms": True,
            "cudnn_deterministic": True,
            "cudnn_benchmark": False,
            "cublas_workspace_config": ":4096:8",
            "set_pythonhashseed": True,
        },
        "out_of_scope": [],
    }
    datasets = {
        "label_space": {"real": 0, "fake": 1},
        "article_transfer_set": ["DS1"],
        "domain_shift_probe": [],
        "short_statement_track": [],
        "datasets": {
            "DS1": {
                "name": "WELFake",
                "role": "t",
                "text_type": "Article",
                "in_article_transfer": True,
                "kaggle_slug": "x",
                "kaggle_version": "1",
                "input_dirname": "x",
                "input_path": "x",
                "label_column": "label",
                "text_columns": {"title": "title", "body": "text"},
                "source_field_available": False,
                "license_note": "t",
            }
        },
        "version_tags": {"vDEDUP": "d"},
        "llm_contamination": {"probe": "x", "report_to": "y", "rule": "z"},
    }
    (cfg_dir / "protocol.yaml").write_text(yaml.dump(protocol), encoding="utf-8")
    (cfg_dir / "datasets.yaml").write_text(yaml.dump(datasets), encoding="utf-8")

    report = tmp_path / "dataset_stats.csv"
    df = compute_dataset_stats(
        processed_dir=processed,
        splits_dir=splits_dir,
        report_path=report,
        config_dir=cfg_dir,
        dataset_ids=["DS1"],
        seeds=[13],
    )
    assert report.exists()

    full_bin = df[(df.dataset_version == "vBIN") & (df.split == "full")].iloc[0]
    full_ded = df[(df.dataset_version == "vDEDUP") & (df.split == "full")].iloc[0]
    assert full_bin["n"] == n
    assert full_ded["n"] == len(vdedup)

    train_ded = df[
        (df.dataset_version == "vDEDUP") & (df.split == "train") & (df.seed == "13")
    ].iloc[0]
    assert train_ded["n"] == len(splits.train)
    assert train_ded["n_real"] + train_ded["n_fake"] == train_ded["n"]

    train_bin = df[(df.dataset_version == "vBIN") & (df.split == "train") & (df.seed == "13")].iloc[
        0
    ]
    # Same logical split size (mapped via pre_dedup_index)
    assert train_bin["n"] == train_ded["n"]
