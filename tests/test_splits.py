"""Tests for EXP-P3 stratified splits (indices only)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from fnb.data.splits import (
    TEST_FRAC,
    TRAIN_FRAC,
    VAL_FRAC,
    create_all_splits,
    create_splits_for_dataset,
    source_disjoint_split,
    source_recoverability,
    stratified_random_split,
    temporal_split,
)
from fnb.utils.io import load_indices


def test_stratified_random_ratios_and_partition():
    # Balanced binary labels, n=200
    y = np.array([0] * 100 + [1] * 100)
    splits = stratified_random_split(y, seed=42)
    n = len(y)
    assert len(splits.train) + len(splits.val) + len(splits.test) == n
    assert abs(len(splits.train) / n - TRAIN_FRAC) < 0.03
    assert abs(len(splits.val) / n - VAL_FRAC) < 0.03
    assert abs(len(splits.test) / n - TEST_FRAC) < 0.03
    # Stratification: class ratio ~0.5 in each part
    for part in (splits.train, splits.val, splits.test):
        ratio = y[part].mean()
        assert 0.4 <= ratio <= 0.6


def test_stratified_seed_reproducibility():
    y = np.array([0, 1] * 50)
    a = stratified_random_split(y, seed=13)
    b = stratified_random_split(y, seed=13)
    c = stratified_random_split(y, seed=21)
    assert np.array_equal(a.train, b.train)
    assert np.array_equal(a.val, b.val)
    assert np.array_equal(a.test, b.test)
    assert not np.array_equal(a.train, c.train)


def test_source_disjoint_no_shared_sources():
    # 20 sources × 5 rows; labels alternate by source
    sources = np.repeat([f"s{i}" for i in range(20)], 5)
    labels = np.array([i % 2 for i in range(20) for _ in range(5)])
    splits = source_disjoint_split(labels, sources, seed=42)
    s_train = set(sources[splits.train])
    s_val = set(sources[splits.val])
    s_test = set(sources[splits.test])
    assert not (s_train & s_val)
    assert not (s_train & s_test)
    assert not (s_val & s_test)
    splits.validate_partition(len(labels))


def test_temporal_split_orders_by_time():
    dates = pd.Series(
        pd.to_datetime(
            ["2020-01-01", "2020-06-01", "2021-01-01", "2021-06-01", "2022-01-01"] * 20
        )
    )
    splits = temporal_split(dates)
    t_train_max = dates.iloc[splits.train].max()
    t_val_min = dates.iloc[splits.val].min()
    t_val_max = dates.iloc[splits.val].max()
    t_test_min = dates.iloc[splits.test].min()
    assert t_train_max <= t_val_min
    assert t_val_max <= t_test_min


def test_ds1_source_not_applicable_when_flag_false():
    df = pd.DataFrame({"label": [0, 1, 0, 1], "text": ["a", "b", "c", "d"]})
    ok, col, note = source_recoverability("DS1", {"source_field_available": False}, df)
    assert ok is False
    assert col is None
    assert "not recoverable" in note.lower() or "false" in note.lower()


def test_create_splits_writes_idx_and_na_regimes(tmp_path: Path):
    def _parquet_ok() -> bool:
        try:
            p = tmp_path / "_p.parquet"
            pd.DataFrame({"x": [1]}).to_parquet(p)
            return True
        except Exception:
            return False

    if not _parquet_ok():
        pytest.skip("parquet engine unavailable (Kaggle has pyarrow)")

    processed = tmp_path / "processed"
    processed.mkdir()
    n = 100
    df = pd.DataFrame(
        {
            "uid": [f"u{i}" for i in range(n)],
            "title": [""] * n,
            "text": [f"doc {i}" for i in range(n)],
            "label": [0, 1] * 50,
        }
    )
    df.to_parquet(processed / "DS1_vDEDUP.parquet")

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
        "seeds": {"primary": [13, 42], "secondary": [13], "reporting": "mean"},
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

    splits_dir = tmp_path / "splits"
    appl_path = tmp_path / "split_applicability.csv"
    results, appl = create_all_splits(
        processed_dir=processed,
        splits_dir=splits_dir,
        applicability_path=appl_path,
        config_dir=cfg_dir,
        dataset_ids=["DS1"],
    )
    # 2 seeds × S-RAND × 3 parts = 6 files; S-SRC/S-TEMP N/A
    assert len(list(splits_dir.glob("*.idx"))) == 6
    assert (splits_dir / "DS1_S-RAND_train_13.idx").exists()
    train = load_indices(splits_dir / "DS1_S-RAND_train_13.idx")
    val = load_indices(splits_dir / "DS1_S-RAND_val_13.idx")
    test = load_indices(splits_dir / "DS1_S-RAND_test_13.idx")
    assert len(set(train) & set(val)) == 0
    assert len(set(train) | set(val) | set(test)) == n

    appl_df = pd.read_csv(appl_path)
    assert set(appl_df["regime"]) == {"S-RAND", "S-SRC", "S-TEMP"}
    assert appl_df.loc[appl_df["regime"] == "S-RAND", "status"].iloc[0] == "ok"
    assert appl_df.loc[appl_df["regime"] == "S-SRC", "status"].iloc[0] == "not_applicable"
    assert appl_df.loc[appl_df["regime"] == "S-TEMP", "status"].iloc[0] == "not_applicable"
    assert len(results) == 2  # two S-RAND seeds only


def test_s_src_ok_when_source_column_present(tmp_path: Path):
    n = 60
    df = pd.DataFrame(
        {
            "label": [0, 1] * 30,
            "source": [f"src{i // 3}" for i in range(n)],
            "text": [f"t{i}" for i in range(n)],
        }
    )
    entry = {"source_field_available": "source"}
    ok, col, _ = source_recoverability("DS1", entry, df)
    assert ok and col == "source"
    results, appl = create_splits_for_dataset(
        "DS1", df, entry, seeds=[13], splits_dir=tmp_path / "splits"
    )
    by_regime = {a.regime: a.status for a in appl}
    assert by_regime["S-SRC"] == "ok"
    assert any(r.regime == "S-SRC" for r in results)
    assert (tmp_path / "splits" / "DS1_S-SRC_train_13.idx").exists()
