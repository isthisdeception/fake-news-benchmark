"""Tests for EXP-P2 within-dataset dedup and EXP-P5 cross-dataset overlap."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

from fnb.data.dedup import (
    across_dataset_keep_mask,
    dedup_all,
    dedup_dataframe,
    exact_duplicate_mask,
    near_duplicate_mask,
)
from fnb.data.xdedup import (
    directed_transfer_pairs,
    xdedup_all,
    xdedup_pair,
)
from fnb.utils.io import load_indices


def _frame(texts: list[str], titles: list[str] | None = None) -> pd.DataFrame:
    n = len(texts)
    return pd.DataFrame(
        {
            "dataset_id": ["DS1"] * n,
            "uid": [f"u{i}" for i in range(n)],
            "title": titles if titles is not None else [""] * n,
            "text": texts,
            "label": [i % 2 for i in range(n)],  # must NOT affect dedup
            "label_raw": [i % 2 for i in range(n)],
            "source_file": ["x"] * n,
        }
    )


def test_exact_duplicate_mask_keeps_first():
    texts = ["aaa", "bbb", "aaa", "ccc", "bbb"]
    keep = exact_duplicate_mask(texts)
    assert keep.tolist() == [True, True, False, True, False]


def test_near_duplicate_removes_almost_identical_at_0_95():
    # Near-identical long strings → cosine ≥ 0.95; a clearly different doc kept.
    base = " ".join([f"token{i}" for i in range(40)])
    near = base + " tokenX"
    other = " ".join([f"other{i}" for i in range(40)])
    texts = [base, near, other]
    keep = near_duplicate_mask(texts, threshold=0.95)
    assert keep[0]
    assert not keep[1]
    assert keep[2]


def test_dedup_dataframe_exact_then_near_counts_and_order():
    base = " ".join([f"word{i}" for i in range(50)])
    near = base + " extra"
    df = _frame(
        texts=[base, base, near, "completely different unique document zeta omega"],
        titles=["same", "same", "same", "other"],
    )
    # Flip labels so if dedup peeked at labels it could mis-count.
    df["label"] = [1, 0, 1, 0]

    result = dedup_dataframe(df, dataset_id="DS1", threshold=0.95, version_tag="vDEDUP")
    # Row0 kept; row1 exact-dup of row0 dropped; row2 near-dup of row0 dropped; row3 kept.
    assert result.counts.n_input == 4
    assert result.counts.n_removed_exact == 1
    assert result.counts.n_removed_near == 1
    assert result.counts.n_kept == 2
    assert (
        result.counts.n_kept + result.counts.n_removed_exact + result.counts.n_removed_near
        == result.counts.n_input
    )
    assert result.survivor_indices == [0, 3]
    assert result.dataframe["pre_dedup_index"].tolist() == [0, 3]
    # Order preserved
    assert result.dataframe["uid"].tolist() == ["u0", "u3"]


def test_threshold_read_not_hardcoded_in_counts():
    df = _frame(["alpha beta gamma"] * 2)
    r = dedup_dataframe(df, dataset_id="DS1", threshold=0.98)
    assert r.counts.near_duplicate_threshold == 0.98


def test_dedup_all_writes_vdedup_and_report(tmp_path: Path):
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
    base = " ".join([f"tok{i}" for i in range(30)])
    df = _frame([base, base, base + " z", "totally unrelated material here now"])
    df.to_parquet(processed / "DS1_vCLEAN-C.parquet")

    cfg_dir = tmp_path / "configs"
    cfg_dir.mkdir()
    # Minimal dedup + datasets configs
    dedup_yaml = {
        "within_dataset": {
            "remove_exact_duplicates": True,
            "near_duplicate_metric": "tfidf_cosine",
            "near_duplicate_threshold": 0.95,
            "version_tag": "vDEDUP",
        },
        "across_dataset": {
            "metric": "tfidf_cosine",
            "threshold": 0.90,
            "side_removed": "test",
            "report_to": "results/interdataset_overlap.csv",
            "version_tag": "vXDEDUP",
        },
        "sensitivity": {
            "thresholds": [0.90, 0.95, 0.98],
            "dataset": "DS1",
            "version_tags": ["vDEDUP@0.90", "vDEDUP@0.95", "vDEDUP@0.98"],
        },
    }
    datasets_yaml = {
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
    (cfg_dir / "dedup.yaml").write_text(yaml.dump(dedup_yaml), encoding="utf-8")
    (cfg_dir / "datasets.yaml").write_text(yaml.dump(datasets_yaml), encoding="utf-8")

    report = tmp_path / "dedup_counts.csv"
    results = dedup_all(
        processed_dir=processed,
        output_dir=processed,
        report_path=report,
        config_dir=cfg_dir,
        dataset_ids=["DS1"],
    )
    assert len(results) == 1
    assert (processed / "DS1_vDEDUP.parquet").exists()
    assert report.exists()
    rep = pd.read_csv(report)
    assert rep.iloc[0]["n_kept"] == results[0].counts.n_kept
    assert results[0].counts.near_duplicate_threshold == 0.95


# --- EXP-P5 cross-dataset overlap (vXDEDUP) ---------------------------------


def _frame_ds(
    dataset_id: str,
    texts: list[str],
    *,
    labels: list[int] | None = None,
    titles: list[str] | None = None,
) -> pd.DataFrame:
    n = len(texts)
    return pd.DataFrame(
        {
            "dataset_id": [dataset_id] * n,
            "uid": [f"{dataset_id}_u{i}" for i in range(n)],
            "title": titles if titles is not None else [""] * n,
            "text": texts,
            "label": labels if labels is not None else [i % 2 for i in range(n)],
            "label_raw": labels if labels is not None else [i % 2 for i in range(n)],
            "source_file": ["x"] * n,
            "pre_dedup_index": list(range(n)),
        }
    )


def test_directed_transfer_pairs_article_offdiag_plus_ds4():
    pairs = directed_transfer_pairs(
        article_transfer_set=["DS1", "DS2", "DS3"],
        domain_shift_probe=["DS4"],
    )
    # 3×2 off-diagonal + 3→DS4 = 9
    assert len(pairs) == 9
    assert ("DS1", "DS1") not in pairs
    assert ("DS1", "DS2") in pairs
    assert ("DS2", "DS1") in pairs
    assert ("DS3", "DS4") in pairs
    assert ("DS4", "DS1") not in pairs  # probe is test-only
    assert ("DS1", "DS5") not in pairs


def test_across_dataset_keep_mask_removes_near_dup_at_0_90_keeps_train_intact():
    base = " ".join([f"token{i}" for i in range(40)])
    near = base + " tokenX"
    other = " ".join([f"other{i}" for i in range(40)])
    train = [base, other]
    test = [near, "completely unique zeta document material here"]
    keep = across_dataset_keep_mask(train, test, threshold=0.90)
    assert not keep[0]  # near-dup of train[0] removed from TEST
    assert keep[1]
    # Train length unchanged by design (mask is test-only)
    assert len(train) == 2


def test_xdedup_pair_removes_test_side_only_and_records_balance():
    base = " ".join([f"word{i}" for i in range(50)])
    near = base + " extra"
    unique_test = "totally distinct omega document for residual keep"
    unique_train = " ".join([f"trainuniq{i}" for i in range(40)])

    train_df = _frame_ds(
        "DS1",
        [base, unique_train],
        labels=[0, 1],
        titles=["t0", "t1"],
    )
    test_df = _frame_ds(
        "DS2",
        [near, unique_test, base],  # near + exact copy of train base
        labels=[0, 1, 0],
        titles=["t0", "t1", "t2"],
    )
    # Flip labels so keep/drop cannot be label-driven
    train_df["label"] = [1, 0]
    test_df["label"] = [1, 0, 1]

    result = xdedup_pair(
        train_df,
        test_df,
        train_dataset_id="DS1",
        test_dataset_id="DS2",
        threshold=0.90,
    )
    # Test rows 0 (near) and 2 (exact) removed; row 1 kept. Train untouched.
    assert result.counts.n_train == 2
    assert result.counts.n_test_before == 3
    assert result.counts.n_overlap_removed == 2
    assert result.counts.n_test_after == 1
    assert result.kept_test_indices == [1]
    assert result.counts.side_removed == "test"
    assert result.counts.same_source_control is True  # DS1↔DS2
    # Residual class balance on kept test row (label=0)
    assert result.counts.n_real_after == 1
    assert result.counts.n_fake_after == 0
    assert result.counts.fake_ratio_after == 0.0
    assert "Reuters" in result.counts.notes


def test_xdedup_directions_are_independent():
    """Overlap is direction-aware: A→B removal ≠ B→A removal."""
    shared = " ".join([f"shared{i}" for i in range(40)])
    only_a = " ".join([f"onlya{i}" for i in range(40)])
    only_b = " ".join([f"onlyb{i}" for i in range(40)])

    df_a = _frame_ds("DS1", [shared, only_a], labels=[0, 1])
    df_b = _frame_ds("DS2", [shared, only_b], labels=[0, 1])

    a_to_b = xdedup_pair(
        df_a, df_b, train_dataset_id="DS1", test_dataset_id="DS2", threshold=0.90
    )
    b_to_a = xdedup_pair(
        df_b, df_a, train_dataset_id="DS2", test_dataset_id="DS1", threshold=0.90
    )
    # Each direction drops the shared item from its own TEST side only
    assert a_to_b.kept_test_indices == [1]
    assert b_to_a.kept_test_indices == [1]
    assert a_to_b.counts.n_overlap_removed == 1
    assert b_to_a.counts.n_overlap_removed == 1
    # Train corpora sizes differ from each other but remain full in each direction
    assert a_to_b.counts.n_train == 2
    assert b_to_a.counts.n_train == 2


def test_xdedup_all_writes_indices_and_overlap_csv(tmp_path: Path):
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
    splits = tmp_path / "splits"
    processed.mkdir()
    base = " ".join([f"tok{i}" for i in range(30)])
    # Minimal vDEDUP for DS1–DS4
    for ds_id, texts in {
        "DS1": [base, " ".join([f"ds1uniq{i}" for i in range(25)])],
        "DS2": [base + " z", " ".join([f"ds2uniq{i}" for i in range(25)])],
        "DS3": [" ".join([f"ds3doc{i}" for i in range(25)])],
        "DS4": [" ".join([f"covid{i}" for i in range(25)])],
    }.items():
        _frame_ds(ds_id, texts).to_parquet(processed / f"{ds_id}_vDEDUP.parquet")

    cfg_dir = tmp_path / "configs"
    cfg_dir.mkdir()
    dedup_yaml = {
        "within_dataset": {
            "remove_exact_duplicates": True,
            "near_duplicate_metric": "tfidf_cosine",
            "near_duplicate_threshold": 0.95,
            "version_tag": "vDEDUP",
        },
        "across_dataset": {
            "metric": "tfidf_cosine",
            "threshold": 0.90,
            "side_removed": "test",
            "report_to": "results/interdataset_overlap.csv",
            "version_tag": "vXDEDUP",
        },
        "sensitivity": {
            "thresholds": [0.90, 0.95, 0.98],
            "dataset": "DS1",
            "version_tags": ["vDEDUP@0.90", "vDEDUP@0.95", "vDEDUP@0.98"],
        },
    }
    datasets_yaml = {
        "label_space": {"real": 0, "fake": 1},
        "article_transfer_set": ["DS1", "DS2", "DS3"],
        "domain_shift_probe": ["DS4"],
        "short_statement_track": ["DS5"],
        "datasets": {
            ds: {
                "name": ds,
                "role": "t",
                "text_type": "Article",
                "in_article_transfer": ds != "DS4",
                "kaggle_slug": "x",
                "kaggle_version": "1",
                "input_dirname": "x",
                "input_path": "x",
                "label_column": "label",
                "text_columns": {"title": "title", "body": "text"},
                "source_field_available": False,
                "license_note": "t",
            }
            for ds in ("DS1", "DS2", "DS3", "DS4")
        },
        "version_tags": {"vXDEDUP": "x"},
        "llm_contamination": {"probe": "x", "report_to": "y", "rule": "z"},
    }
    (cfg_dir / "dedup.yaml").write_text(yaml.dump(dedup_yaml), encoding="utf-8")
    (cfg_dir / "datasets.yaml").write_text(yaml.dump(datasets_yaml), encoding="utf-8")

    report = tmp_path / "interdataset_overlap.csv"
    results = xdedup_all(
        processed_dir=processed,
        splits_dir=splits,
        report_path=report,
        config_dir=cfg_dir,
    )
    assert len(results) == 9
    assert report.exists()
    rep = pd.read_csv(report)
    assert len(rep) == 9
    assert set(rep["train_dataset_id"]) <= {"DS1", "DS2", "DS3"}
    # DS1→DS2 same-source flag
    row = rep[(rep["train_dataset_id"] == "DS1") & (rep["test_dataset_id"] == "DS2")].iloc[0]
    assert bool(row["same_source_control"])
    # Index file exists and matches kept indices
    idx_path = splits / "DS1_to_DS2_vXDEDUP_test.idx"
    assert idx_path.exists()
    kept = load_indices(idx_path)
    match = next(
        r for r in results if r.train_dataset_id == "DS1" and r.test_dataset_id == "DS2"
    )
    assert kept == match.kept_test_indices
    # Near-dup of DS1 base should be removed from DS2 test side
    assert match.counts.n_overlap_removed >= 1
    assert match.counts.threshold == 0.90
