"""Tests for EXP-P2 within-dataset deduplication."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

from fnb.data.dedup import (
    dedup_all,
    dedup_dataframe,
    exact_duplicate_mask,
    near_duplicate_mask,
)


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
