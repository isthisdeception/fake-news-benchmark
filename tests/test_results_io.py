"""Tests for result-row schema + append-only writer (S15)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fnb.evaluation.results_io import (
    PROVENANCE_COLUMNS,
    ResultSchemaError,
    aggregate_seeds,
    make_result_row,
    validate_result_row,
    write_result_rows,
)


def _row(seed: int, macro_f1: float, accuracy: float = 0.9, **kwargs):
    base = dict(
        dataset_version_tag="vDEDUP",
        split_type="S-RAND",
        model_id="BERT",
        seed=seed,
        run_id=f"exp1_BERT_vDEDUP_S-RAND_seed{seed}",
        config_hash="abc123",
        git_sha="deadbeef",
        metrics={"macro_f1": macro_f1, "accuracy": accuracy},
    )
    base.update(kwargs)
    return make_result_row(**base)


def test_provenance_columns_frozen():
    assert "protocol_version" in PROVENANCE_COLUMNS
    assert "git_sha" in PROVENANCE_COLUMNS
    assert "config_hash" in PROVENANCE_COLUMNS


def test_validate_rejects_missing_protocol_version():
    row = {
        "dataset_version_tag": "vDEDUP",
        "split_type": "S-RAND",
        "model_id": "LR",
        "seed": 13,
        "run_id": "r1",
        "config_hash": "h",
        "git_sha": "g",
        "macro_f1": 0.5,
    }
    with pytest.raises(ResultSchemaError, match="protocol_version"):
        validate_result_row(row)


def test_validate_rejects_wrong_protocol_version():
    row = _row(13, 0.5)
    row["protocol_version"] = "v0.9"
    with pytest.raises(ResultSchemaError, match="protocol_version must be"):
        validate_result_row(row)


def test_write_result_rows_appends_never_overwrites(tmp_path: Path):
    path = tmp_path / "ds1_indist.csv"
    write_result_rows(path, [_row(13, 0.80), _row(21, 0.82)])
    write_result_rows(path, [_row(42, 0.81)])

    df = pd.read_csv(path)
    assert len(df) == 3
    assert list(df["seed"]) == [13, 21, 42]
    for col in PROVENANCE_COLUMNS:
        assert col in df.columns
    assert set(df["protocol_version"]) == {"v1.0"}


def test_write_rejects_row_missing_provenance(tmp_path: Path):
    path = tmp_path / "bad.csv"
    bad = {
        "protocol_version": "v1.0",
        "dataset_version_tag": "vDEDUP",
        "split_type": "S-RAND",
        "model_id": "LR",
        "seed": 13,
        # missing run_id, config_hash, git_sha
        "macro_f1": 0.5,
    }
    with pytest.raises(ResultSchemaError, match="missing required provenance"):
        write_result_rows(path, [bad])
    assert not path.exists()


def test_aggregate_seeds_matches_manual_mean_std():
    rows = [_row(13, 0.50), _row(21, 0.70), _row(42, 0.60)]
    df = pd.DataFrame(rows)
    agg = aggregate_seeds(df, metric_columns=["macro_f1", "accuracy"])

    assert len(agg) == 1
    vals = [0.50, 0.70, 0.60]
    assert agg.iloc[0]["macro_f1_mean"] == pytest.approx(float(np.mean(vals)))
    assert agg.iloc[0]["macro_f1_std"] == pytest.approx(float(np.std(vals, ddof=1)))
    assert agg.iloc[0]["macro_f1_n"] == 3
    # Never best-run
    assert agg.iloc[0]["macro_f1_mean"] < max(vals)


def test_aggregate_seeds_groups_by_model_and_split():
    rows = [
        _row(13, 0.5, model_id="BERT"),
        _row(21, 0.7, model_id="BERT"),
        make_result_row(
            dataset_version_tag="vDEDUP",
            split_type="S-RAND",
            model_id="LR",
            seed=13,
            run_id="r_lr",
            config_hash="h",
            git_sha="g",
            metrics={"macro_f1": 0.4, "accuracy": 0.4},
        ),
    ]
    agg = aggregate_seeds(pd.DataFrame(rows), metric_columns=["macro_f1"])
    assert len(agg) == 2
    bert = agg[agg["model_id"] == "BERT"].iloc[0]
    lr = agg[agg["model_id"] == "LR"].iloc[0]
    assert bert["macro_f1_mean"] == pytest.approx(0.6)
    assert lr["macro_f1_n"] == 1
    assert lr["macro_f1_std"] == 0.0
