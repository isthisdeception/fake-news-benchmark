"""Tests for EXP-1 (S20): best_encoder tie-break, prediction I/O, CLI dispatch.

These are offline unit tests — no data/GPU needed. They exercise:
1. ``select_best_encoder`` deterministic tie-break logic
2. ``save_predictions`` / ``load_predictions`` round-trip
3. ``compute_encoder_pairwise_stats`` on synthetic predictions
4. ``write_best_encoder`` file format
5. ``primary_seeds`` reads from protocol.yaml
6. CLI parser accepts EXP-1 flags
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fnb.experiments.exp1_indist import (
    ENCODER_MODEL_IDS,
    EXP1_MODEL_IDS,
    EXPERIMENT_ID,
    compute_encoder_pairwise_stats,
    load_predictions,
    primary_seeds,
    save_predictions,
    select_best_encoder,
    write_best_encoder,
)


# ---------------------------------------------------------------------------
# best_encoder selection
# ---------------------------------------------------------------------------

def _make_indist_df(
    rows: list[dict],
) -> pd.DataFrame:
    """Build a minimal ds1_indist-shaped DataFrame from row dicts."""
    base = {
        "protocol_version": "v1.0",
        "dataset_version_tag": "vDEDUP",
        "split_type": "S-RAND",
        "run_id": "test",
        "config_hash": "abc",
        "git_sha": "dead",
    }
    records = [{**base, **r} for r in rows]
    return pd.DataFrame.from_records(records)


def test_select_best_encoder_highest_f1_wins():
    """When one encoder has strictly higher mean macro-F1, it wins."""
    rows = []
    for seed in [13, 42]:
        rows.append({"model_id": "BERT", "seed": seed, "macro_f1": 0.95, "ece": 0.05})
        rows.append({"model_id": "ROBERTA", "seed": seed, "macro_f1": 0.93, "ece": 0.04})
        rows.append({"model_id": "DEBERTA", "seed": seed, "macro_f1": 0.94, "ece": 0.03})
        rows.append({"model_id": "DISTIL", "seed": seed, "macro_f1": 0.91, "ece": 0.06})
        rows.append({"model_id": "ALBERT", "seed": seed, "macro_f1": 0.90, "ece": 0.07})
    df = _make_indist_df(rows)
    assert select_best_encoder(df) == "BERT"


def test_select_best_encoder_f1_tie_break_by_ece():
    """When macro-F1 is tied, lower ECE wins."""
    rows = []
    for seed in [13, 42]:
        rows.append({"model_id": "BERT", "seed": seed, "macro_f1": 0.95, "ece": 0.06})
        rows.append({"model_id": "ROBERTA", "seed": seed, "macro_f1": 0.95, "ece": 0.03})
        rows.append({"model_id": "DEBERTA", "seed": seed, "macro_f1": 0.94, "ece": 0.01})
        rows.append({"model_id": "DISTIL", "seed": seed, "macro_f1": 0.95, "ece": 0.08})
        rows.append({"model_id": "ALBERT", "seed": seed, "macro_f1": 0.90, "ece": 0.02})
    df = _make_indist_df(rows)
    # BERT/ROBERTA/DISTIL all at 0.95; ROBERTA has lowest ECE among them.
    assert select_best_encoder(df) == "ROBERTA"


def test_select_best_encoder_f1_ece_tie_break_by_params():
    """When macro-F1 AND ECE are tied, fewest params wins."""
    rows = []
    for seed in [13, 42]:
        rows.append({"model_id": "BERT", "seed": seed, "macro_f1": 0.95, "ece": 0.05, "n_params": 110_000_000})
        rows.append({"model_id": "ROBERTA", "seed": seed, "macro_f1": 0.95, "ece": 0.05, "n_params": 125_000_000})
        rows.append({"model_id": "DEBERTA", "seed": seed, "macro_f1": 0.95, "ece": 0.05, "n_params": 86_000_000})
        rows.append({"model_id": "DISTIL", "seed": seed, "macro_f1": 0.95, "ece": 0.05, "n_params": 66_000_000})
        rows.append({"model_id": "ALBERT", "seed": seed, "macro_f1": 0.95, "ece": 0.05, "n_params": 12_000_000})
    df = _make_indist_df(rows)
    # All tied on F1 and ECE → fewest params → ALBERT
    assert select_best_encoder(df) == "ALBERT"


def test_select_best_encoder_with_explicit_param_counts():
    """param_counts dict overrides n_params column."""
    rows = []
    for seed in [13, 42]:
        for mid in ENCODER_MODEL_IDS:
            rows.append({"model_id": mid, "seed": seed, "macro_f1": 0.95, "ece": 0.05})
    df = _make_indist_df(rows)
    pc = {"BERT": 110_000_000, "ROBERTA": 125_000_000, "DEBERTA": 86_000_000,
          "DISTIL": 66_000_000, "ALBERT": 12_000_000}
    assert select_best_encoder(df, param_counts=pc) == "ALBERT"


def test_select_best_encoder_no_encoder_rows_raises():
    rows = [{"model_id": "LR", "seed": 13, "macro_f1": 0.90, "ece": float("nan")}]
    df = _make_indist_df(rows)
    with pytest.raises(ValueError, match="no encoder"):
        select_best_encoder(df)


def test_select_best_encoder_aggregates_across_seeds():
    """Verifies aggregation, not best-run-only selection."""
    rows = []
    # BERT: seeds give [0.90, 1.00] → mean=0.95
    rows.append({"model_id": "BERT", "seed": 13, "macro_f1": 0.90, "ece": 0.05})
    rows.append({"model_id": "BERT", "seed": 42, "macro_f1": 1.00, "ece": 0.05})
    # ROBERTA: seeds give [0.94, 0.94] → mean=0.94 (would beat BERT's best single)
    rows.append({"model_id": "ROBERTA", "seed": 13, "macro_f1": 0.94, "ece": 0.03})
    rows.append({"model_id": "ROBERTA", "seed": 42, "macro_f1": 0.94, "ece": 0.03})
    for mid in ("DEBERTA", "DISTIL", "ALBERT"):
        for seed in [13, 42]:
            rows.append({"model_id": mid, "seed": seed, "macro_f1": 0.80, "ece": 0.10})
    df = _make_indist_df(rows)
    # BERT mean=0.95 > ROBERTA mean=0.94 → BERT wins via mean, not best-run
    assert select_best_encoder(df) == "BERT"


# ---------------------------------------------------------------------------
# Prediction I/O round-trip
# ---------------------------------------------------------------------------

def test_save_load_predictions_roundtrip(tmp_path: Path):
    path = tmp_path / "preds" / "BERT_seed42.npz"
    y_true = np.array([0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 1])
    y_prob = np.array([0.2, 0.9, 0.4, 0.1, 0.8])
    test_idx = np.array([10, 20, 30, 40, 50])
    save_predictions(
        path, y_true=y_true, y_pred=y_pred, y_prob=y_prob,
        test_indices=test_idx, model_id="BERT", seed=42,
    )
    assert path.is_file()
    loaded = load_predictions(path)
    np.testing.assert_array_equal(loaded["y_true"], y_true)
    np.testing.assert_array_equal(loaded["y_pred"], y_pred)
    np.testing.assert_allclose(loaded["y_prob"], y_prob)
    np.testing.assert_array_equal(loaded["test_indices"], test_idx)


# ---------------------------------------------------------------------------
# write_best_encoder file format
# ---------------------------------------------------------------------------

def test_write_best_encoder_format(tmp_path: Path):
    path = tmp_path / "results" / "best_encoder.txt"
    write_best_encoder(path, "DEBERTA")
    text = path.read_text(encoding="utf-8")
    assert text.strip() == "DEBERTA"
    assert text.endswith("\n")


# ---------------------------------------------------------------------------
# primary_seeds from config
# ---------------------------------------------------------------------------

def test_primary_seeds_from_protocol_yaml():
    seeds = primary_seeds()
    assert seeds == [13, 21, 42, 87, 100]
    assert all(isinstance(s, int) for s in seeds)


# ---------------------------------------------------------------------------
# Pairwise stats on synthetic predictions
# ---------------------------------------------------------------------------

def test_compute_pairwise_stats_synthetic(tmp_path: Path):
    """Verify pairwise stats runs on two synthetic encoders with 2 seeds."""
    art = tmp_path / "artifacts"
    encoder_ids = ("BERT", "ROBERTA")
    seeds = [13, 42]

    rng = np.random.default_rng(99)
    n_test = 60
    y_true = rng.integers(0, 2, size=n_test)

    indist_rows = []
    for mid in encoder_ids:
        for seed in seeds:
            y_pred = y_true.copy()
            n_flip = 5 if mid == "BERT" else 12
            flip_idx = rng.choice(n_test, size=n_flip, replace=False)
            y_pred[flip_idx] = 1 - y_pred[flip_idx]
            y_prob = rng.uniform(0.3, 0.7, size=n_test)
            save_predictions(
                art / "exp1_preds" / f"{mid}_seed{seed}.npz",
                y_true=y_true, y_pred=y_pred, y_prob=y_prob,
                test_indices=np.arange(n_test), model_id=mid, seed=seed,
            )
            from sklearn.metrics import f1_score
            mf1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
            indist_rows.append({
                "model_id": mid, "seed": seed, "macro_f1": mf1, "ece": 0.05,
            })

    df = _make_indist_df(indist_rows)
    pair_df = compute_encoder_pairwise_stats(
        ds1_indist=df, artifacts_dir=art, seeds=seeds,
        encoder_ids=encoder_ids, n_resamples=200, bootstrap_seed=7,
    )
    assert len(pair_df) == 1  # C(2,2)=1 pair
    row = pair_df.iloc[0]
    assert row["comparison"] == "BERT_vs_ROBERTA"
    assert row["hypothesis"] == "H1"
    assert "p_raw" in pair_df.columns
    assert "p_adjusted" in pair_df.columns
    assert "cliffs_delta" in pair_df.columns
    assert "mcnemar_statistic" in pair_df.columns
    assert "significant" in pair_df.columns
    assert "claim_allowed" in pair_df.columns


# ---------------------------------------------------------------------------
# EXP1_MODEL_IDS matches protocol
# ---------------------------------------------------------------------------

def test_exp1_model_ids_correct():
    assert EXP1_MODEL_IDS == ("LR", "SVM", "RF", "LGBM", "BERT", "ROBERTA", "DEBERTA", "DISTIL", "ALBERT")
    assert len(EXP1_MODEL_IDS) == 9
    assert EXPERIMENT_ID == "EXP-1"


# ---------------------------------------------------------------------------
# CLI parser
# ---------------------------------------------------------------------------

def test_cli_parser_exp1_basic():
    from scripts.run_experiment import build_parser
    args = build_parser().parse_args(["--exp", "EXP-1", "--seed", "13"])
    assert args.exp == "EXP-1"
    assert args.seed == 13


def test_cli_parser_exp1_all_seeds():
    from scripts.run_experiment import build_parser
    args = build_parser().parse_args(["--exp", "EXP-1", "--all-seeds"])
    assert args.all_seeds is True
    assert args.seed is None


def test_cli_parser_exp1_model_flag():
    from scripts.run_experiment import build_parser
    args = build_parser().parse_args(["--exp", "EXP-1", "--model", "BERT", "--seed", "42"])
    assert args.model == "BERT"


def test_cli_parser_exp1_finalize_only():
    from scripts.run_experiment import build_parser
    args = build_parser().parse_args(["--exp", "EXP-1", "--finalize-only"])
    assert args.finalize_only is True
