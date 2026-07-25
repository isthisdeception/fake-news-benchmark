"""Known-answer tests for METRICS-CORE / ECE / TRANSFER / ROBUST (S13)."""

from __future__ import annotations

import math

import numpy as np
import pytest
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from fnb.evaluation.metrics import (
    PRIMARY_METRIC,
    aggregate_seed_metrics,
    core_metrics,
    ece,
    is_ece_eligible,
    mean_std,
    robustness_metrics,
    transfer_metrics,
)


@pytest.fixture
def toy_binary() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Fixed arrays with both classes present.
    y_true = np.array([0, 0, 1, 1, 0, 1, 1, 0])
    y_pred = np.array([0, 1, 1, 1, 0, 0, 1, 0])
    y_prob = np.array([0.1, 0.7, 0.8, 0.9, 0.2, 0.4, 0.85, 0.3])
    return y_true, y_pred, y_prob


def test_primary_metric_is_macro_f1():
    assert PRIMARY_METRIC == "macro_f1"


def test_core_metrics_match_sklearn(toy_binary):
    y_true, y_pred, y_prob = toy_binary
    m = core_metrics(y_true, y_pred, y_prob)

    assert m["macro_f1"] == pytest.approx(
        f1_score(y_true, y_pred, average="macro", zero_division=0)
    )
    assert m["accuracy"] == pytest.approx(accuracy_score(y_true, y_pred))
    assert m["macro_precision"] == pytest.approx(
        precision_score(y_true, y_pred, average="macro", zero_division=0)
    )
    assert m["macro_recall"] == pytest.approx(
        recall_score(y_true, y_pred, average="macro", zero_division=0)
    )
    per = f1_score(y_true, y_pred, labels=[0, 1], average=None, zero_division=0)
    assert m["per_class_f1_real"] == pytest.approx(per[0])
    assert m["per_class_f1_fake"] == pytest.approx(per[1])
    assert m["roc_auc"] == pytest.approx(roc_auc_score(y_true, y_prob))
    assert m["pr_auc"] == pytest.approx(average_precision_score(y_true, y_prob))


def test_core_metrics_accept_2d_proba(toy_binary):
    y_true, y_pred, y_prob = toy_binary
    proba_2d = np.column_stack([1.0 - y_prob, y_prob])
    m1 = core_metrics(y_true, y_pred, y_prob)
    m2 = core_metrics(y_true, y_pred, proba_2d)
    assert m1["roc_auc"] == pytest.approx(m2["roc_auc"])
    assert m1["pr_auc"] == pytest.approx(m2["pr_auc"])


def test_core_metrics_auc_nan_without_probs(toy_binary):
    y_true, y_pred, _ = toy_binary
    m = core_metrics(y_true, y_pred, y_prob=None)
    assert math.isnan(m["roc_auc"])
    assert math.isnan(m["pr_auc"])
    assert math.isfinite(m["macro_f1"])


def test_ece_hand_computed_two_bins():
    # Predicted-class confidence ECE, n_bins=2, edges [0, 0.5), [0.5, 1].
    # y=1,p=0.9 → pred1 conf0.9 correct
    # y=0,p=0.9 → pred1 conf0.9 incorrect
    # y=1,p=0.55 → pred1 conf0.55 correct
    # y=0,p=0.55 → pred1 conf0.55 incorrect
    # All four in bin [0.5,1]: acc=0.5, mean_conf=0.725 → ECE=0.225
    y_true = [1, 0, 1, 0]
    y_prob = [0.9, 0.9, 0.55, 0.55]
    assert ece(y_true, y_prob, n_bins=2) == pytest.approx(0.225)


def test_ece_reads_15_bins_from_config():
    # Smoke: default path uses configs/metrics.yaml bins=15 and returns finite.
    y_true = np.array([0, 1, 0, 1, 0, 1])
    y_prob = np.array([0.1, 0.9, 0.2, 0.8, 0.4, 0.6])
    val = ece(y_true, y_prob)  # n_bins from config
    assert math.isfinite(val)
    assert 0.0 <= val <= 1.0


def test_ece_gated_for_llm_zero_shot_and_missing_probs():
    y_true = [0, 1]
    y_prob = [0.2, 0.8]
    assert math.isnan(ece(y_true, y_prob, model_id="llm_zero_shot"))
    assert math.isnan(ece(y_true, None))
    assert is_ece_eligible(model_id="BERT", has_probabilities=True)
    assert not is_ece_eligible(model_id="llm_zero_shot", has_probabilities=True)
    assert not is_ece_eligible(model_id="BERT", has_probabilities=False)


def test_transfer_metrics_delta_and_std():
    # ΔF1 = in − cross; std across targets of the cross-domain F1s.
    out = transfer_metrics(0.90, [0.70, 0.60, 0.80])
    assert out["delta_f1"] == pytest.approx((0.20 + 0.30 + 0.10) / 3)
    assert out["delta_f1_per_target"] == pytest.approx([0.20, 0.30, 0.10])
    assert out["std_macro_f1_across_targets"] == pytest.approx(
        float(np.std(np.array([0.70, 0.60, 0.80]), ddof=0))
    )


def test_transfer_metrics_single_target():
    out = transfer_metrics(0.85, 0.75)
    assert out["delta_f1"] == pytest.approx(0.10)
    assert out["std_macro_f1_across_targets"] == 0.0


def test_robustness_metrics_asr_and_mean_sim():
    # 4 examples; clean correct on 0,1,2; attack flips 0 and 1 (success).
    y_true = np.array([1, 0, 1, 0])
    y_clean = np.array([1, 0, 1, 1])  # last wrong on clean → not in ASR denom
    y_atk = np.array([0, 1, 1, 0])  # flips 0,1; 2 stays correct; 3 becomes correct
    sims = np.array([0.95, 0.92, 0.97, 0.91])
    r = robustness_metrics(y_true, y_clean, y_atk, sims)
    assert r["clean_accuracy"] == pytest.approx(0.75)
    assert r["accuracy_under_attack"] == pytest.approx(0.5)  # indices 2,3 correct
    # ASR = 2 successes / 3 originally correct
    assert r["attack_success_rate"] == pytest.approx(2 / 3)
    assert r["mean_semantic_similarity_of_successful_attacks"] == pytest.approx(
        (0.95 + 0.92) / 2
    )


def test_mean_std_never_best_run():
    # Aggregation is mean±std, not max.
    stats = mean_std([0.5, 0.7, 0.6])
    assert stats["mean"] == pytest.approx(0.6)
    assert stats["std"] == pytest.approx(float(np.std([0.5, 0.7, 0.6], ddof=1)))
    assert stats["mean"] < max([0.5, 0.7, 0.6])


def test_aggregate_seed_metrics_skips_nan():
    rows = [
        {"macro_f1": 0.5, "ece": 0.1},
        {"macro_f1": 0.7, "ece": float("nan")},
        {"macro_f1": 0.6, "ece": 0.3},
    ]
    agg = aggregate_seed_metrics(rows)
    assert agg["macro_f1"]["mean"] == pytest.approx(0.6)
    assert agg["macro_f1"]["n"] == 3
    assert agg["ece"]["mean"] == pytest.approx(0.2)
    assert agg["ece"]["n"] == 2
