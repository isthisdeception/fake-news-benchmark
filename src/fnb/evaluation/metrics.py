"""METRICS-CORE / ECE / TRANSFER / ROBUST (protocol §8; matrix §0.7).

All functions operate on arrays for a **single seed/run**. Aggregate across
seeds with :func:`mean_std` / :func:`aggregate_seed_metrics` — never select a
best run (``configs/metrics.yaml`` ``aggregation: mean_std_across_seeds``).

Label space (§4.3): ``real=0``, ``fake=1``.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from fnb.config import load_config

PRIMARY_METRIC = "macro_f1"

# Labels (§4.3)
LABEL_REAL = 0
LABEL_FAKE = 1


def _as_1d_int(y: Any) -> np.ndarray:
    arr = np.asarray(y)
    if arr.ndim != 1:
        arr = arr.ravel()
    return arr.astype(int, copy=False)


def _as_prob_positive(y_prob: Any) -> np.ndarray:
    """Return P(fake=1) as a 1-d float array.

    Accepts shape ``(n,)`` (positive-class scores) or ``(n, 2)`` (class probs).
    """
    if y_prob is None:
        raise ValueError("y_prob is required for score-based metrics (ROC-AUC / PR-AUC / ECE)")
    arr = np.asarray(y_prob, dtype=float)
    if arr.ndim == 2:
        if arr.shape[1] != 2:
            raise ValueError(f"y_prob with ndim=2 must have shape (n, 2); got {arr.shape}")
        return arr[:, LABEL_FAKE]
    if arr.ndim != 1:
        raise ValueError(f"y_prob must be 1-d or (n, 2); got shape {arr.shape}")
    return arr


def _n_bins_from_config(config_dir: str | Path | None = None) -> int:
    cfg = load_config("metrics", config_dir)
    return int(cfg.ece.bins)


def is_ece_eligible(
    *,
    model_id: str | None = None,
    has_probabilities: bool = True,
    config_dir: str | Path | None = None,
) -> bool:
    """Return whether ECE may be computed (protocol §8 / eval R7).

    LLM zero-shot (and any id listed under ``ece.exclude``) is ineligible.
    Models without valid probability outputs are ineligible.
    """
    if not has_probabilities:
        return False
    cfg = load_config("metrics", config_dir)
    if not cfg.ece.probabilistic_models_only:
        return True
    exclude = {str(x) for x in cfg.ece.exclude}
    if model_id is not None and str(model_id) in exclude:
        return False
    return True


def core_metrics(
    y_true: Any,
    y_pred: Any,
    y_prob: Any | None = None,
) -> dict[str, float]:
    """Compute METRICS-CORE for one evaluation (protocol §8).

    Parameters
    ----------
    y_true, y_pred:
        Integer labels in ``{0,1}``.
    y_prob:
        Positive-class scores ``P(fake=1)`` (1-d) or class probabilities ``(n, 2)``.
        Required for ROC-AUC / PR-AUC; pass ``None`` only when those must be
        omitted (they become ``nan``).

    Returns
    -------
    dict
        Keys: ``macro_f1`` (primary), ``accuracy``, ``macro_precision``,
        ``macro_recall``, ``per_class_f1_fake``, ``per_class_f1_real``,
        ``roc_auc``, ``pr_auc``.
    """
    yt = _as_1d_int(y_true)
    yp = _as_1d_int(y_pred)
    if len(yt) != len(yp):
        raise ValueError(f"y_true/y_pred length mismatch: {len(yt)} vs {len(yp)}")
    if len(yt) == 0:
        raise ValueError("y_true/y_pred must be non-empty")

    per_class = f1_score(yt, yp, labels=[LABEL_REAL, LABEL_FAKE], average=None, zero_division=0)
    # sklearn order follows ``labels=[real, fake]``
    f1_real = float(per_class[0])
    f1_fake = float(per_class[1])

    out: dict[str, float] = {
        "macro_f1": float(f1_score(yt, yp, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(yt, yp)),
        "macro_precision": float(precision_score(yt, yp, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(yt, yp, average="macro", zero_division=0)),
        "per_class_f1_fake": f1_fake,
        "per_class_f1_real": f1_real,
        "roc_auc": float("nan"),
        "pr_auc": float("nan"),
    }

    if y_prob is not None:
        scores = _as_prob_positive(y_prob)
        if len(scores) != len(yt):
            raise ValueError(f"y_prob length mismatch: {len(scores)} vs {len(yt)}")
        # AUC needs both classes present; otherwise undefined → nan.
        if len(np.unique(yt)) < 2:
            out["roc_auc"] = float("nan")
            out["pr_auc"] = float("nan")
        else:
            out["roc_auc"] = float(roc_auc_score(yt, scores))
            out["pr_auc"] = float(average_precision_score(yt, scores))
    return out


def ece(
    y_true: Any,
    y_prob: Any,
    n_bins: int | None = None,
    *,
    model_id: str | None = None,
    config_dir: str | Path | None = None,
) -> float:
    """Expected Calibration Error (equal-width bins on predicted-class confidence).

    Gated (protocol §8): returns ``nan`` when the model is ECE-ineligible
    (e.g. ``llm_zero_shot``) or when ``y_prob`` is ``None``. Bin count defaults
    to ``configs/metrics.yaml`` ``ece.bins`` (frozen **15**).
    """
    if y_prob is None:
        return float("nan")
    if not is_ece_eligible(model_id=model_id, has_probabilities=True, config_dir=config_dir):
        return float("nan")

    bins = int(n_bins) if n_bins is not None else _n_bins_from_config(config_dir)
    if bins < 1:
        raise ValueError(f"n_bins must be >= 1; got {bins}")

    yt = _as_1d_int(y_true)
    p_pos = _as_prob_positive(y_prob)
    if len(yt) != len(p_pos):
        raise ValueError(f"y_true/y_prob length mismatch: {len(yt)} vs {len(p_pos)}")
    if len(yt) == 0:
        return float("nan")

    pred = (p_pos >= 0.5).astype(int)
    conf = np.where(pred == LABEL_FAKE, p_pos, 1.0 - p_pos)
    correct = (pred == yt).astype(float)

    edges = np.linspace(0.0, 1.0, bins + 1)
    n = float(len(yt))
    total = 0.0
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        if i == bins - 1:
            mask = (conf >= lo) & (conf <= hi)
        else:
            mask = (conf >= lo) & (conf < hi)
        count = int(mask.sum())
        if count == 0:
            continue
        acc_bin = float(correct[mask].mean())
        conf_bin = float(conf[mask].mean())
        total += (count / n) * abs(acc_bin - conf_bin)
    return float(total)


def transfer_metrics(
    in_domain_f1: float,
    cross_domain_f1: float | Sequence[float],
) -> dict[str, Any]:
    """TRANSFER metrics (protocol §8, RQ-A).

    ``ΔF1 = in_domain_macro_f1 − cross_domain_macro_f1``.
    Dispersion is the std of macro-F1 across transfer targets.
    """
    in_f1 = float(in_domain_f1)
    cross = np.atleast_1d(np.asarray(cross_domain_f1, dtype=float))
    if cross.size == 0:
        raise ValueError("cross_domain_f1 must be non-empty")
    deltas = in_f1 - cross
    std_targets = float(np.std(cross, ddof=0)) if cross.size > 1 else 0.0
    return {
        "delta_f1": float(deltas.mean()),
        "delta_f1_per_target": [float(x) for x in deltas],
        "std_macro_f1_across_targets": std_targets,
        "in_domain_macro_f1": in_f1,
        "cross_domain_macro_f1_mean": float(cross.mean()),
        "cross_domain_macro_f1": [float(x) for x in cross],
    }


def robustness_metrics(
    y_true: Any,
    y_pred_clean: Any,
    y_pred_attacked: Any,
    similarities: Any,
) -> dict[str, float]:
    """ROBUST metrics (protocol §8, RQ-C).

    Attack Success Rate (ASR) is the fraction of **originally correct** examples
    that become incorrect under attack (standard label-preserving ASR). Mean
    semantic similarity is computed over successful attacks only. Inputs are
    assumed already filtered by the USE ≥ 0.90 constraint at attack time (R9).
    """
    yt = _as_1d_int(y_true)
    yc = _as_1d_int(y_pred_clean)
    ya = _as_1d_int(y_pred_attacked)
    sim = np.asarray(similarities, dtype=float).ravel()
    n = len(yt)
    if not (len(yc) == n and len(ya) == n and len(sim) == n):
        raise ValueError(
            f"length mismatch: y_true={n}, clean={len(yc)}, attacked={len(ya)}, sim={len(sim)}"
        )
    if n == 0:
        raise ValueError("inputs must be non-empty")

    clean_correct = yc == yt
    attacked_correct = ya == yt
    successful = clean_correct & ~attacked_correct
    n_clean_correct = int(clean_correct.sum())
    n_success = int(successful.sum())

    clean_acc = float(clean_correct.mean())
    attacked_acc = float(attacked_correct.mean())
    asr = float(n_success / n_clean_correct) if n_clean_correct > 0 else float("nan")
    if n_success > 0:
        mean_sim = float(sim[successful].mean())
    else:
        mean_sim = float("nan")

    return {
        "clean_accuracy": clean_acc,
        "accuracy_under_attack": attacked_acc,
        "attack_success_rate": asr,
        "mean_semantic_similarity_of_successful_attacks": mean_sim,
    }


def mean_std(values: Iterable[float]) -> dict[str, float]:
    """Mean ± sample std (ddof=1) over per-seed values; never best-run-only."""
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        raise ValueError("values must be non-empty")
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    return {"mean": mean, "std": std, "n": int(arr.size)}


def aggregate_seed_metrics(
    per_seed: Sequence[Mapping[str, float]],
) -> dict[str, dict[str, float]]:
    """Aggregate a list of per-seed metric dicts to mean ± std per key.

    Skips non-finite values per key (e.g. gated ECE ``nan`` on a seed).
    """
    if not per_seed:
        raise ValueError("per_seed must be non-empty")
    keys: list[str] = []
    seen: set[str] = set()
    for row in per_seed:
        for k in row:
            if k not in seen:
                seen.add(k)
                keys.append(k)
    out: dict[str, dict[str, float]] = {}
    for key in keys:
        vals = [float(row[key]) for row in per_seed if key in row and math.isfinite(float(row[key]))]
        if not vals:
            out[key] = {"mean": float("nan"), "std": float("nan"), "n": 0}
        else:
            out[key] = mean_std(vals)
    return out


__all__ = [
    "PRIMARY_METRIC",
    "aggregate_seed_metrics",
    "core_metrics",
    "ece",
    "is_ece_eligible",
    "mean_std",
    "robustness_metrics",
    "transfer_metrics",
]
