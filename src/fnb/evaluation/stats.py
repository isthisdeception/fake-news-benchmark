"""STATS-CONF statistical stack (protocol §10; evaluation R5/R6).

Primary difference test: paired bootstrap on the **macro-F1 difference**
(10 000 stratified resamples, 95% CI of the difference, two-sided p).

Also: McNemar (paired predictions), Cliff's delta (per-seed scores),
Benjamini–Hochberg FDR across the **entire** confirmatory family, and
Friedman + Nemenyi (exploratory ranking only).

A confirmatory claim requires the full triple
``(BH-adjusted p, effect size, CI of the difference)`` — never a raw delta.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats as scipy_stats
from sklearn.metrics import f1_score
from statsmodels.stats.contingency_tables import mcnemar as sm_mcnemar
from statsmodels.stats.multitest import multipletests

from fnb.config import load_config

LABEL_REAL = 0
LABEL_FAKE = 1


@dataclass(frozen=True)
class BootstrapDiffResult:
    """Paired-bootstrap macro-F1 difference (A − B)."""

    mean_diff: float
    ci_low: float
    ci_high: float
    p_value: float
    n_resamples: int
    stratified: bool
    ci_level: float
    observed_diff: float
    seed: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class McNemarResult:
    statistic: float
    p_value: float
    n_b: int  # A wrong, B right
    n_c: int  # A right, B wrong
    n_discordant: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ConfirmatoryComparison:
    """One member of the confirmatory family after family-wide BH-FDR."""

    name: str
    p_raw: float
    p_adjusted: float
    effect_size: float  # Cliff's delta
    ci_low: float
    ci_high: float
    mean_diff: float
    significant: bool
    claim_allowed: bool

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _as_1d_int(y: Any) -> np.ndarray:
    return np.asarray(y).astype(int, copy=False).ravel()


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def _load_stats_cfg(config_dir: str | Path | None = None):
    return load_config("stats", config_dir)


def _stratified_indices(
    y: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Bootstrap sample indices stratified by binary label."""
    parts: list[np.ndarray] = []
    for label in (LABEL_REAL, LABEL_FAKE):
        idx = np.flatnonzero(y == label)
        if idx.size == 0:
            continue
        draw = rng.choice(idx, size=idx.size, replace=True)
        parts.append(draw)
    if not parts:
        # Degenerate: resample all indices with replacement.
        return rng.choice(np.arange(len(y)), size=len(y), replace=True)
    return np.concatenate(parts)


def paired_bootstrap_diff(
    y_true: Any,
    pred_a: Any,
    pred_b: Any,
    *,
    n_resamples: int | None = None,
    stratified: bool | None = None,
    ci: float | None = None,
    seed: int | None = None,
    config_dir: str | Path | None = None,
) -> BootstrapDiffResult:
    """Paired bootstrap on macro-F1(A) − macro-F1(B) (protocol §10).

    Stratified resampling (by ``y_true``) is the frozen default. Returns the
    mean bootstrap difference, percentile CI of the difference, and a
    two-sided p-value for H0: difference = 0.
    """
    cfg = _load_stats_cfg(config_dir)
    n = int(cfg.bootstrap.n_resamples if n_resamples is None else n_resamples)
    use_strat = bool(cfg.bootstrap.stratified if stratified is None else stratified)
    ci_level = float(cfg.bootstrap.ci if ci is None else ci)
    if n < 1:
        raise ValueError(f"n_resamples must be >= 1; got {n}")
    if not (0.0 < ci_level < 1.0):
        raise ValueError(f"ci must be in (0,1); got {ci_level}")

    yt = _as_1d_int(y_true)
    ya = _as_1d_int(pred_a)
    yb = _as_1d_int(pred_b)
    if not (len(yt) == len(ya) == len(yb)):
        raise ValueError("y_true / pred_a / pred_b length mismatch")
    if len(yt) == 0:
        raise ValueError("inputs must be non-empty")

    observed = _macro_f1(yt, ya) - _macro_f1(yt, yb)
    rng = np.random.default_rng(seed)
    diffs = np.empty(n, dtype=float)
    all_idx = np.arange(len(yt))

    for i in range(n):
        if use_strat:
            idx = _stratified_indices(yt, rng)
        else:
            idx = rng.choice(all_idx, size=len(yt), replace=True)
        diffs[i] = _macro_f1(yt[idx], ya[idx]) - _macro_f1(yt[idx], yb[idx])

    alpha = 1.0 - ci_level
    ci_low = float(np.quantile(diffs, alpha / 2.0))
    ci_high = float(np.quantile(diffs, 1.0 - alpha / 2.0))
    mean_diff = float(np.mean(diffs))

    # Two-sided bootstrap p: twice the smaller tail relative to 0.
    p_le = float(np.mean(diffs <= 0.0))
    p_ge = float(np.mean(diffs >= 0.0))
    p_value = float(min(1.0, 2.0 * min(p_le, p_ge)))

    return BootstrapDiffResult(
        mean_diff=mean_diff,
        ci_low=ci_low,
        ci_high=ci_high,
        p_value=p_value,
        n_resamples=n,
        stratified=use_strat,
        ci_level=ci_level,
        observed_diff=float(observed),
        seed=seed,
    )


def mcnemar_test(
    y_true: Any,
    pred_a: Any,
    pred_b: Any,
) -> McNemarResult:
    """McNemar test on paired predictions (same test set; protocol §10).

    Contrasts correctness of A vs B. Uses statsmodels McNemar
    (exact for small discordant counts; continuity-corrected otherwise).
    """
    yt = _as_1d_int(y_true)
    ya = _as_1d_int(pred_a)
    yb = _as_1d_int(pred_b)
    if not (len(yt) == len(ya) == len(yb)):
        raise ValueError("y_true / pred_a / pred_b length mismatch")

    a_correct = ya == yt
    b_correct = yb == yt
    # Standard 2×2: rows A correct/wrong, cols B correct/wrong
    # n_b = A wrong, B right; n_c = A right, B wrong
    n_b = int((~a_correct & b_correct).sum())
    n_c = int((a_correct & ~b_correct).sum())
    table = np.array(
        [
            [int((a_correct & b_correct).sum()), n_c],
            [n_b, int((~a_correct & ~b_correct).sum())],
        ],
        dtype=int,
    )
    # exact=True for small discordant counts; otherwise corrected chi2
    n_disc = n_b + n_c
    result = sm_mcnemar(table, exact=n_disc < 25, correction=True)
    return McNemarResult(
        statistic=float(result.statistic),
        p_value=float(result.pvalue),
        n_b=n_b,
        n_c=n_c,
        n_discordant=n_disc,
    )


# Public alias matching the S14 prompt name.
mcnemar = mcnemar_test


def cliffs_delta(scores_a: Any, scores_b: Any) -> float:
    """Cliff's delta effect size on two independent score samples.

    δ ∈ [-1, 1]. Positive ⇒ A tends to be larger than B. Used on per-seed
    macro-F1 (protocol §10).
    """
    a = np.asarray(scores_a, dtype=float).ravel()
    b = np.asarray(scores_b, dtype=float).ravel()
    if a.size == 0 or b.size == 0:
        raise ValueError("scores_a and scores_b must be non-empty")
    # Dominance count without building full n×m matrix when possible.
    gt = 0
    lt = 0
    for x in a:
        gt += int(np.sum(x > b))
        lt += int(np.sum(x < b))
    return float((gt - lt) / (a.size * b.size))


def bh_fdr(
    pvalues: Sequence[float],
    q: float | None = None,
    *,
    config_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Benjamini–Hochberg FDR (protocol §10).

    Returns adjusted p-values and reject flags. ``q`` defaults to
    ``configs/stats.yaml`` ``multiple_comparisons.q`` (0.05).
    """
    cfg = _load_stats_cfg(config_dir)
    alpha = float(cfg.multiple_comparisons.q if q is None else q)
    p = np.asarray(list(pvalues), dtype=float)
    if p.ndim != 1:
        raise ValueError("pvalues must be 1-d")
    if p.size == 0:
        return {
            "p_adjusted": np.array([], dtype=float),
            "reject": np.array([], dtype=bool),
            "q": alpha,
        }
    reject, p_adj, _, _ = multipletests(p, alpha=alpha, method="fdr_bh")
    return {
        "p_adjusted": np.asarray(p_adj, dtype=float),
        "reject": np.asarray(reject, dtype=bool),
        "q": alpha,
    }


def friedman_nemenyi(
    score_matrix: Any,
    *,
    labels: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Friedman omnibus + Nemenyi post-hoc (exploratory ranking only).

    ``score_matrix`` shape ``(n_blocks, n_models)`` — e.g. seeds × models or
    datasets × models. Higher scores rank better.
    """
    mat = np.asarray(score_matrix, dtype=float)
    if mat.ndim != 2 or mat.shape[0] < 2 or mat.shape[1] < 2:
        raise ValueError("score_matrix must be 2-d with >=2 blocks and >=2 models")

    # scipy friedmanchisquare takes one array per treatment (column).
    cols = [mat[:, j] for j in range(mat.shape[1])]
    stat, p_omni = scipy_stats.friedmanchisquare(*cols)

    try:
        import scikit_posthocs as sp
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "scikit-posthocs is required for Nemenyi post-hoc (see requirements.txt)"
        ) from exc

    # posthoc_nemenyi_friedman expects blocks × treatments
    p_pair = sp.posthoc_nemenyi_friedman(mat)
    names = list(labels) if labels is not None else [f"m{j}" for j in range(mat.shape[1])]
    if len(names) != mat.shape[1]:
        raise ValueError("labels length must match n_models")
    p_pair.index = names
    p_pair.columns = names

    return {
        "friedman_statistic": float(stat),
        "friedman_p_value": float(p_omni),
        "nemenyi_pvalues": p_pair,
        "scope": "secondary_exploratory",
    }


def confirmatory_claim_allowed(
    *,
    p_adjusted: float | None,
    effect_size: float | None,
    ci_low: float | None,
    ci_high: float | None,
    q: float | None = None,
    config_dir: str | Path | None = None,
) -> bool:
    """True only when the confirmatory triple is complete and BH-significant.

    Enforces protocol §10 / R5: never claim from a raw delta alone.
    """
    cfg = _load_stats_cfg(config_dir)
    alpha = float(cfg.multiple_comparisons.q if q is None else q)
    vals = (p_adjusted, effect_size, ci_low, ci_high)
    if any(v is None for v in vals):
        return False
    if any(not math.isfinite(float(v)) for v in vals):  # type: ignore[arg-type]
        return False
    return float(p_adjusted) < alpha  # type: ignore[arg-type]


def assemble_confirmatory(
    family: Sequence[Mapping[str, Any]],
    *,
    q: float | None = None,
    config_dir: str | Path | None = None,
) -> list[ConfirmatoryComparison]:
    """Apply BH-FDR across the **entire** confirmatory family.

    Each mapping must provide:
    ``name``, ``p_raw``, ``effect_size``, ``ci_low``, ``ci_high``,
    and optionally ``mean_diff``.
    """
    if not family:
        raise ValueError("confirmatory family must be non-empty")

    names: list[str] = []
    p_raw: list[float] = []
    effects: list[float] = []
    ci_lows: list[float] = []
    ci_highs: list[float] = []
    mean_diffs: list[float] = []

    for i, row in enumerate(family):
        required = ("name", "p_raw", "effect_size", "ci_low", "ci_high")
        missing = [k for k in required if k not in row or row[k] is None]
        if missing:
            raise ValueError(
                f"family[{i}] missing confirmatory triple fields: {missing}"
            )
        names.append(str(row["name"]))
        p_raw.append(float(row["p_raw"]))
        effects.append(float(row["effect_size"]))
        ci_lows.append(float(row["ci_low"]))
        ci_highs.append(float(row["ci_high"]))
        mean_diffs.append(float(row.get("mean_diff", float("nan"))))

    bh = bh_fdr(p_raw, q=q, config_dir=config_dir)
    p_adj = bh["p_adjusted"]
    reject = bh["reject"]
    alpha = float(bh["q"])

    out: list[ConfirmatoryComparison] = []
    for i, name in enumerate(names):
        allowed = confirmatory_claim_allowed(
            p_adjusted=float(p_adj[i]),
            effect_size=effects[i],
            ci_low=ci_lows[i],
            ci_high=ci_highs[i],
            q=alpha,
            config_dir=config_dir,
        )
        out.append(
            ConfirmatoryComparison(
                name=name,
                p_raw=p_raw[i],
                p_adjusted=float(p_adj[i]),
                effect_size=effects[i],
                ci_low=ci_lows[i],
                ci_high=ci_highs[i],
                mean_diff=mean_diffs[i],
                significant=bool(reject[i]),
                claim_allowed=allowed,
            )
        )
    return out


__all__ = [
    "BootstrapDiffResult",
    "ConfirmatoryComparison",
    "McNemarResult",
    "assemble_confirmatory",
    "bh_fdr",
    "cliffs_delta",
    "confirmatory_claim_allowed",
    "friedman_nemenyi",
    "mcnemar",
    "mcnemar_test",
    "paired_bootstrap_diff",
]
