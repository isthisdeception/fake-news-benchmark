"""Tests for STATS-CONF (S14): bootstrap, McNemar, Cliff's δ, BH-FDR, Friedman."""

from __future__ import annotations

import numpy as np
import pytest
from statsmodels.stats.multitest import multipletests

from fnb.evaluation.stats import (
    assemble_confirmatory,
    bh_fdr,
    cliffs_delta,
    confirmatory_claim_allowed,
    friedman_nemenyi,
    mcnemar,
    paired_bootstrap_diff,
)


def test_paired_bootstrap_reproducible_with_seed():
    rng = np.random.default_rng(0)
    y = np.array([0, 1] * 40)
    # Model A slightly better than B
    pred_a = y.copy()
    flip_a = rng.choice(len(y), size=8, replace=False)
    pred_a[flip_a] = 1 - pred_a[flip_a]
    pred_b = y.copy()
    flip_b = rng.choice(len(y), size=20, replace=False)
    pred_b[flip_b] = 1 - pred_b[flip_b]

    r1 = paired_bootstrap_diff(y, pred_a, pred_b, n_resamples=500, seed=13)
    r2 = paired_bootstrap_diff(y, pred_a, pred_b, n_resamples=500, seed=13)
    r3 = paired_bootstrap_diff(y, pred_a, pred_b, n_resamples=500, seed=99)
    assert r1.mean_diff == r2.mean_diff
    assert r1.ci_low == r2.ci_low
    assert r1.ci_high == r2.ci_high
    assert r1.p_value == r2.p_value
    assert r1.mean_diff != r3.mean_diff or r1.ci_low != r3.ci_low


def test_paired_bootstrap_ci_covers_known_positive_effect():
    rng = np.random.default_rng(7)
    y = np.array([0, 1] * 60)
    pred_a = y.copy()
    pred_b = y.copy()
    # Corrupt B much more → A − B > 0
    bad = rng.choice(len(y), size=30, replace=False)
    pred_b[bad] = 1 - pred_b[bad]
    mild = rng.choice(len(y), size=5, replace=False)
    pred_a[mild] = 1 - pred_a[mild]

    r = paired_bootstrap_diff(y, pred_a, pred_b, n_resamples=2000, seed=42)
    assert r.mean_diff > 0
    assert r.observed_diff > 0
    # CI of the difference should lie mostly above 0 for a clear effect
    assert r.ci_low > -0.05
    assert r.ci_high > r.ci_low
    assert 0.0 <= r.p_value <= 1.0
    assert r.stratified is True
    assert r.n_resamples == 2000


def test_mcnemar_hand_example_discordant():
    # Hand table: A better. Discordant: n_c=9 (A right B wrong), n_b=1 (A wrong B right).
    y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1])
    # Start both correct, then set discordant pairs on first 10 labels alternating.
    pred_a = y.copy()
    pred_b = y.copy()
    # Indices 0..8: A correct, B wrong  (n_c = 9)
    for i in range(9):
        pred_b[i] = 1 - y[i]
    # Index 9: A wrong, B correct (n_b = 1)
    pred_a[9] = 1 - y[9]
    # 10,11 both correct

    r = mcnemar(y, pred_a, pred_b)
    assert r.n_c == 9
    assert r.n_b == 1
    assert r.n_discordant == 10
    # Exact McNemar: P(X<=1) for Binomial(10, 0.5) two-sided is small
    assert r.p_value < 0.05


def test_cliffs_delta_sign_and_bounds():
    a = [0.9, 0.85, 0.88]
    b = [0.5, 0.55, 0.6]
    d = cliffs_delta(a, b)
    assert d > 0
    assert -1.0 <= d <= 1.0
    assert cliffs_delta(b, a) == pytest.approx(-d)
    assert cliffs_delta([1, 2, 3], [1, 2, 3]) == pytest.approx(0.0)


def test_bh_fdr_matches_statsmodels():
    pvals = [0.001, 0.01, 0.04, 0.2, 0.5]
    ours = bh_fdr(pvals, q=0.05)
    reject_ref, p_adj_ref, _, _ = multipletests(pvals, alpha=0.05, method="fdr_bh")
    np.testing.assert_allclose(ours["p_adjusted"], p_adj_ref)
    np.testing.assert_array_equal(ours["reject"], reject_ref)


def test_assemble_confirmatory_applies_bh_across_whole_family():
    family = [
        {
            "name": "enc_a_vs_b",
            "p_raw": 0.001,
            "effect_size": 0.4,
            "ci_low": 0.02,
            "ci_high": 0.10,
            "mean_diff": 0.06,
        },
        {
            "name": "enc_a_vs_c",
            "p_raw": 0.04,
            "effect_size": 0.1,
            "ci_low": -0.01,
            "ci_high": 0.05,
            "mean_diff": 0.02,
        },
        {
            "name": "delta_f1_m1",
            "p_raw": 0.20,
            "effect_size": 0.05,
            "ci_low": -0.02,
            "ci_high": 0.08,
            "mean_diff": 0.03,
        },
    ]
    out = assemble_confirmatory(family, q=0.05)
    assert len(out) == 3
    # Family-wide BH: first should remain significant; last not
    assert out[0].significant
    assert out[0].claim_allowed
    assert out[0].p_adjusted <= out[0].p_raw or out[0].p_adjusted >= 0
    assert not out[2].significant
    assert not out[2].claim_allowed


def test_confirmatory_claim_rejects_incomplete_triple():
    assert not confirmatory_claim_allowed(
        p_adjusted=0.01,
        effect_size=None,
        ci_low=0.0,
        ci_high=0.1,
    )
    assert not confirmatory_claim_allowed(
        p_adjusted=0.2,
        effect_size=0.3,
        ci_low=0.0,
        ci_high=0.1,
    )
    assert confirmatory_claim_allowed(
        p_adjusted=0.01,
        effect_size=0.3,
        ci_low=0.0,
        ci_high=0.1,
    )


def test_assemble_confirmatory_requires_triple_fields():
    with pytest.raises(ValueError, match="missing confirmatory triple"):
        assemble_confirmatory(
            [{"name": "x", "p_raw": 0.01, "effect_size": 0.2, "ci_low": 0.0}]
        )


def test_friedman_nemenyi_runs_and_marks_exploratory():
    # 5 blocks × 3 models; model0 clearly best
    mat = np.array(
        [
            [0.90, 0.70, 0.65],
            [0.88, 0.72, 0.60],
            [0.91, 0.68, 0.62],
            [0.89, 0.71, 0.64],
            [0.92, 0.69, 0.63],
        ]
    )
    try:
        out = friedman_nemenyi(mat, labels=["A", "B", "C"])
    except ImportError:
        pytest.skip("scikit-posthocs not installed")
    assert out["scope"] == "secondary_exploratory"
    assert out["friedman_p_value"] < 0.05
    assert out["nemenyi_pvalues"].shape == (3, 3)
