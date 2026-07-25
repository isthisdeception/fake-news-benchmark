"""fnb.evaluation — metric suite, statistical stack, result I/O."""

from __future__ import annotations

from .metrics import (
    PRIMARY_METRIC,
    aggregate_seed_metrics,
    core_metrics,
    ece,
    is_ece_eligible,
    mean_std,
    robustness_metrics,
    transfer_metrics,
)
from .stats import (
    BootstrapDiffResult,
    ConfirmatoryComparison,
    McNemarResult,
    assemble_confirmatory,
    bh_fdr,
    cliffs_delta,
    confirmatory_claim_allowed,
    friedman_nemenyi,
    mcnemar,
    paired_bootstrap_diff,
)

__all__ = [
    "PRIMARY_METRIC",
    "BootstrapDiffResult",
    "ConfirmatoryComparison",
    "McNemarResult",
    "aggregate_seed_metrics",
    "assemble_confirmatory",
    "bh_fdr",
    "cliffs_delta",
    "confirmatory_claim_allowed",
    "core_metrics",
    "ece",
    "friedman_nemenyi",
    "is_ece_eligible",
    "mcnemar",
    "mean_std",
    "paired_bootstrap_diff",
    "robustness_metrics",
    "transfer_metrics",
]
