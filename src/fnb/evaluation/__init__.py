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
