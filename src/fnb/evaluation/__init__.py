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
from .results_io import (
    AGGREGATE_KEYS,
    PROVENANCE_COLUMNS,
    ResultSchemaError,
    aggregate_seeds,
    make_result_row,
    validate_result_row,
    write_result_rows,
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
    "AGGREGATE_KEYS",
    "PRIMARY_METRIC",
    "PROVENANCE_COLUMNS",
    "BootstrapDiffResult",
    "ConfirmatoryComparison",
    "McNemarResult",
    "ResultSchemaError",
    "aggregate_seed_metrics",
    "aggregate_seeds",
    "assemble_confirmatory",
    "bh_fdr",
    "cliffs_delta",
    "confirmatory_claim_allowed",
    "core_metrics",
    "ece",
    "friedman_nemenyi",
    "is_ece_eligible",
    "make_result_row",
    "mcnemar",
    "mean_std",
    "paired_bootstrap_diff",
    "robustness_metrics",
    "transfer_metrics",
    "validate_result_row",
    "write_result_rows",
]
