"""fnb.data — acquisition, binarization, preprocessing, dedup, splits, stats."""

from __future__ import annotations

from .acquire import (
    DatasetSnapshot,
    acquire_snapshots,
    discover_report,
    hash_directory,
    list_input_datasets,
    resolve_input_path,
)
from .binarize import (
    BinarizeResult,
    MappingReportRow,
    binarize_all,
    binarize_dataframe,
    binarize_dataset,
    map_label,
)

__all__ = [
    "BinarizeResult",
    "DatasetSnapshot",
    "MappingReportRow",
    "acquire_snapshots",
    "binarize_all",
    "binarize_dataframe",
    "binarize_dataset",
    "discover_report",
    "hash_directory",
    "list_input_datasets",
    "map_label",
    "resolve_input_path",
]
