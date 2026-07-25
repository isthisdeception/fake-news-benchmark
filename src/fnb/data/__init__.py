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
from .preprocess import (
    PreprocessResult,
    clean_text_classical,
    clean_text_neural,
    preprocess_all,
)

__all__ = [
    "BinarizeResult",
    "DatasetSnapshot",
    "MappingReportRow",
    "PreprocessResult",
    "acquire_snapshots",
    "binarize_all",
    "binarize_dataframe",
    "binarize_dataset",
    "clean_text_classical",
    "clean_text_neural",
    "discover_report",
    "hash_directory",
    "list_input_datasets",
    "map_label",
    "preprocess_all",
    "resolve_input_path",
]
