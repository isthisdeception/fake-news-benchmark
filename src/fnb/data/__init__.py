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
from .dedup import DedupResult, dedup_all, dedup_dataframe
from .preprocess import (
    PreprocessResult,
    clean_text_classical,
    clean_text_neural,
    preprocess_all,
)
from .splits import SplitRunResult, create_all_splits, stratified_random_split

__all__ = [
    "BinarizeResult",
    "DatasetSnapshot",
    "DedupResult",
    "MappingReportRow",
    "PreprocessResult",
    "SplitRunResult",
    "acquire_snapshots",
    "binarize_all",
    "binarize_dataframe",
    "binarize_dataset",
    "clean_text_classical",
    "clean_text_neural",
    "create_all_splits",
    "dedup_all",
    "dedup_dataframe",
    "discover_report",
    "hash_directory",
    "list_input_datasets",
    "map_label",
    "preprocess_all",
    "resolve_input_path",
    "stratified_random_split",
]
