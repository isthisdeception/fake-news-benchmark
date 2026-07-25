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

__all__ = [
    "DatasetSnapshot",
    "acquire_snapshots",
    "discover_report",
    "hash_directory",
    "list_input_datasets",
    "resolve_input_path",
]
