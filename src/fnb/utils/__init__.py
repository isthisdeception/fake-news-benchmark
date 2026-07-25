"""Utility subpackage: determinism/seeding, logging, hashing, run registry, IO."""

from __future__ import annotations

from .hashing import (
    sha256_bytes,
    sha256_dataframe,
    sha256_file,
    sha256_json,
    sha256_str,
)
from .io import (
    append_csv_row,
    ensure_dir,
    load_indices,
    read_csv,
    read_json,
    read_parquet,
    save_indices,
    write_csv,
    write_json,
    write_parquet,
)
from .logging_utils import get_logger, setup_run_logging
from .run_registry import (
    RunContext,
    collect_environment,
    finish_run,
    find_git_root,
    git_commit_sha,
    git_is_dirty,
    make_experiment_id,
    make_run_id,
    start_run,
    write_metadata,
)
from .seeding import (
    CUBLAS_WORKSPACE_CONFIG,
    SeedBundle,
    seed_worker,
    set_global_seed,
)

__all__ = [
    # seeding
    "set_global_seed",
    "SeedBundle",
    "seed_worker",
    "CUBLAS_WORKSPACE_CONFIG",
    # hashing
    "sha256_bytes",
    "sha256_str",
    "sha256_file",
    "sha256_json",
    "sha256_dataframe",
    # io
    "ensure_dir",
    "read_json",
    "write_json",
    "read_csv",
    "write_csv",
    "append_csv_row",
    "read_parquet",
    "write_parquet",
    "save_indices",
    "load_indices",
    # logging
    "setup_run_logging",
    "get_logger",
    # run registry
    "RunContext",
    "start_run",
    "finish_run",
    "make_run_id",
    "make_experiment_id",
    "collect_environment",
    "find_git_root",
    "git_commit_sha",
    "git_is_dirty",
    "write_metadata",
]
