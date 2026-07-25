"""Canonical result-row schema + append-only CSV writer (eval R3/R4).

Every result row must carry full provenance:

* ``protocol_version`` (= ``v1.0``)
* ``dataset_version_tag``, ``split_type``, ``model_id``, ``seed``
* ``run_id``, ``config_hash``, ``git_sha``

plus metric columns. Aggregates are **mean ± std across seeds** — never
best-run-only (R3). Writers **append**; they never overwrite an existing CSV.
"""

from __future__ import annotations

import csv
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fnb.config.schema import PROTOCOL_VERSION
from fnb.evaluation.metrics import mean_std
from fnb.utils.io import ensure_dir

# Frozen provenance columns (evaluation_protocol R4; reproducibility §10).
PROVENANCE_COLUMNS: tuple[str, ...] = (
    "protocol_version",
    "dataset_version_tag",
    "split_type",
    "model_id",
    "seed",
    "run_id",
    "config_hash",
    "git_sha",
)

# Grouping keys for seed aggregation (seed itself is aggregated away).
AGGREGATE_KEYS: tuple[str, ...] = (
    "protocol_version",
    "dataset_version_tag",
    "split_type",
    "model_id",
)


class ResultSchemaError(ValueError):
    """Raised when a result row fails provenance / schema validation."""


def validate_result_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and normalize one result row; reject incomplete provenance."""
    if not isinstance(row, Mapping):
        raise ResultSchemaError("result row must be a mapping")

    missing = [c for c in PROVENANCE_COLUMNS if c not in row or row[c] is None or row[c] == ""]
    if missing:
        raise ResultSchemaError(
            f"result row missing required provenance columns: {missing}"
        )

    proto = str(row["protocol_version"])
    if proto != PROTOCOL_VERSION:
        raise ResultSchemaError(
            f"protocol_version must be {PROTOCOL_VERSION!r}, got {proto!r}"
        )

    out = dict(row)
    out["protocol_version"] = PROTOCOL_VERSION
    out["dataset_version_tag"] = str(out["dataset_version_tag"])
    out["split_type"] = str(out["split_type"])
    out["model_id"] = str(out["model_id"])
    out["run_id"] = str(out["run_id"])
    out["config_hash"] = str(out["config_hash"])
    out["git_sha"] = str(out["git_sha"])
    # Keep seed as int when possible for stable CSV typing.
    try:
        out["seed"] = int(out["seed"])
    except (TypeError, ValueError) as exc:
        raise ResultSchemaError(f"seed must be an integer, got {out['seed']!r}") from exc
    return out


def _column_order(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    extras: list[str] = []
    seen = set(PROVENANCE_COLUMNS)
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                extras.append(str(key))
    return list(PROVENANCE_COLUMNS) + extras


def write_result_rows(
    csv_path: str | Path,
    rows: Sequence[Mapping[str, Any]],
) -> Path:
    """Validate ``rows`` and **append** them to ``csv_path`` (never overwrite).

    If the file already exists, new rows must supply every existing column.
    New metric columns are allowed only when creating a fresh file (or when the
    existing file is empty).
    """
    if not rows:
        raise ResultSchemaError("rows must be non-empty")

    validated = [validate_result_row(r) for r in rows]
    path = Path(csv_path)
    ensure_dir(path.parent)

    if path.exists() and path.stat().st_size > 0:
        with path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            if reader.fieldnames is None:
                raise ResultSchemaError(f"existing CSV has no header: {path}")
            fieldnames = list(reader.fieldnames)
        for i, row in enumerate(validated):
            missing = [c for c in fieldnames if c not in row]
            if missing:
                raise ResultSchemaError(
                    f"row[{i}] missing columns required by existing CSV {path}: {missing}"
                )
        with path.open("a", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
            for row in validated:
                writer.writerow({k: row.get(k, "") for k in fieldnames})
    else:
        fieldnames = _column_order(validated)
        with path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for row in validated:
                writer.writerow({k: row.get(k, "") for k in fieldnames})
    return path


def _is_metric_column(name: str) -> bool:
    if name in PROVENANCE_COLUMNS:
        return False
    if name in AGGREGATE_KEYS:
        return False
    return True


def aggregate_seeds(
    df: pd.DataFrame,
    *,
    metric_columns: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Aggregate per-seed rows to mean ± std (R3).

    Groups by ``(protocol_version, dataset_version_tag, split_type, model_id)``.
    For each metric column emits ``{metric}_mean``, ``{metric}_std``, ``{metric}_n``.
    Never selects a best run.
    """
    if df is None or len(df) == 0:
        raise ResultSchemaError("dataframe must be non-empty")

    required = list(AGGREGATE_KEYS) + ["seed"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ResultSchemaError(f"dataframe missing columns for aggregation: {missing}")

    if metric_columns is None:
        metrics = [c for c in df.columns if _is_metric_column(c)]
    else:
        metrics = list(metric_columns)
        for m in metrics:
            if m not in df.columns:
                raise ResultSchemaError(f"metric column not in dataframe: {m}")

    if not metrics:
        raise ResultSchemaError("no metric columns to aggregate")

    records: list[dict[str, Any]] = []
    grouped = df.groupby(list(AGGREGATE_KEYS), dropna=False, sort=True)
    for keys, group in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)
        row: dict[str, Any] = dict(zip(AGGREGATE_KEYS, keys, strict=True))
        for metric in metrics:
            vals = pd.to_numeric(group[metric], errors="coerce").to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                row[f"{metric}_mean"] = float("nan")
                row[f"{metric}_std"] = float("nan")
                row[f"{metric}_n"] = 0
            else:
                stats = mean_std(vals.tolist())
                row[f"{metric}_mean"] = stats["mean"]
                row[f"{metric}_std"] = stats["std"]
                row[f"{metric}_n"] = stats["n"]
        records.append(row)

    return pd.DataFrame.from_records(records)


def make_result_row(
    *,
    dataset_version_tag: str,
    split_type: str,
    model_id: str,
    seed: int,
    run_id: str,
    config_hash: str,
    git_sha: str,
    metrics: Mapping[str, Any] | None = None,
    protocol_version: str = PROTOCOL_VERSION,
    **extra: Any,
) -> dict[str, Any]:
    """Build a validated result row with provenance stamped."""
    row: dict[str, Any] = {
        "protocol_version": protocol_version,
        "dataset_version_tag": dataset_version_tag,
        "split_type": split_type,
        "model_id": model_id,
        "seed": seed,
        "run_id": run_id,
        "config_hash": config_hash,
        "git_sha": git_sha,
    }
    if metrics:
        row.update(dict(metrics))
    if extra:
        row.update(extra)
    return validate_result_row(row)


__all__ = [
    "AGGREGATE_KEYS",
    "PROVENANCE_COLUMNS",
    "ResultSchemaError",
    "aggregate_seeds",
    "make_result_row",
    "validate_result_row",
    "write_result_rows",
]
