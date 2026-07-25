"""EXP-P3: stratified 70/15/15 splits — indices only (protocol §4.4).

Regimes:

* ``S-RAND`` — stratified random split (all datasets × primary seeds).
* ``S-SRC`` — source-disjoint split (DS1 only **iff** source recoverable).
* ``S-TEMP`` — temporal split (only where a reliable timestamp column exists).

Unavailable regimes are recorded as ``not_applicable`` in
``results/split_applicability.csv`` (never silently skipped).

Index files (into ``vDEDUP`` row positions ``0..n-1``)::

    data/splits/{DSx}_{S-RAND|S-SRC|S-TEMP}_{train|val|test}_{seed}.idx

Split BEFORE any resampling or classifier vectorizer fitting.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from fnb.config import load_config, load_config_raw
from fnb.utils.io import ensure_dir, save_indices, write_csv

logger = logging.getLogger("fnb.data.splits")

DEFAULT_PROCESSED_DIR = Path("data/processed")
DEFAULT_SPLITS_DIR = Path("data/splits")
DEFAULT_APPLICABILITY_PATH = Path("results/split_applicability.csv")

# Frozen protocol §4.4
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15

Regime = Literal["S-RAND", "S-SRC", "S-TEMP"]
Part = Literal["train", "val", "test"]
REGIMES: tuple[Regime, ...] = ("S-RAND", "S-SRC", "S-TEMP")
PARTS: tuple[Part, ...] = ("train", "val", "test")

_TIMESTAMP_CANDIDATES = ("timestamp", "date", "datetime", "published", "publish_date")


@dataclass
class SplitIndices:
    """Train/val/test index arrays (into vDEDUP positions)."""

    train: np.ndarray
    val: np.ndarray
    test: np.ndarray

    def as_dict(self) -> dict[str, np.ndarray]:
        return {"train": self.train, "val": self.val, "test": self.test}

    def validate_partition(self, n: int) -> None:
        all_idx = np.concatenate([self.train, self.val, self.test])
        if len(all_idx) != n:
            raise AssertionError(
                f"split size {len(all_idx)} != n={n} (train/val/test must cover all rows)"
            )
        if len(np.unique(all_idx)) != n:
            raise AssertionError("split indices are not disjoint or have duplicates")
        if set(all_idx.tolist()) != set(range(n)):
            raise AssertionError("split indices do not cover 0..n-1 exactly")


@dataclass
class ApplicabilityRow:
    dataset_id: str
    regime: str
    status: str  # "ok" | "not_applicable"
    note: str
    n_seeds_written: int = 0


@dataclass
class SplitRunResult:
    dataset_id: str
    regime: str
    seed: int
    splits: SplitIndices
    paths: dict[str, Path] = field(default_factory=dict)


def index_path(
    dataset_id: str,
    regime: Regime | str,
    part: Part | str,
    seed: int,
    *,
    splits_dir: str | Path = DEFAULT_SPLITS_DIR,
) -> Path:
    return Path(splits_dir) / f"{dataset_id}_{regime}_{part}_{seed}.idx"


def stratified_random_split(
    labels: np.ndarray | pd.Series,
    *,
    seed: int,
    train_frac: float = TRAIN_FRAC,
    val_frac: float = VAL_FRAC,
    test_frac: float = TEST_FRAC,
) -> SplitIndices:
    """Stratified 70/15/15 random split on row indices."""
    if abs(train_frac + val_frac + test_frac - 1.0) > 1e-9:
        raise ValueError("train/val/test fractions must sum to 1")
    y = np.asarray(labels)
    n = len(y)
    indices = np.arange(n)
    hold_frac = val_frac + test_frac
    idx_train, idx_hold, _y_train, y_hold = train_test_split(
        indices,
        y,
        test_size=hold_frac,
        random_state=seed,
        stratify=y,
    )
    relative_test = test_frac / hold_frac
    idx_val, idx_test, _, _ = train_test_split(
        idx_hold,
        y_hold,
        test_size=relative_test,
        random_state=seed,
        stratify=y_hold,
    )
    splits = SplitIndices(
        train=np.sort(idx_train.astype(int)),
        val=np.sort(idx_val.astype(int)),
        test=np.sort(idx_test.astype(int)),
    )
    splits.validate_partition(n)
    return splits


def source_disjoint_split(
    labels: np.ndarray | pd.Series,
    sources: np.ndarray | pd.Series,
    *,
    seed: int,
    train_frac: float = TRAIN_FRAC,
    val_frac: float = VAL_FRAC,
    test_frac: float = TEST_FRAC,
) -> SplitIndices:
    """Assign whole sources to train/val/test (no source shared across parts)."""
    y = np.asarray(labels)
    src = pd.Series(sources).astype(str).fillna("__MISSING_SOURCE__")
    n = len(y)
    _ = test_frac  # size remainder after train/val targets
    rng = np.random.RandomState(seed)

    groups: dict[str, list[int]] = {}
    for i, s in enumerate(src.tolist()):
        groups.setdefault(s, []).append(i)

    source_ids = list(groups.keys())
    rng.shuffle(source_ids)

    target_train = int(round(train_frac * n))
    target_val = int(round(val_frac * n))

    train_idx: list[int] = []
    val_idx: list[int] = []
    test_idx: list[int] = []
    n_train = n_val = 0

    for s in source_ids:
        members = groups[s]
        size = len(members)
        if n_train < target_train:
            train_idx.extend(members)
            n_train += size
        elif n_val < target_val:
            val_idx.extend(members)
            n_val += size
        else:
            test_idx.extend(members)

    if not train_idx or not val_idx or not test_idx:
        raise ValueError(
            "source-disjoint split failed: too few distinct sources to form "
            "non-empty train/val/test partitions"
        )

    splits = SplitIndices(
        train=np.sort(np.asarray(train_idx, dtype=int)),
        val=np.sort(np.asarray(val_idx, dtype=int)),
        test=np.sort(np.asarray(test_idx, dtype=int)),
    )
    splits.validate_partition(n)
    s_train = set(src.iloc[splits.train].tolist())
    s_val = set(src.iloc[splits.val].tolist())
    s_test = set(src.iloc[splits.test].tolist())
    if s_train & s_val or s_train & s_test or s_val & s_test:
        raise AssertionError("source-disjoint invariant violated")
    return splits


def temporal_split(
    timestamps: np.ndarray | pd.Series,
    *,
    train_frac: float = TRAIN_FRAC,
    val_frac: float = VAL_FRAC,
    test_frac: float = TEST_FRAC,
) -> SplitIndices:
    """Cut by sorted time: earliest → train, middle → val, latest → test."""
    _ = test_frac
    ts = pd.to_datetime(pd.Series(timestamps), utc=True, errors="coerce")
    if ts.isna().all():
        raise ValueError("no parseable timestamps for temporal split")
    order = np.lexsort((np.arange(len(ts)), ts.fillna(pd.Timestamp.max).astype("int64").to_numpy()))
    n = len(order)
    n_train = int(round(train_frac * n))
    n_val = int(round(val_frac * n))
    train = order[:n_train]
    val = order[n_train : n_train + n_val]
    test = order[n_train + n_val :]
    if len(test) == 0 or len(val) == 0 or len(train) == 0:
        raise ValueError("temporal split produced an empty partition")
    splits = SplitIndices(
        train=np.sort(train.astype(int)),
        val=np.sort(val.astype(int)),
        test=np.sort(test.astype(int)),
    )
    splits.validate_partition(n)
    return splits


def detect_timestamp_column(df: pd.DataFrame) -> str | None:
    cols = {c.lower(): c for c in df.columns}
    for name in _TIMESTAMP_CANDIDATES:
        if name in cols:
            return cols[name]
    return None


def source_recoverability(
    dataset_id: str,
    entry: dict[str, Any],
    df: pd.DataFrame,
) -> tuple[bool, str | None, str]:
    """Return ``(ok, source_column, note)`` for S-SRC applicability."""
    if dataset_id != "DS1":
        return (
            False,
            None,
            "S-SRC only defined for DS1 (protocol EXP-P3 / experiment_matrix)",
        )

    flag = entry.get("source_field_available", False)
    if flag is False or flag is None or str(flag).strip().lower() in {"", "false", "no", "0"}:
        return (
            False,
            None,
            "source_field_available=false in datasets.yaml — "
            "WELFake has no recoverable publisher/source id",
        )

    col = str(flag) if not isinstance(flag, bool) else "source"
    if col not in df.columns:
        for cand in ("source", "publisher", "subject"):
            if cand in df.columns:
                return True, cand, f"using column {cand!r} for source-disjoint split"
        return (
            False,
            None,
            f"configured source field {col!r} not present in vDEDUP columns",
        )
    return True, col, f"source-disjoint split using column {col!r}"


def write_split_files(
    dataset_id: str,
    regime: Regime | str,
    seed: int,
    splits: SplitIndices,
    *,
    splits_dir: str | Path = DEFAULT_SPLITS_DIR,
) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for part, arr in splits.as_dict().items():
        path = index_path(dataset_id, regime, part, seed, splits_dir=splits_dir)
        save_indices(arr.tolist(), path)
        paths[part] = path
    return paths


def create_splits_for_dataset(
    dataset_id: str,
    df: pd.DataFrame,
    entry: dict[str, Any],
    *,
    seeds: list[int],
    splits_dir: str | Path = DEFAULT_SPLITS_DIR,
) -> tuple[list[SplitRunResult], list[ApplicabilityRow]]:
    """Create all applicable regime×seed splits for one dataset."""
    if "label" not in df.columns:
        raise KeyError(f"{dataset_id}: vDEDUP frame missing 'label'")
    labels = df["label"].to_numpy()
    n = len(df)
    results: list[SplitRunResult] = []
    applicability: list[ApplicabilityRow] = []

    for seed in seeds:
        splits = stratified_random_split(labels, seed=seed)
        paths = write_split_files(dataset_id, "S-RAND", seed, splits, splits_dir=splits_dir)
        results.append(
            SplitRunResult(
                dataset_id=dataset_id,
                regime="S-RAND",
                seed=seed,
                splits=splits,
                paths=paths,
            )
        )
    applicability.append(
        ApplicabilityRow(
            dataset_id=dataset_id,
            regime="S-RAND",
            status="ok",
            note=(f"stratified {TRAIN_FRAC:.0%}/{VAL_FRAC:.0%}/{TEST_FRAC:.0%} on vDEDUP n={n}"),
            n_seeds_written=len(seeds),
        )
    )

    src_ok, src_col, src_note = source_recoverability(dataset_id, entry, df)
    if src_ok and src_col is not None:
        for seed in seeds:
            splits = source_disjoint_split(labels, df[src_col], seed=seed)
            paths = write_split_files(dataset_id, "S-SRC", seed, splits, splits_dir=splits_dir)
            results.append(
                SplitRunResult(
                    dataset_id=dataset_id,
                    regime="S-SRC",
                    seed=seed,
                    splits=splits,
                    paths=paths,
                )
            )
        applicability.append(
            ApplicabilityRow(
                dataset_id=dataset_id,
                regime="S-SRC",
                status="ok",
                note=src_note,
                n_seeds_written=len(seeds),
            )
        )
    else:
        applicability.append(
            ApplicabilityRow(
                dataset_id=dataset_id,
                regime="S-SRC",
                status="not_applicable",
                note=src_note,
                n_seeds_written=0,
            )
        )

    ts_col = detect_timestamp_column(df)
    if ts_col is not None:
        try:
            splits = temporal_split(df[ts_col])
            for seed in seeds:
                paths = write_split_files(dataset_id, "S-TEMP", seed, splits, splits_dir=splits_dir)
                results.append(
                    SplitRunResult(
                        dataset_id=dataset_id,
                        regime="S-TEMP",
                        seed=seed,
                        splits=splits,
                        paths=paths,
                    )
                )
            applicability.append(
                ApplicabilityRow(
                    dataset_id=dataset_id,
                    regime="S-TEMP",
                    status="ok",
                    note=f"temporal cut on column {ts_col!r}",
                    n_seeds_written=len(seeds),
                )
            )
        except ValueError as exc:
            applicability.append(
                ApplicabilityRow(
                    dataset_id=dataset_id,
                    regime="S-TEMP",
                    status="not_applicable",
                    note=f"timestamp column {ts_col!r} unusable: {exc}",
                    n_seeds_written=0,
                )
            )
    else:
        applicability.append(
            ApplicabilityRow(
                dataset_id=dataset_id,
                regime="S-TEMP",
                status="not_applicable",
                note="no reliable timestamp column in vDEDUP (protocol §4.4 honest N/A)",
                n_seeds_written=0,
            )
        )

    return results, applicability


def create_all_splits(
    *,
    processed_dir: str | Path = DEFAULT_PROCESSED_DIR,
    splits_dir: str | Path = DEFAULT_SPLITS_DIR,
    applicability_path: str | Path = DEFAULT_APPLICABILITY_PATH,
    config_dir: str | Path | None = None,
    dataset_ids: list[str] | None = None,
    seeds: list[int] | None = None,
) -> tuple[list[SplitRunResult], list[ApplicabilityRow]]:
    """Run EXP-P3 for all datasets; write ``.idx`` files + applicability CSV."""
    protocol = load_config("protocol", config_dir)
    if seeds is None:
        seeds = list(protocol.seeds.primary)

    raw_ds = load_config_raw("datasets", config_dir)
    datasets = raw_ds.get("datasets") or {}
    ids = dataset_ids or list(datasets.keys())

    ensure_dir(splits_dir)
    all_results: list[SplitRunResult] = []
    all_appl: list[ApplicabilityRow] = []

    for ds_id in ids:
        entry = dict(datasets.get(ds_id) or {})
        vdedup = Path(processed_dir) / f"{ds_id}_vDEDUP.parquet"
        if not vdedup.is_file():
            raise FileNotFoundError(f"{ds_id}: missing {vdedup} — run EXP-P2 (--stage P2) first")
        df = pd.read_parquet(vdedup)
        results, appl = create_splits_for_dataset(
            ds_id, df, entry, seeds=seeds, splits_dir=splits_dir
        )
        all_results.extend(results)
        all_appl.extend(appl)
        logger.info(
            "%s: wrote %d split files; applicability=%s",
            ds_id,
            sum(len(r.paths) for r in results),
            {a.regime: a.status for a in appl},
        )

    write_csv(pd.DataFrame([asdict(a) for a in all_appl]), applicability_path, index=False)
    logger.info("Wrote split applicability → %s", applicability_path)
    return all_results, all_appl


__all__ = [
    "PARTS",
    "REGIMES",
    "TEST_FRAC",
    "TRAIN_FRAC",
    "VAL_FRAC",
    "ApplicabilityRow",
    "SplitIndices",
    "SplitRunResult",
    "create_all_splits",
    "create_splits_for_dataset",
    "detect_timestamp_column",
    "index_path",
    "source_disjoint_split",
    "source_recoverability",
    "stratified_random_split",
    "temporal_split",
    "write_split_files",
]
