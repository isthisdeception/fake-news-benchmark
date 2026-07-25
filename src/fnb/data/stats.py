"""EXP-P4: dataset statistics from committed split indices (protocol §4.4).

Records per dataset / version / split: N, class counts/ratio, mean/median
article length (chars + words). Versions: ``vBIN`` and ``vDEDUP``.

Split indices (``S-RAND``) address **vDEDUP** row positions. For ``vBIN``, the
same logical split is obtained via ``vDEDUP.pre_dedup_index`` (survivor → vBIN
row). A ``split=full`` row is also written for each dataset×version (whole
corpus, no seed).
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fnb.config import load_config, load_config_raw
from fnb.utils.io import load_indices, write_csv

logger = logging.getLogger("fnb.data.stats")

DEFAULT_PROCESSED_DIR = Path("data/processed")
DEFAULT_SPLITS_DIR = Path("data/splits")
DEFAULT_REPORT_PATH = Path("results/dataset_stats.csv")
VERSIONS = ("vBIN", "vDEDUP")
PARTS = ("train", "val", "test")


@dataclass
class StatsRow:
    dataset_id: str
    dataset_version: str
    regime: str
    seed: str  # "" for full-corpus rows
    split: str  # train|val|test|full
    n: int
    n_real: int
    n_fake: int
    fake_ratio: float
    mean_chars: float
    median_chars: float
    mean_words: float
    median_words: float


def article_lengths(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return (char_lengths, word_lengths) for title+text rows."""
    n = len(df)
    if "title" in df.columns:
        title = df["title"].fillna("").astype(str)
    else:
        title = pd.Series([""] * n, index=df.index)
    if "text" in df.columns:
        text = df["text"].fillna("").astype(str)
    else:
        text = pd.Series([""] * n, index=df.index)
    combined = (title + " " + text).str.strip()
    chars = combined.str.len().to_numpy(dtype=np.float64)
    words = combined.str.split().str.len().fillna(0).to_numpy(dtype=np.float64)
    return chars, words


def summarize_subset(df: pd.DataFrame) -> dict[str, Any]:
    """Compute N / class / length stats for a dataframe subset."""
    n = len(df)
    if n == 0:
        return {
            "n": 0,
            "n_real": 0,
            "n_fake": 0,
            "fake_ratio": float("nan"),
            "mean_chars": float("nan"),
            "median_chars": float("nan"),
            "mean_words": float("nan"),
            "median_words": float("nan"),
        }
    labels = df["label"].to_numpy()
    n_real = int((labels == 0).sum())
    n_fake = int((labels == 1).sum())
    chars, words = article_lengths(df)
    return {
        "n": n,
        "n_real": n_real,
        "n_fake": n_fake,
        "fake_ratio": round(n_fake / n, 6),
        "mean_chars": round(float(np.mean(chars)), 3),
        "median_chars": round(float(np.median(chars)), 3),
        "mean_words": round(float(np.mean(words)), 3),
        "median_words": round(float(np.median(words)), 3),
    }


def _row(
    *,
    dataset_id: str,
    dataset_version: str,
    regime: str,
    seed: str,
    split: str,
    stats: dict[str, Any],
) -> StatsRow:
    return StatsRow(
        dataset_id=dataset_id,
        dataset_version=dataset_version,
        regime=regime,
        seed=seed,
        split=split,
        **stats,
    )


def stats_for_dataset(
    dataset_id: str,
    *,
    processed_dir: str | Path = DEFAULT_PROCESSED_DIR,
    splits_dir: str | Path = DEFAULT_SPLITS_DIR,
    seeds: Iterable[int],
    regime: str = "S-RAND",
) -> list[StatsRow]:
    """Compute stats rows for one dataset (vBIN + vDEDUP)."""
    processed_dir = Path(processed_dir)
    splits_dir = Path(splits_dir)
    rows: list[StatsRow] = []

    paths = {v: processed_dir / f"{dataset_id}_{v}.parquet" for v in VERSIONS}
    frames: dict[str, pd.DataFrame] = {}
    for ver, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(
                f"{dataset_id}: missing {path} — run prior pipeline stages first"
            )
        frames[ver] = pd.read_parquet(path)

    # Full-corpus rows (no seed)
    for ver, df in frames.items():
        rows.append(
            _row(
                dataset_id=dataset_id,
                dataset_version=ver,
                regime="full",
                seed="",
                split="full",
                stats=summarize_subset(df),
            )
        )

    vdedup = frames["vDEDUP"]
    if "pre_dedup_index" not in vdedup.columns:
        raise KeyError(
            f"{dataset_id}: vDEDUP missing pre_dedup_index (needed to map splits → vBIN)"
        )
    pre_idx = vdedup["pre_dedup_index"].to_numpy(dtype=int)

    for seed in seeds:
        parts_idx: dict[str, list[int]] = {}
        for part in PARTS:
            path = splits_dir / f"{dataset_id}_{regime}_{part}_{seed}.idx"
            if not path.is_file():
                raise FileNotFoundError(
                    f"missing split index {path} — run EXP-P3 (--stage P3) first"
                )
            parts_idx[part] = load_indices(path)

        # vDEDUP: indices address vDEDUP rows directly
        for part, idx in parts_idx.items():
            subset = vdedup.iloc[idx]
            rows.append(
                _row(
                    dataset_id=dataset_id,
                    dataset_version="vDEDUP",
                    regime=regime,
                    seed=str(seed),
                    split=part,
                    stats=summarize_subset(subset),
                )
            )

        # vBIN: map via pre_dedup_index (same logical split, raw text lengths)
        vbin = frames["vBIN"]
        for part, idx in parts_idx.items():
            bin_rows = pre_idx[np.asarray(idx, dtype=int)]
            # Guard against out-of-range
            if bin_rows.max(initial=-1) >= len(vbin) or bin_rows.min(initial=0) < 0:
                raise IndexError(
                    f"{dataset_id}: pre_dedup_index out of range for vBIN (n={len(vbin)})"
                )
            subset = vbin.iloc[bin_rows]
            rows.append(
                _row(
                    dataset_id=dataset_id,
                    dataset_version="vBIN",
                    regime=regime,
                    seed=str(seed),
                    split=part,
                    stats=summarize_subset(subset),
                )
            )

    return rows


def compute_dataset_stats(
    *,
    processed_dir: str | Path = DEFAULT_PROCESSED_DIR,
    splits_dir: str | Path = DEFAULT_SPLITS_DIR,
    report_path: str | Path = DEFAULT_REPORT_PATH,
    config_dir: str | Path | None = None,
    dataset_ids: list[str] | None = None,
    seeds: list[int] | None = None,
) -> pd.DataFrame:
    """Run EXP-P4 for all datasets; write ``results/dataset_stats.csv``."""
    protocol = load_config("protocol", config_dir)
    if seeds is None:
        seeds = list(protocol.seeds.primary)

    raw = load_config_raw("datasets", config_dir)
    ids = dataset_ids or list((raw.get("datasets") or {}).keys())

    all_rows: list[StatsRow] = []
    for ds_id in ids:
        ds_rows = stats_for_dataset(
            ds_id,
            processed_dir=processed_dir,
            splits_dir=splits_dir,
            seeds=seeds,
        )
        all_rows.extend(ds_rows)
        logger.info("%s: %d stats rows", ds_id, len(ds_rows))

    df = pd.DataFrame([asdict(r) for r in all_rows])
    write_csv(df, report_path, index=False)
    logger.info("Wrote dataset stats → %s (%d rows)", report_path, len(df))
    return df


__all__ = [
    "StatsRow",
    "article_lengths",
    "compute_dataset_stats",
    "stats_for_dataset",
    "summarize_subset",
]
