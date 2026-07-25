"""EXP-P2: within-dataset exact + near-duplicate removal (protocol §4.5).

Pipeline per dataset:

1. Load ``{DSx}_vCLEAN-C.parquet`` (classical track — TF-IDF cosine matches the
   classical feature family; one ``vDEDUP`` aligns both families via shared ``uid``).
2. Remove **exact** duplicates (SHA-256 of combined title+text), keep first.
3. Remove **near** duplicates with TF-IDF cosine ≥ threshold from
   ``configs/dedup.yaml`` (frozen 0.95), keep first.
4. Write ``{DSx}_vDEDUP.parquet`` + append row to ``results/dedup_counts.csv``.

Does **not** peek at labels. Uses sparse TF-IDF + blocked cosine to avoid a
dense ``n×n`` similarity matrix.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from fnb.config import load_config, load_config_raw
from fnb.utils.io import ensure_dir, write_csv, write_parquet

logger = logging.getLogger("fnb.data.dedup")

DEFAULT_OUTPUT_DIR = Path("data/processed")
DEFAULT_REPORT_PATH = Path("results/dedup_counts.csv")
DEFAULT_BLOCK_SIZE = 512


@dataclass
class DedupCounts:
    """One row of ``results/dedup_counts.csv``."""

    dataset_id: str
    n_input: int
    n_after_exact: int
    n_removed_exact: int
    n_kept: int
    n_removed_near: int
    n_removed_total: int
    pct_removed: float
    near_duplicate_threshold: float
    version_tag: str
    text_source: str = "vCLEAN-C"
    notes: str = ""


@dataclass
class DedupResult:
    dataset_id: str
    dataframe: pd.DataFrame
    counts: DedupCounts
    survivor_indices: list[int] = field(default_factory=list)
    output_path: Path | None = None


def combine_title_text(df: pd.DataFrame) -> pd.Series:
    """Build the string used for exact-hash and TF-IDF (title + text)."""
    title = df["title"] if "title" in df.columns else pd.Series([""] * len(df))
    text = df["text"] if "text" in df.columns else pd.Series([""] * len(df))
    title = title.fillna("").astype(str)
    text = text.fillna("").astype(str)
    combined = (title + " " + text).str.strip()
    # If both empty, keep empty string (exact-dup among empties → keep first).
    return combined


def exact_duplicate_mask(texts: pd.Series | list[str]) -> np.ndarray:
    """Return boolean keep-mask: True for first occurrence of each exact hash."""
    keep = np.ones(len(texts), dtype=bool)
    seen: set[str] = set()
    for i, t in enumerate(texts):
        digest = hashlib.sha256(t.encode("utf-8")).hexdigest()
        if digest in seen:
            keep[i] = False
        else:
            seen.add(digest)
    return keep


def near_duplicate_mask(
    texts: list[str],
    *,
    threshold: float,
    block_size: int = DEFAULT_BLOCK_SIZE,
) -> np.ndarray:
    """Keep-mask after near-dup removal (TF-IDF cosine ≥ ``threshold``).

    Greedy: for each kept row ``i`` (in order), drop later rows ``j > i`` with
    cosine(i, j) ≥ threshold. Blocked ``X_block @ X.T`` avoids a full dense
    ``n×n`` matrix.
    """
    n = len(texts)
    keep = np.ones(n, dtype=bool)
    if n <= 1:
        return keep

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=1,
        max_features=50_000,
        dtype=np.float32,
    )
    try:
        x = vectorizer.fit_transform(texts)
    except ValueError:
        # All empty / no tokens — nothing to compare; keep all (exact pass already ran).
        return keep

    x = normalize(x, norm="l2", copy=False)
    # Process in row blocks; only mark j > i.
    for start in range(0, n, block_size):
        end = min(start + block_size, n)
        # Skip entirely-dropped blocks.
        if not keep[start:end].any():
            continue
        sims = (x[start:end] @ x.T).toarray()  # (block, n) float32
        for bi, i in enumerate(range(start, end)):
            if not keep[i]:
                continue
            # Candidates after i
            if i + 1 >= n:
                continue
            row = sims[bi, i + 1 :]
            dup_local = np.where(row >= threshold)[0]
            if dup_local.size:
                keep[dup_local + (i + 1)] = False
    return keep


def dedup_dataframe(
    df: pd.DataFrame,
    *,
    dataset_id: str,
    threshold: float,
    version_tag: str = "vDEDUP",
    block_size: int = DEFAULT_BLOCK_SIZE,
    remove_exact: bool = True,
) -> DedupResult:
    """Exact then near-dup removal; preserve original order of survivors.

    Adds ``pre_dedup_index`` (0-based position in the input frame) for split
    alignment. Does not use the ``label`` column.
    """
    if "label" in df.columns:
        # Explicit non-use: never condition keep/drop on labels.
        _ = df["label"]

    work = df.reset_index(drop=True).copy()
    work.insert(0, "pre_dedup_index", np.arange(len(work), dtype=np.int64))
    n_input = len(work)
    texts = combine_title_text(work)

    if remove_exact:
        exact_keep = exact_duplicate_mask(texts)
    else:
        exact_keep = np.ones(n_input, dtype=bool)
    n_after_exact = int(exact_keep.sum())
    n_removed_exact = n_input - n_after_exact

    # Near-dup only among exact survivors (re-index texts).
    exact_indices = np.flatnonzero(exact_keep)
    texts_exact = texts.iloc[exact_indices].tolist()
    near_keep_local = near_duplicate_mask(texts_exact, threshold=threshold, block_size=block_size)
    final_keep = np.zeros(n_input, dtype=bool)
    final_keep[exact_indices[near_keep_local]] = True

    survivors = work.loc[final_keep].reset_index(drop=True)
    survivor_indices = survivors["pre_dedup_index"].astype(int).tolist()
    n_kept = len(survivors)
    n_removed_near = n_after_exact - n_kept
    n_removed_total = n_input - n_kept
    pct = (100.0 * n_removed_total / n_input) if n_input else 0.0

    counts = DedupCounts(
        dataset_id=dataset_id,
        n_input=n_input,
        n_after_exact=n_after_exact,
        n_removed_exact=n_removed_exact,
        n_kept=n_kept,
        n_removed_near=n_removed_near,
        n_removed_total=n_removed_total,
        pct_removed=round(pct, 4),
        near_duplicate_threshold=float(threshold),
        version_tag=version_tag,
        text_source="vCLEAN-C",
        notes="",
    )
    return DedupResult(
        dataset_id=dataset_id,
        dataframe=survivors,
        counts=counts,
        survivor_indices=survivor_indices,
    )


def dedup_dataset(
    dataset_id: str,
    *,
    processed_dir: str | Path = DEFAULT_OUTPUT_DIR,
    output_dir: str | Path | None = None,
    threshold: float | None = None,
    version_tag: str | None = None,
    config_dir: str | Path | None = None,
    block_size: int = DEFAULT_BLOCK_SIZE,
) -> DedupResult:
    """Load vCLEAN-C → dedup → write ``{DSx}_vDEDUP.parquet``."""
    cfg = load_config("dedup", config_dir)
    within = cfg.within_dataset
    thr = float(within.near_duplicate_threshold if threshold is None else threshold)
    tag = version_tag or str(within.version_tag)
    remove_exact = bool(within.remove_exact_duplicates)

    processed_dir = Path(processed_dir)
    output_dir = Path(output_dir) if output_dir is not None else processed_dir
    src = processed_dir / f"{dataset_id}_vCLEAN-C.parquet"
    if not src.is_file():
        raise FileNotFoundError(f"{dataset_id}: missing {src} — run EXP-P1b (--stage P1b) first")

    df = pd.read_parquet(src)
    result = dedup_dataframe(
        df,
        dataset_id=dataset_id,
        threshold=thr,
        version_tag=tag,
        block_size=block_size,
        remove_exact=remove_exact,
    )
    out = output_dir / f"{dataset_id}_{tag}.parquet"
    write_parquet(result.dataframe, out)
    result.output_path = out
    logger.info(
        "%s dedup kept=%d/%d (exact-%d near-%d, %.2f%% removed) → %s",
        dataset_id,
        result.counts.n_kept,
        result.counts.n_input,
        result.counts.n_removed_exact,
        result.counts.n_removed_near,
        result.counts.pct_removed,
        out,
    )
    return result


def write_dedup_counts(
    counts: list[DedupCounts],
    path: str | Path = DEFAULT_REPORT_PATH,
) -> Path:
    """Write ``results/dedup_counts.csv``."""
    df = pd.DataFrame([asdict(c) for c in counts])
    return write_csv(df, path, index=False)


def dedup_all(
    *,
    processed_dir: str | Path = DEFAULT_OUTPUT_DIR,
    output_dir: str | Path | None = None,
    report_path: str | Path = DEFAULT_REPORT_PATH,
    config_dir: str | Path | None = None,
    dataset_ids: list[str] | None = None,
    block_size: int = DEFAULT_BLOCK_SIZE,
) -> list[DedupResult]:
    """Run EXP-P2 for all (or selected) datasets; write vDEDUP + counts CSV."""
    _ = load_config("dedup", config_dir)  # validate
    if dataset_ids is None:
        raw = load_config_raw("datasets", config_dir)
        dataset_ids = list((raw.get("datasets") or {}).keys())

    ensure_dir(Path(output_dir) if output_dir else processed_dir)
    results: list[DedupResult] = []
    for ds_id in dataset_ids:
        results.append(
            dedup_dataset(
                ds_id,
                processed_dir=processed_dir,
                output_dir=output_dir,
                config_dir=config_dir,
                block_size=block_size,
            )
        )
    write_dedup_counts([r.counts for r in results], report_path)
    logger.info("Wrote dedup counts → %s", report_path)
    return results


__all__ = [
    "DedupCounts",
    "DedupResult",
    "combine_title_text",
    "dedup_all",
    "dedup_dataframe",
    "dedup_dataset",
    "exact_duplicate_mask",
    "near_duplicate_mask",
    "write_dedup_counts",
]
