"""EXP-P5: cross-dataset overlap removal (protocol §4.5).

For every directed transfer pair:

* Off-diagonal among ``article_transfer_set`` {DS1, DS2, DS3}
* Each article-transfer source → each ``domain_shift_probe`` (DS→DS4)

Remove **test-side** items whose TF-IDF cosine ≥ threshold (frozen 0.90 in
``configs/dedup.yaml``) to **any** training-side item. Training-side items are
never removed.

Inputs are full ``{DSx}_vDEDUP.parquet`` corpora (dataset-level
``vXDEDUP(train→test)``; not seed-dependent). Outputs:

* Cleaned test indices into the **test** dataset's vDEDUP row positions::

      data/splits/{train}_to_{test}_vXDEDUP_test.idx

* Overlap counts + residual class balance in
  ``results/interdataset_overlap.csv`` (WELFake↔ISOT / DS1↔DS2 flagged —
  they share Reuters real news; W10).
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from fnb.config import load_config, load_config_raw
from fnb.data.dedup import across_dataset_keep_mask, combine_title_text
from fnb.utils.io import ensure_dir, save_indices, write_csv

logger = logging.getLogger("fnb.data.xdedup")

DEFAULT_PROCESSED_DIR = Path("data/processed")
DEFAULT_SPLITS_DIR = Path("data/splits")
DEFAULT_REPORT_PATH = Path("results/interdataset_overlap.csv")
DEFAULT_BLOCK_SIZE = 512

# Same-source control pair (WELFake ↔ ISOT share Reuters); always report residual.
WELFAKE_ISOT_PAIR = frozenset({"DS1", "DS2"})


@dataclass
class OverlapCounts:
    """One row of ``results/interdataset_overlap.csv``."""

    train_dataset_id: str
    test_dataset_id: str
    threshold: float
    side_removed: str
    n_train: int
    n_test_before: int
    n_overlap_removed: int
    n_test_after: int
    pct_removed: float
    n_real_before: int
    n_fake_before: int
    fake_ratio_before: float
    n_real_after: int
    n_fake_after: int
    fake_ratio_after: float
    version_tag: str
    same_source_control: bool
    notes: str = ""


@dataclass
class XDedupResult:
    train_dataset_id: str
    test_dataset_id: str
    kept_test_indices: list[int]
    counts: OverlapCounts
    index_path: Path | None = None


def directed_transfer_pairs(
    *,
    article_transfer_set: list[str],
    domain_shift_probe: list[str],
) -> list[tuple[str, str]]:
    """Enumerate directed (train, test) pairs for EXP-P5 / EXP-A1–A2.

    Off-diagonal article pairs + each article source → each domain-shift probe.
    DS5 (LIAR) is never included (§4.2).
    """
    pairs: list[tuple[str, str]] = []
    for train_id in article_transfer_set:
        for test_id in article_transfer_set:
            if train_id != test_id:
                pairs.append((train_id, test_id))
    for train_id in article_transfer_set:
        for probe_id in domain_shift_probe:
            pairs.append((train_id, probe_id))
    return pairs


def xdedup_index_path(
    train_dataset_id: str,
    test_dataset_id: str,
    *,
    splits_dir: str | Path = DEFAULT_SPLITS_DIR,
) -> Path:
    """Path for cleaned test indices: ``{train}_to_{test}_vXDEDUP_test.idx``."""
    return Path(splits_dir) / f"{train_dataset_id}_to_{test_dataset_id}_vXDEDUP_test.idx"


def _class_counts(labels: np.ndarray | pd.Series) -> tuple[int, int, float]:
    y = np.asarray(labels)
    n = len(y)
    n_real = int((y == 0).sum())
    n_fake = int((y == 1).sum())
    ratio = round(n_fake / n, 6) if n else float("nan")
    return n_real, n_fake, ratio


def _pair_note(train_id: str, test_id: str) -> tuple[bool, str]:
    ids = {train_id, test_id}
    if ids == WELFAKE_ISOT_PAIR:
        return (
            True,
            "WELFake↔ISOT same-source control (share Reuters real news; W10) — "
            "residual test size and class balance after removal",
        )
    if test_id == "DS4":
        return False, "domain-shift probe target (EXP-A2)"
    return False, ""


def xdedup_pair(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    train_dataset_id: str,
    test_dataset_id: str,
    threshold: float,
    version_tag: str = "vXDEDUP",
    side_removed: str = "test",
    block_size: int = DEFAULT_BLOCK_SIZE,
) -> XDedupResult:
    """Remove test-side overlaps against the full training corpus.

    Indices in ``kept_test_indices`` are 0-based positions into ``test_df``
    (vDEDUP row order). Does not use labels for keep/drop decisions.
    """
    if side_removed != "test":
        raise ValueError(
            f"protocol §4.5 requires side_removed='test', got {side_removed!r}"
        )
    if "label" in train_df.columns:
        _ = train_df["label"]  # explicit non-use for keep/drop
    if "label" in test_df.columns:
        _ = test_df["label"]

    train_work = train_df.reset_index(drop=True)
    test_work = test_df.reset_index(drop=True)
    train_texts = combine_title_text(train_work).tolist()
    test_texts = combine_title_text(test_work).tolist()

    keep = across_dataset_keep_mask(
        train_texts,
        test_texts,
        threshold=threshold,
        block_size=block_size,
    )
    kept_idx = np.flatnonzero(keep).astype(int).tolist()
    n_before = len(test_work)
    n_after = len(kept_idx)
    n_removed = n_before - n_after
    pct = (100.0 * n_removed / n_before) if n_before else 0.0

    labels_before = test_work["label"] if "label" in test_work.columns else pd.Series(dtype=int)
    labels_after = test_work.loc[keep, "label"] if "label" in test_work.columns else pd.Series(dtype=int)
    n_real_b, n_fake_b, ratio_b = _class_counts(labels_before)
    n_real_a, n_fake_a, ratio_a = _class_counts(labels_after)
    same_src, note = _pair_note(train_dataset_id, test_dataset_id)

    counts = OverlapCounts(
        train_dataset_id=train_dataset_id,
        test_dataset_id=test_dataset_id,
        threshold=float(threshold),
        side_removed=side_removed,
        n_train=len(train_work),
        n_test_before=n_before,
        n_overlap_removed=n_removed,
        n_test_after=n_after,
        pct_removed=round(pct, 4),
        n_real_before=n_real_b,
        n_fake_before=n_fake_b,
        fake_ratio_before=ratio_b,
        n_real_after=n_real_a,
        n_fake_after=n_fake_a,
        fake_ratio_after=ratio_a,
        version_tag=version_tag,
        same_source_control=same_src,
        notes=note,
    )
    return XDedupResult(
        train_dataset_id=train_dataset_id,
        test_dataset_id=test_dataset_id,
        kept_test_indices=kept_idx,
        counts=counts,
    )


def xdedup_datasets(
    train_dataset_id: str,
    test_dataset_id: str,
    *,
    processed_dir: str | Path = DEFAULT_PROCESSED_DIR,
    splits_dir: str | Path = DEFAULT_SPLITS_DIR,
    threshold: float | None = None,
    version_tag: str | None = None,
    config_dir: str | Path | None = None,
    block_size: int = DEFAULT_BLOCK_SIZE,
) -> XDedupResult:
    """Load vDEDUP pair → cross-dedup → write cleaned test ``.idx``."""
    cfg = load_config("dedup", config_dir)
    across = cfg.across_dataset
    thr = float(across.threshold if threshold is None else threshold)
    tag = version_tag or str(across.version_tag)
    side = str(across.side_removed)

    processed_dir = Path(processed_dir)
    train_path = processed_dir / f"{train_dataset_id}_vDEDUP.parquet"
    test_path = processed_dir / f"{test_dataset_id}_vDEDUP.parquet"
    if not train_path.is_file():
        raise FileNotFoundError(
            f"{train_dataset_id}: missing {train_path} — run EXP-P2 (--stage P2) first"
        )
    if not test_path.is_file():
        raise FileNotFoundError(
            f"{test_dataset_id}: missing {test_path} — run EXP-P2 (--stage P2) first"
        )

    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)
    result = xdedup_pair(
        train_df,
        test_df,
        train_dataset_id=train_dataset_id,
        test_dataset_id=test_dataset_id,
        threshold=thr,
        version_tag=tag,
        side_removed=side,
        block_size=block_size,
    )
    out = xdedup_index_path(
        train_dataset_id, test_dataset_id, splits_dir=splits_dir
    )
    save_indices(result.kept_test_indices, out)
    result.index_path = out
    logger.info(
        "%s→%s vXDEDUP: kept %d/%d test (removed %d, %.2f%%) thr=%.2f → %s",
        train_dataset_id,
        test_dataset_id,
        result.counts.n_test_after,
        result.counts.n_test_before,
        result.counts.n_overlap_removed,
        result.counts.pct_removed,
        thr,
        out,
    )
    return result


def write_overlap_report(
    counts: list[OverlapCounts],
    path: str | Path = DEFAULT_REPORT_PATH,
) -> Path:
    """Write ``results/interdataset_overlap.csv``."""
    df = pd.DataFrame([asdict(c) for c in counts])
    return write_csv(df, path, index=False)


def xdedup_all(
    *,
    processed_dir: str | Path = DEFAULT_PROCESSED_DIR,
    splits_dir: str | Path = DEFAULT_SPLITS_DIR,
    report_path: str | Path | None = None,
    config_dir: str | Path | None = None,
    block_size: int = DEFAULT_BLOCK_SIZE,
) -> list[XDedupResult]:
    """Run EXP-P5 for all directed transfer pairs; write indices + overlap CSV."""
    dedup_cfg = load_config("dedup", config_dir)
    across = dedup_cfg.across_dataset
    raw = load_config_raw("datasets", config_dir)
    article = list(raw.get("article_transfer_set") or [])
    probe = list(raw.get("domain_shift_probe") or [])
    pairs = directed_transfer_pairs(
        article_transfer_set=article,
        domain_shift_probe=probe,
    )
    if not pairs:
        raise ValueError("no transfer pairs found in datasets.yaml")

    out_report = Path(report_path) if report_path is not None else Path(str(across.report_to))
    ensure_dir(splits_dir)

    results: list[XDedupResult] = []
    for train_id, test_id in pairs:
        results.append(
            xdedup_datasets(
                train_id,
                test_id,
                processed_dir=processed_dir,
                splits_dir=splits_dir,
                config_dir=config_dir,
                block_size=block_size,
            )
        )
    write_overlap_report([r.counts for r in results], out_report)
    logger.info("Wrote interdataset overlap report → %s (%d pairs)", out_report, len(results))
    return results


__all__ = [
    "OverlapCounts",
    "XDedupResult",
    "directed_transfer_pairs",
    "write_overlap_report",
    "xdedup_all",
    "xdedup_datasets",
    "xdedup_index_path",
    "xdedup_pair",
]
