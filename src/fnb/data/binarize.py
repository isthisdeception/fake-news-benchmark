"""EXP-P1a: frozen label binarization ``{real=0, fake=1}`` + mapping report.

Applies protocol §4.3 / EXP-P1a:

* Global label space ``{real = 0, fake = 1}`` (from ``configs/datasets.yaml``).
* LIAR 6-way → binary via the frozen map in ``datasets.DS5.liar_binary_map``.
* Dataset-specific source polarities / filename-derived labels for DS1–DS4.
* Rows with no clean binary mapping are **dropped and counted** (never silent).

Outputs:

* ``data/processed/{DSx}_vBIN.parquet``
* ``results/label_mapping_report.csv``
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from fnb.config import load_config_raw
from fnb.data.acquire import resolve_input_path
from fnb.utils.io import ensure_dir, write_csv, write_parquet

logger = logging.getLogger("fnb.data.binarize")

DEFAULT_OUTPUT_DIR = Path("data/processed")
DEFAULT_REPORT_PATH = Path("results/label_mapping_report.csv")

# Official LIAR TSV column order (Wang 2017); files have no header.
_LIAR_COLUMNS = [
    "id",
    "label",
    "statement",
    "subject",
    "speaker",
    "job_title",
    "state_info",
    "party",
    "barely_true_counts",
    "false_counts",
    "half_true_counts",
    "mostly_true_counts",
    "pants_on_fire_counts",
    "context",
]

_CANONICAL_COLUMNS = [
    "dataset_id",
    "uid",
    "title",
    "text",
    "label",
    "label_raw",
    "source_file",
]


@dataclass
class MappingReportRow:
    """One row of ``results/label_mapping_report.csv``."""

    dataset_id: str
    name: str
    n_raw: int
    n_kept: int
    n_dropped_ambiguous: int
    n_dropped_missing_label: int
    n_real: int
    n_fake: int
    applied_mapping: str
    label_source: str
    notes: str = ""


@dataclass
class BinarizeResult:
    """Per-dataset binarization output."""

    dataset_id: str
    dataframe: pd.DataFrame
    report: MappingReportRow
    output_path: Path | None = None
    extras: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Token normalization + mapping
# --------------------------------------------------------------------------- #
def normalize_label_token(value: Any) -> str | None:
    """Normalize a raw label cell to a comparable token, or None if empty."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, bool):
        return str(int(value))
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if pd.isna(value):
            return None
        # 0.0 / 1.0 → "0" / "1"
        if float(value).is_integer():
            return str(int(value))
        return str(value).strip().lower()
    text = str(value).strip().lower()
    if not text or text in {"nan", "none", "null", ""}:
        return None
    # LIAR / filename variants: pants_fire → pants-fire
    text = text.replace("_", "-").replace(" ", "-")
    return text


def build_token_to_binary(
    *,
    label_space: dict[str, int],
    fake_tokens: Iterable[Any],
    real_tokens: Iterable[Any],
) -> dict[str, int]:
    """Build ``normalized_token → {0,1}`` from fake/real token lists + label_space."""
    real_id = int(label_space["real"])
    fake_id = int(label_space["fake"])
    if real_id != 0 or fake_id != 1:
        raise ValueError(f"Frozen global label space must be real=0, fake=1; got {label_space!r}")
    mapping: dict[str, int] = {}
    for tok in real_tokens:
        key = normalize_label_token(tok)
        if key is not None:
            mapping[key] = real_id
    for tok in fake_tokens:
        key = normalize_label_token(tok)
        if key is not None:
            mapping[key] = fake_id
    return mapping


def map_label(value: Any, token_to_binary: dict[str, int]) -> int | None:
    """Map one raw label to global binary, or None if ambiguous / missing."""
    key = normalize_label_token(value)
    if key is None:
        return None
    return token_to_binary.get(key)


def apply_binary_map(
    label_raw: pd.Series,
    token_to_binary: dict[str, int],
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Return ``(label_binary, is_missing, is_ambiguous)`` boolean/int series.

    * missing: empty / NaN raw label
    * ambiguous: non-empty raw label with no mapping entry
    * label_binary: Int64 series with ``<NA>`` where dropped
    """
    tokens = label_raw.map(normalize_label_token)
    is_missing = tokens.isna()
    mapped = tokens.map(lambda t: token_to_binary.get(t) if t is not None else None)
    is_ambiguous = (~is_missing) & mapped.isna()
    label_binary = pd.Series(mapped, index=label_raw.index, dtype="Int64")
    return label_binary, is_missing, is_ambiguous


# --------------------------------------------------------------------------- #
# Path helpers
# --------------------------------------------------------------------------- #
def _find_file(root: Path, relative_or_name: str) -> Path | None:
    """Resolve ``relative_or_name`` under ``root``, falling back to basename search."""
    rel = relative_or_name.replace("\\", "/").lstrip("/")
    direct = root / rel
    if direct.is_file():
        return direct
    # Also try without a leading nested dirname (Kaggle flat vs nested upload).
    parts = Path(rel).parts
    if len(parts) > 1:
        alt = root / Path(*parts[1:])
        if alt.is_file():
            return alt
    basename = Path(rel).name
    matches = sorted(p for p in root.rglob(basename) if p.is_file())
    return matches[0] if matches else None


def _col(df: pd.DataFrame, name: str | None) -> str | None:
    """Case-insensitive column lookup; return actual column name or None."""
    if not name:
        return None
    if name in df.columns:
        return name
    lowered = {str(c).lower(): c for c in df.columns}
    return lowered.get(name.lower())


def _series_or_empty(df: pd.DataFrame, col_name: str | None) -> pd.Series:
    if col_name is None:
        return pd.Series([pd.NA] * len(df), index=df.index, dtype="object")
    actual = _col(df, col_name)
    if actual is None:
        return pd.Series([pd.NA] * len(df), index=df.index, dtype="object")
    return df[actual]


# --------------------------------------------------------------------------- #
# Raw loaders (vRAW → standardized frame with label_raw)
# --------------------------------------------------------------------------- #
def _standardize(
    *,
    dataset_id: str,
    frames: list[pd.DataFrame],
) -> pd.DataFrame:
    """Concatenate partial frames already holding title/text/label_raw/source_file."""
    if not frames:
        raise ValueError(f"{dataset_id}: no rows loaded")
    df = pd.concat(frames, ignore_index=True)
    df.insert(0, "dataset_id", dataset_id)
    if "uid" not in df.columns:
        df.insert(1, "uid", [f"{dataset_id}_{i}" for i in range(len(df))])
    for col in ("title", "text", "label_raw", "source_file"):
        if col not in df.columns:
            df[col] = pd.NA
    return df


def load_ds1_welfake(root: Path, entry: dict[str, Any]) -> pd.DataFrame:
    """WELFake CSV (Zenodo / Kaggle): title, text, label with 0=fake, 1=real."""
    # Prefer the well-known filename; else first CSV under root.
    path = _find_file(root, "WELFake_Dataset.csv")
    if path is None:
        csvs = sorted(root.rglob("*.csv"))
        if not csvs:
            raise FileNotFoundError(f"DS1: no CSV under {root}")
        path = csvs[0]
    raw = pd.read_csv(path)
    text_cols = entry.get("text_columns") or {}
    title_c = text_cols.get("title", "title")
    body_c = text_cols.get("body", "text")
    label_c = entry.get("label_column", "label")
    frame = pd.DataFrame(
        {
            "title": _series_or_empty(raw, title_c),
            "text": _series_or_empty(raw, body_c),
            "label_raw": _series_or_empty(raw, label_c),
            "source_file": path.name,
        }
    )
    return _standardize(dataset_id="DS1", frames=[frame])


def load_ds2_isot(root: Path, entry: dict[str, Any]) -> pd.DataFrame:
    """ISOT: True.csv → real, Fake.csv → fake (label from filename)."""
    text_cols = entry.get("text_columns") or {}
    title_c = text_cols.get("title", "title")
    body_c = text_cols.get("body", "text")
    frames: list[pd.DataFrame] = []
    for fname, label_raw in (("True.csv", "real"), ("Fake.csv", "fake")):
        path = _find_file(root, fname)
        if path is None:
            # Case-insensitive basename search
            matches = [p for p in root.rglob("*.csv") if p.name.lower() == fname.lower()]
            if not matches:
                raise FileNotFoundError(f"DS2: {fname} not found under {root}")
            path = matches[0]
        raw = pd.read_csv(path)
        frames.append(
            pd.DataFrame(
                {
                    "title": _series_or_empty(raw, title_c),
                    "text": _series_or_empty(raw, body_c),
                    "label_raw": label_raw,
                    "source_file": path.name,
                }
            )
        )
    return _standardize(dataset_id="DS2", frames=frames)


def load_ds3_fakenewsnet(root: Path, entry: dict[str, Any]) -> pd.DataFrame:
    """FakeNewsNet metadata CSVs; label from ``*_fake`` / ``*_real`` filename.

    Body hydration is later (``text_columns.body`` may be TBD). Until then,
    ``text`` is filled from ``title`` so downstream stages have a text field.
    """
    text_cols = entry.get("text_columns") or {}
    title_c = text_cols.get("title", "title")
    body_c = text_cols.get("body")
    if body_c and str(body_c).upper() == "TBD":
        body_c = None
    source_files = entry.get("source_files") or [
        "politifact_fake.csv",
        "politifact_real.csv",
        "gossipcop_fake.csv",
        "gossipcop_real.csv",
    ]
    frames: list[pd.DataFrame] = []
    for rel in source_files:
        path = _find_file(root, rel)
        if path is None:
            raise FileNotFoundError(f"DS3: {rel} not found under {root}")
        stem = path.stem.lower()
        if stem.endswith("_fake") or "_fake" in stem:
            label_raw = "fake"
        elif stem.endswith("_real") or "_real" in stem:
            label_raw = "real"
        else:
            raise ValueError(
                f"DS3: cannot derive label from filename {path.name!r} (expected *_fake / *_real)"
            )
        raw = pd.read_csv(path)
        title = _series_or_empty(raw, title_c)
        body = _series_or_empty(raw, body_c) if body_c else title.copy()
        # Prefer body when present; else fall back to title.
        text = body.where(body.notna() & (body.astype(str).str.strip() != ""), title)
        frames.append(
            pd.DataFrame(
                {
                    "title": title,
                    "text": text,
                    "label_raw": label_raw,
                    "source_file": path.as_posix(),
                }
            )
        )
    return _standardize(dataset_id="DS3", frames=frames)


def load_ds4_covid(root: Path, entry: dict[str, Any]) -> pd.DataFrame:
    """CONSTRAINT COVID-19 Fake News CSVs: string labels ``real`` / ``fake``."""
    text_cols = entry.get("text_columns") or {}
    body_c = text_cols.get("body", "tweet")
    label_c = entry.get("label_column", "label")
    source_files = entry.get("source_files") or [
        "Constraint_Train.csv",
        "Constraint_Val.csv",
        "english_test_with_labels.csv",
    ]
    frames: list[pd.DataFrame] = []
    for rel in source_files:
        path = _find_file(root, rel)
        if path is None:
            raise FileNotFoundError(f"DS4: {rel} not found under {root}")
        raw = pd.read_csv(path)
        frames.append(
            pd.DataFrame(
                {
                    "title": pd.NA,
                    "text": _series_or_empty(raw, body_c),
                    "label_raw": _series_or_empty(raw, label_c),
                    "source_file": path.name,
                }
            )
        )
    return _standardize(dataset_id="DS4", frames=frames)


def load_ds5_liar(root: Path, entry: dict[str, Any]) -> pd.DataFrame:
    """Official LIAR TSVs (no header): column 2 = 6-way label, column 3 = statement."""
    tsv_files = entry.get("tsv_files") or ["train.tsv", "valid.tsv", "test.tsv"]
    has_header = bool(entry.get("tsv_has_header", False))
    text_cols = entry.get("text_columns") or {}
    statement_c = text_cols.get("statement", "statement")
    label_c = entry.get("label_column", "label")
    frames: list[pd.DataFrame] = []
    for name in tsv_files:
        path = _find_file(root, name)
        if path is None:
            raise FileNotFoundError(f"DS5: {name} not found under {root}")
        if path.stat().st_size == 0:
            continue
        try:
            if has_header:
                raw = pd.read_csv(path, sep="\t")
            else:
                raw = pd.read_csv(
                    path,
                    sep="\t",
                    header=None,
                    names=_LIAR_COLUMNS,
                    quoting=3,  # QUOTE_NONE — statements may contain quotes
                    dtype=str,
                )
        except pd.errors.EmptyDataError:
            continue
        if len(raw) == 0:
            continue
        frames.append(
            pd.DataFrame(
                {
                    "title": pd.NA,
                    "text": _series_or_empty(raw, statement_c),
                    "label_raw": _series_or_empty(raw, label_c),
                    "source_file": path.name,
                }
            )
        )
    return _standardize(dataset_id="DS5", frames=frames)


_LOADERS = {
    "DS1": load_ds1_welfake,
    "DS2": load_ds2_isot,
    "DS3": load_ds3_fakenewsnet,
    "DS4": load_ds4_covid,
    "DS5": load_ds5_liar,
}


def load_raw_dataset(
    dataset_id: str,
    entry: dict[str, Any],
    root: str | Path,
) -> pd.DataFrame:
    """Load one dataset from its attached snapshot directory into a raw frame."""
    loader = _LOADERS.get(dataset_id)
    if loader is None:
        raise KeyError(f"No raw loader registered for {dataset_id}")
    return loader(Path(root), entry)


# --------------------------------------------------------------------------- #
# Per-dataset mapping specs (from config; no invented protocol values)
# --------------------------------------------------------------------------- #
def mapping_spec_for_dataset(
    dataset_id: str,
    entry: dict[str, Any],
    label_space: dict[str, int],
) -> tuple[dict[str, int], str, str]:
    """Return ``(token_to_binary, applied_mapping_str, label_source_str)``.

    Mapping rules are read from ``configs/datasets.yaml`` (and the frozen
    protocol values mirrored there). Filename-derived labels for DS2/DS3 use
    the tokens ``real`` / ``fake`` after the loader assigns them.
    """
    if dataset_id == "DS5":
        liar_map = entry.get("liar_binary_map") or {}
        fake_tokens = list(liar_map.get("fake") or [])
        real_tokens = list(liar_map.get("real") or [])
        if not fake_tokens or not real_tokens:
            raise ValueError("DS5.liar_binary_map missing fake/real token lists")
        token_map = build_token_to_binary(
            label_space=label_space, fake_tokens=fake_tokens, real_tokens=real_tokens
        )
        applied = (
            f"LIAR 6-way→binary (frozen §4.3): "
            f"{{{', '.join(fake_tokens)}}}→fake={label_space['fake']}; "
            f"{{{', '.join(real_tokens)}}}→real={label_space['real']}"
        )
        return token_map, applied, "tsv column 'label' (6-way)"

    # Optional explicit source_binary_map: {fake: [...], real: [...]}
    source_map = entry.get("source_binary_map")
    if source_map:
        fake_tokens = list(source_map.get("fake") or [])
        real_tokens = list(source_map.get("real") or [])
        token_map = build_token_to_binary(
            label_space=label_space, fake_tokens=fake_tokens, real_tokens=real_tokens
        )
        applied = (
            f"source_binary_map→global {{real={label_space['real']}, "
            f"fake={label_space['fake']}}}: fake←{fake_tokens}; real←{real_tokens}"
        )
        label_source = str(entry.get("label_column") or "label")
        if dataset_id in {"DS2", "DS3"}:
            label_source = "filename-derived label_raw (real|fake)"
        return token_map, applied, label_source

    # Default for filename-derived real/fake (DS2/DS3) and string labels (DS4).
    token_map = build_token_to_binary(
        label_space=label_space,
        fake_tokens=["fake", "1"] if dataset_id != "DS1" else ["fake"],
        real_tokens=["real", "0"] if dataset_id != "DS1" else ["real"],
    )
    # DS1 without source_binary_map must not silently guess polarity.
    if dataset_id == "DS1":
        raise ValueError(
            "DS1 requires source_binary_map in datasets.yaml "
            "(WELFake source polarity: Zenodo 0=fake, 1=real → remap to global)"
        )
    if dataset_id in {"DS2", "DS3"}:
        applied = f"filename→{{real={label_space['real']}, fake={label_space['fake']}}}"
        return token_map, applied, "filename-derived label_raw (real|fake)"
    applied = f"string labels→{{real={label_space['real']}, fake={label_space['fake']}}}"
    return token_map, applied, str(entry.get("label_column") or "label")


# --------------------------------------------------------------------------- #
# Core binarize
# --------------------------------------------------------------------------- #
def binarize_dataframe(
    raw: pd.DataFrame,
    *,
    dataset_id: str,
    name: str,
    token_to_binary: dict[str, int],
    applied_mapping: str,
    label_source: str,
    label_space: dict[str, int],
) -> BinarizeResult:
    """Apply binary map; drop + count missing/ambiguous rows; return vBIN frame."""
    if "label_raw" not in raw.columns:
        raise KeyError("raw dataframe must contain 'label_raw'")

    n_raw = len(raw)
    label_bin, is_missing, is_ambiguous = apply_binary_map(raw["label_raw"], token_to_binary)
    n_missing = int(is_missing.sum())
    n_ambiguous = int(is_ambiguous.sum())
    keep_mask = ~(is_missing | is_ambiguous)

    kept = raw.loc[keep_mask].copy()
    kept["label"] = label_bin.loc[keep_mask].astype(int)
    # Canonical column order
    for col in _CANONICAL_COLUMNS:
        if col not in kept.columns:
            kept[col] = pd.NA
    kept = kept.loc[:, _CANONICAL_COLUMNS].reset_index(drop=True)
    kept["uid"] = [f"{dataset_id}_{i}" for i in range(len(kept))]

    labels = set(kept["label"].unique().tolist()) if len(kept) else set()
    if labels and labels != {0, 1} and not labels.issubset({0, 1}):
        raise AssertionError(f"{dataset_id}: label set {labels} is not subset of {{0,1}}")

    n_real = int((kept["label"] == int(label_space["real"])).sum()) if len(kept) else 0
    n_fake = int((kept["label"] == int(label_space["fake"])).sum()) if len(kept) else 0
    notes_parts = []
    if n_ambiguous:
        notes_parts.append(f"dropped {n_ambiguous} unrecognized label_raw values")
    if n_missing:
        notes_parts.append(f"dropped {n_missing} missing label_raw values")
    report = MappingReportRow(
        dataset_id=dataset_id,
        name=name,
        n_raw=n_raw,
        n_kept=len(kept),
        n_dropped_ambiguous=n_ambiguous,
        n_dropped_missing_label=n_missing,
        n_real=n_real,
        n_fake=n_fake,
        applied_mapping=applied_mapping,
        label_source=label_source,
        notes="; ".join(notes_parts),
    )
    if report.n_kept + report.n_dropped_ambiguous + report.n_dropped_missing_label != n_raw:
        raise AssertionError(
            f"{dataset_id}: count integrity failed "
            f"({report.n_kept}+{report.n_dropped_ambiguous}+"
            f"{report.n_dropped_missing_label} != {n_raw})"
        )
    return BinarizeResult(dataset_id=dataset_id, dataframe=kept, report=report)


def binarize_dataset(
    dataset_id: str,
    *,
    entry: dict[str, Any],
    label_space: dict[str, int],
    root: str | Path,
    output_dir: str | Path | None = DEFAULT_OUTPUT_DIR,
) -> BinarizeResult:
    """Load → map → optionally write ``{DSx}_vBIN.parquet``."""
    raw = load_raw_dataset(dataset_id, entry, root)
    token_map, applied, label_source = mapping_spec_for_dataset(dataset_id, entry, label_space)
    result = binarize_dataframe(
        raw,
        dataset_id=dataset_id,
        name=str(entry.get("name") or dataset_id),
        token_to_binary=token_map,
        applied_mapping=applied,
        label_source=label_source,
        label_space=label_space,
    )
    if output_dir is not None:
        out = Path(output_dir) / f"{dataset_id}_vBIN.parquet"
        write_parquet(result.dataframe, out)
        result.output_path = out
        logger.info(
            "%s vBIN wrote %s (kept=%d dropped=%d)",
            dataset_id,
            out,
            result.report.n_kept,
            result.report.n_dropped_ambiguous + result.report.n_dropped_missing_label,
        )
    return result


def write_label_mapping_report(
    reports: list[MappingReportRow],
    path: str | Path = DEFAULT_REPORT_PATH,
) -> Path:
    """Write ``results/label_mapping_report.csv``."""
    rows = [asdict(r) for r in reports]
    df = pd.DataFrame(rows)
    return write_csv(df, path, index=False)


def binarize_all(
    *,
    input_root: str | Path = Path("/kaggle/input"),
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    report_path: str | Path = DEFAULT_REPORT_PATH,
    config_dir: str | Path | None = None,
    dataset_ids: list[str] | None = None,
) -> list[BinarizeResult]:
    """Run EXP-P1a for all (or selected) datasets; write vBIN + report CSV."""
    raw_cfg = load_config_raw("datasets", config_dir)
    label_space = dict(raw_cfg.get("label_space") or {})
    if int(label_space.get("real", -1)) != 0 or int(label_space.get("fake", -1)) != 1:
        raise ValueError(
            f"configs/datasets.yaml label_space must be {{real:0, fake:1}}, got {label_space}"
        )

    datasets = raw_cfg.get("datasets") or {}
    ids = dataset_ids or list(datasets.keys())
    input_root = Path(input_root)
    ensure_dir(output_dir)

    results: list[BinarizeResult] = []
    reports: list[MappingReportRow] = []
    for ds_id in ids:
        entry = dict(datasets.get(ds_id) or {})
        if not entry:
            raise KeyError(f"{ds_id} not found in datasets.yaml")
        resolved = resolve_input_path(ds_id, entry, input_root)
        if resolved is None:
            configured = str(entry.get("input_path") or "").strip()
            if configured and configured.upper() != "TBD" and Path(configured).is_dir():
                resolved = Path(configured)
        if resolved is None:
            raise FileNotFoundError(
                f"{ds_id}: attached snapshot not found under {input_root} "
                f"(input_path={entry.get('input_path')!r})"
            )
        # Prefer configured input_path when it exists (Kaggle nested layout).
        configured = str(entry.get("input_path") or "").strip()
        if configured and configured.upper() != "TBD" and Path(configured).is_dir():
            resolved = Path(configured)

        result = binarize_dataset(
            ds_id,
            entry=entry,
            label_space=label_space,
            root=resolved,
            output_dir=output_dir,
        )
        results.append(result)
        reports.append(result.report)

    write_label_mapping_report(reports, report_path)
    logger.info("Wrote label mapping report → %s", report_path)
    return results


# Re-export for tests / interactive use
__all__ = [
    "BinarizeResult",
    "MappingReportRow",
    "apply_binary_map",
    "binarize_all",
    "binarize_dataframe",
    "binarize_dataset",
    "build_token_to_binary",
    "load_raw_dataset",
    "map_label",
    "mapping_spec_for_dataset",
    "normalize_label_token",
    "write_label_mapping_report",
]
