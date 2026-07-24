"""Small, dependency-light IO helpers: parquet, CSV, JSON, and split-index files.

Centralizes read/write so paths, encodings, and formats are consistent across
the project. Split indices are stored as newline-separated integers in ``.idx``
files (git-friendly, indices only — never the data itself; §4.4).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable, Sequence


def ensure_dir(path: str | Path) -> Path:
    """Create a directory (and parents) if needed; return it as a Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


# --- JSON --------------------------------------------------------------------
def write_json(obj: Any, path: str | Path, *, indent: int = 2) -> Path:
    p = Path(path)
    ensure_dir(p.parent)
    p.write_text(json.dumps(obj, indent=indent, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    return p


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


# --- Parquet (pandas/pyarrow) ------------------------------------------------
def write_parquet(df: Any, path: str | Path, **kwargs: Any) -> Path:
    """Write a pandas DataFrame to parquet."""
    p = Path(path)
    ensure_dir(p.parent)
    df.to_parquet(p, **kwargs)
    return p


def read_parquet(path: str | Path, **kwargs: Any) -> Any:
    """Read a parquet file into a pandas DataFrame."""
    import pandas as pd

    return pd.read_parquet(path, **kwargs)


# --- CSV ---------------------------------------------------------------------
def write_csv(df: Any, path: str | Path, *, index: bool = False, **kwargs: Any) -> Path:
    """Write a pandas DataFrame to CSV."""
    p = Path(path)
    ensure_dir(p.parent)
    df.to_csv(p, index=index, **kwargs)
    return p


def read_csv(path: str | Path, **kwargs: Any) -> Any:
    """Read a CSV file into a pandas DataFrame."""
    import pandas as pd

    return pd.read_csv(path, **kwargs)


def append_csv_row(path: str | Path, row: dict[str, Any], *, fieldnames: Sequence[str] | None = None) -> Path:
    """Append a single dict row to a CSV, writing a header if the file is new.

    Append-only registry writer (reproducibility §9, §10). ``fieldnames``
    defaults to the row's keys when the file does not yet exist.
    """
    p = Path(path)
    ensure_dir(p.parent)
    exists = p.exists()
    names = list(fieldnames) if fieldnames is not None else list(row.keys())
    with p.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=names)
        if not exists:
            writer.writeheader()
        writer.writerow(row)
    return p


# --- Split indices (.idx) ----------------------------------------------------
def save_indices(indices: Iterable[int], path: str | Path) -> Path:
    """Save integer split indices as a newline-separated ``.idx`` file."""
    p = Path(path)
    ensure_dir(p.parent)
    with p.open("w", encoding="utf-8") as fh:
        for i in indices:
            fh.write(f"{int(i)}\n")
    return p


def load_indices(path: str | Path) -> list[int]:
    """Load integer split indices from a ``.idx`` file."""
    text = Path(path).read_text(encoding="utf-8")
    return [int(line) for line in text.splitlines() if line.strip()]
