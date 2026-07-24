"""Deterministic SHA-256 hashing of files, bytes, strings, JSON, and DataFrames.

Used across the project to stamp provenance (reproducibility §8, §9): config
hashes, dataset snapshot hashes (§4.6), and checkpoint integrity hashes (§7).
All helpers are stable: the same logical input always yields the same digest.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

_CHUNK = 1 << 20  # 1 MiB streaming chunk for large files


def sha256_bytes(data: bytes) -> str:
    """SHA-256 hex digest of raw bytes."""
    return hashlib.sha256(data).hexdigest()


def sha256_str(text: str, *, encoding: str = "utf-8") -> str:
    """SHA-256 hex digest of a string."""
    return sha256_bytes(text.encode(encoding))


def sha256_file(path: str | Path) -> str:
    """SHA-256 hex digest of a file's contents (streamed, memory-safe)."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"File not found: {p}")
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_json(obj: Any) -> str:
    """SHA-256 hex digest of a JSON-serializable object (canonical form).

    Keys are sorted and separators normalized so logically-equal objects hash
    identically regardless of key order or incidental whitespace.
    """
    canonical = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return sha256_str(canonical)


def sha256_dataframe(df: Any) -> str:
    """SHA-256 hex digest of a pandas DataFrame (order-sensitive, content-stable).

    Hashes column names, dtypes, and per-row hashes so the digest is stable
    across processes for identical data (independent of the object's memory
    layout). Requires pandas.
    """
    import pandas as pd

    row_hashes = pd.util.hash_pandas_object(df, index=True).values.tobytes()
    schema = json.dumps(
        {"columns": list(map(str, df.columns)), "dtypes": [str(t) for t in df.dtypes]},
        sort_keys=True,
    ).encode("utf-8")
    h = hashlib.sha256()
    h.update(schema)
    h.update(row_hashes)
    return h.hexdigest()
