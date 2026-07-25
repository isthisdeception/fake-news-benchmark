"""EXP-P0: register attached Kaggle input datasets and compute snapshot hashes.

The pinned Kaggle dataset *version* IS the frozen snapshot (protocol §4.6).
Raw bytes stay under ``/kaggle/input/`` (read-only) and are never copied into
the git repo. This module:

1. Resolves each DS1–DS5 entry in ``configs/datasets.yaml`` to an attached
   directory under ``/kaggle/input/``.
2. Computes a deterministic content SHA-256 over every file in that directory
   (sorted by relative path).
3. Writes ``data/SNAPSHOT_HASHES.txt`` and ``results/kaggle_dataset_manifest.json``.

FakeNewsNet is used only from its attached hydrated snapshot — never re-crawled.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from fnb.config import load_config, load_config_raw
from fnb.utils.hashing import sha256_file, sha256_str
from fnb.utils.io import ensure_dir, write_json

logger = logging.getLogger("fnb.data.acquire")

DEFAULT_INPUT_ROOT = Path("/kaggle/input")
DEFAULT_HASHES_PATH = Path("data/SNAPSHOT_HASHES.txt")
DEFAULT_MANIFEST_PATH = Path("results/kaggle_dataset_manifest.json")

# Candidate mount-folder name fragments used when input_path is still TBD.
_NAME_HINTS: dict[str, list[str]] = {
    "DS1": ["welfake", "fake-news-classification", "welfake-dataset"],
    "DS2": ["isot", "fake-and-real-news", "fake-news-detection-isot"],
    "DS3": ["fakenewsnet", "fake-news-net", "politifact", "gossipcop"],
    "DS4": ["covid19-fake", "covid-19-fake", "covid-fake-news", "constraint"],
    "DS5": ["liar", "liar-dataset", "liar-fake-news"],
}


@dataclass
class DatasetSnapshot:
    """Resolved snapshot provenance for one dataset id."""

    dataset_id: str
    name: str
    input_path: str
    sha256: str
    n_files: int
    total_bytes: int
    kaggle_slug: str
    kaggle_version: str
    license_note: str
    recovery_fraction: str | float | None = None
    files: list[dict[str, Any]] = field(default_factory=list)
    status: str = "ok"
    notes: str = ""


def list_input_datasets(input_root: str | Path = DEFAULT_INPUT_ROOT) -> list[Path]:
    """Return first-level directories under ``/kaggle/input`` (attached datasets)."""
    root = Path(input_root)
    if not root.is_dir():
        return []
    return sorted(p for p in root.iterdir() if p.is_dir())


def _iter_files(directory: Path) -> Iterable[Path]:
    for path in sorted(directory.rglob("*")):
        if path.is_file():
            # Skip hidden / notebook checkpoint noise.
            if any(part.startswith(".") for part in path.parts):
                continue
            yield path


def hash_directory(directory: str | Path) -> tuple[str, int, int, list[dict[str, Any]]]:
    """Deterministic content hash of every file under ``directory``.

    Returns ``(digest, n_files, total_bytes, per_file_records)``.
    The digest is SHA-256 of the newline-joined ``relative_path\\tfile_sha256``
    lines (sorted), so the same tree always yields the same digest.
    """
    root = Path(directory)
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {root}")

    lines: list[str] = []
    records: list[dict[str, Any]] = []
    total_bytes = 0
    for path in _iter_files(root):
        rel = path.relative_to(root).as_posix()
        digest = sha256_file(path)
        size = path.stat().st_size
        total_bytes += size
        lines.append(f"{rel}\t{digest}")
        records.append({"path": rel, "sha256": digest, "bytes": size})

    if not lines:
        raise ValueError(f"No files found under {root}")

    tree_digest = sha256_str("\n".join(lines) + "\n")
    return tree_digest, len(records), total_bytes, records


def _slug_dirname(slug: str) -> str | None:
    """Kaggle mounts ``owner/dataset-name`` at ``/kaggle/input/dataset-name``."""
    slug = (slug or "").strip()
    if not slug or slug.upper() == "TBD":
        return None
    return slug.split("/")[-1]


def resolve_input_path(
    dataset_id: str,
    entry: dict[str, Any],
    input_root: Path,
    attached: list[Path] | None = None,
) -> Path | None:
    """Resolve a dataset entry to an attached directory, or return None."""
    configured = str(entry.get("input_path") or "").strip()
    if configured and configured.upper() != "TBD":
        path = Path(configured)
        if path.is_dir():
            return path

    attached = attached if attached is not None else list_input_datasets(input_root)
    by_name = {p.name.lower(): p for p in attached}

    dirname = _slug_dirname(str(entry.get("kaggle_slug") or ""))
    if dirname and dirname.lower() in by_name:
        return by_name[dirname.lower()]

    # Explicit override field (optional): input_dirname
    explicit = str(entry.get("input_dirname") or "").strip()
    if explicit and explicit.lower() in by_name:
        return by_name[explicit.lower()]

    # Fuzzy match on hints + dataset name tokens.
    hints = list(_NAME_HINTS.get(dataset_id, []))
    name = str(entry.get("name") or "").lower()
    hints.extend(tok for tok in name.replace("(", " ").replace(")", " ").replace("+", " ").split() if len(tok) > 3)

    for attached_dir in attached:
        lowered = attached_dir.name.lower()
        if any(h.lower() in lowered for h in hints):
            return attached_dir
    return None


def _entry_or_empty(datasets_cfg: Any, dataset_id: str) -> dict[str, Any]:
    if hasattr(datasets_cfg, "datasets"):
        raw = datasets_cfg.datasets.get(dataset_id, {})
    else:
        raw = (datasets_cfg.get("datasets") or {}).get(dataset_id, {})
    return dict(raw) if isinstance(raw, dict) else {}


def acquire_snapshots(
    *,
    input_root: str | Path = DEFAULT_INPUT_ROOT,
    hashes_path: str | Path = DEFAULT_HASHES_PATH,
    manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
    config_dir: str | Path | None = None,
    dataset_ids: list[str] | None = None,
) -> list[DatasetSnapshot]:
    """Hash all (or selected) registered datasets and write ledger + manifest.

    Returns one :class:`DatasetSnapshot` per requested dataset id. Missing
    attachments are recorded with ``status="missing"`` rather than aborting the
    whole run, so partial progress is visible on Kaggle.
    """
    input_root = Path(input_root)
    hashes_path = Path(hashes_path)
    manifest_path = Path(manifest_path)

    # Prefer raw dict so TBD string fields survive without schema friction.
    raw_cfg = load_config_raw("datasets", config_dir)
    validated = load_config("datasets", config_dir)  # still validate structure
    _ = validated

    all_ids = list((raw_cfg.get("datasets") or {}).keys())
    ids = dataset_ids or all_ids
    attached = list_input_datasets(input_root)

    snapshots: list[DatasetSnapshot] = []
    for ds_id in ids:
        entry = _entry_or_empty(raw_cfg, ds_id)
        name = str(entry.get("name") or ds_id)
        resolved = resolve_input_path(ds_id, entry, input_root, attached)
        if resolved is None:
            snap = DatasetSnapshot(
                dataset_id=ds_id,
                name=name,
                input_path="",
                sha256="",
                n_files=0,
                total_bytes=0,
                kaggle_slug=str(entry.get("kaggle_slug") or "TBD"),
                kaggle_version=str(entry.get("kaggle_version") or "TBD"),
                license_note=str(entry.get("license_note") or "TBD"),
                recovery_fraction=entry.get("recovery_fraction"),
                status="missing",
                notes=f"No attached directory found under {input_root} for {ds_id}",
            )
            snapshots.append(snap)
            logger.warning("%s: %s", ds_id, snap.notes)
            continue

        digest, n_files, total_bytes, files = hash_directory(resolved)
        snap = DatasetSnapshot(
            dataset_id=ds_id,
            name=name,
            input_path=str(resolved),
            sha256=digest,
            n_files=n_files,
            total_bytes=total_bytes,
            kaggle_slug=str(entry.get("kaggle_slug") or "TBD"),
            kaggle_version=str(entry.get("kaggle_version") or "TBD"),
            license_note=str(entry.get("license_note") or "TBD"),
            recovery_fraction=entry.get("recovery_fraction"),
            files=files,
            status="ok",
            notes="",
        )
        snapshots.append(snap)
        logger.info(
            "%s hashed path=%s sha256=%s files=%d bytes=%d",
            ds_id, resolved, digest[:16], n_files, total_bytes,
        )

    write_snapshot_hashes(snapshots, hashes_path)
    write_manifest(snapshots, manifest_path, input_root=input_root, attached=attached)
    return snapshots


def write_snapshot_hashes(snapshots: list[DatasetSnapshot], path: str | Path) -> Path:
    """Write the frozen content-hash ledger (protocol §4.6)."""
    path = Path(path)
    ensure_dir(path.parent)
    lines = [
        "# SNAPSHOT_HASHES.txt — frozen dataset content hashes (protocol §4.6)",
        f"# Generated (UTC): {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        "# Format: DATASET_ID  SHA256  N_FILES  TOTAL_BYTES  INPUT_PATH  SLUG@VERSION",
        "# Raw bytes remain under /kaggle/input/; only this ledger is committed to git.",
        "",
    ]
    for s in snapshots:
        if s.status != "ok":
            lines.append(
                f"# {s.dataset_id}  MISSING  — {s.notes}"
            )
            continue
        slug_ver = f"{s.kaggle_slug}@{s.kaggle_version}"
        lines.append(
            f"{s.dataset_id}  {s.sha256}  {s.n_files}  {s.total_bytes}  {s.input_path}  {slug_ver}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_manifest(
    snapshots: list[DatasetSnapshot],
    path: str | Path,
    *,
    input_root: Path,
    attached: list[Path],
) -> Path:
    """Write a JSON manifest of resolved Kaggle attachments + hashes."""
    payload = {
        "protocol_version": "v1.0",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "input_root": str(input_root),
        "attached_datasets": [p.name for p in attached],
        "snapshots": [asdict(s) for s in snapshots],
        "ok_count": sum(1 for s in snapshots if s.status == "ok"),
        "missing_count": sum(1 for s in snapshots if s.status != "ok"),
    }
    return write_json(payload, path)


def discover_report(input_root: str | Path = DEFAULT_INPUT_ROOT) -> str:
    """Human-readable listing of what is currently attached (for Kaggle cells)."""
    root = Path(input_root)
    lines = [f"Attached under {root}:"]
    dirs = list_input_datasets(root)
    if not dirs:
        lines.append("  (none — attach DS1–DS5 via Add Data, or not running on Kaggle)")
        return "\n".join(lines)
    for d in dirs:
        n_files = sum(1 for _ in _iter_files(d))
        lines.append(f"  - {d.name}/  ({n_files} files)")
    return "\n".join(lines)
