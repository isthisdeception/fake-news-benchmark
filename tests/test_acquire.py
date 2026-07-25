"""Tests for fnb.data.acquire (EXP-P0 snapshot hashing)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from fnb.data.acquire import (
    DatasetSnapshot,
    acquire_snapshots,
    hash_directory,
    list_input_datasets,
    resolve_input_path,
    write_snapshot_hashes,
)


def _make_tree(root: Path, files: dict[str, bytes]) -> Path:
    for rel, content in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    return root


def test_hash_directory_is_stable_and_order_independent(tmp_path: Path):
    a = _make_tree(tmp_path / "a", {"b/x.txt": b"hello", "a.txt": b"world"})
    b = _make_tree(tmp_path / "b", {"a.txt": b"world", "b/x.txt": b"hello"})
    ha, na, ba, _ = hash_directory(a)
    hb, nb, bb, _ = hash_directory(b)
    assert ha == hb
    assert na == nb == 2
    assert ba == bb == len(b"hello") + len(b"world")
    assert len(ha) == 64


def test_hash_directory_changes_when_content_changes(tmp_path: Path):
    d1 = _make_tree(tmp_path / "d1", {"f.txt": b"v1"})
    d2 = _make_tree(tmp_path / "d2", {"f.txt": b"v2"})
    h1, *_ = hash_directory(d1)
    h2, *_ = hash_directory(d2)
    assert h1 != h2


def test_hash_directory_empty_raises(tmp_path: Path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError):
        hash_directory(empty)


def test_resolve_nested_kaggle_layout(tmp_path: Path):
    """Kaggle nested layout: /kaggle/input/datasets/<owner>/<name>/."""
    nested = tmp_path / "datasets" / "saurabhshahane" / "fake-news-classification"
    nested.mkdir(parents=True)
    (nested / "WELFake_Dataset.csv").write_text("x", encoding="utf-8")
    attached = list_input_datasets(tmp_path)
    assert nested in attached
    entry = {
        "kaggle_slug": "saurabhshahane/fake-news-classification",
        "input_dirname": "fake-news-classification",
        "input_path": "TBD",
        "name": "WELFake",
    }
    resolved = resolve_input_path("DS1", entry, tmp_path, attached)
    assert resolved == nested


def test_resolve_input_path_by_dirname(tmp_path: Path):
    attached = tmp_path / "fake-news-classification"
    attached.mkdir()
    (attached / "WELFake_Dataset.csv").write_text("x", encoding="utf-8")
    entry = {
        "kaggle_slug": "saurabhshahane/fake-news-classification",
        "input_dirname": "fake-news-classification",
        "input_path": "TBD",
        "name": "WELFake",
    }
    resolved = resolve_input_path("DS1", entry, tmp_path, [attached])
    assert resolved == attached


def test_acquire_snapshots_writes_ledger_and_manifest(tmp_path: Path):
    # Minimal datasets.yaml pointing at a temp attached tree.
    cfg_dir = tmp_path / "configs"
    cfg_dir.mkdir()
    input_root = tmp_path / "input"
    ds_dir = input_root / "fake-news-classification"
    _make_tree(ds_dir, {"WELFake_Dataset.csv": b"title,text,label\na,b,1\n"})

    datasets_yaml = {
        "label_space": {"real": 0, "fake": 1},
        "article_transfer_set": ["DS1"],
        "domain_shift_probe": [],
        "short_statement_track": [],
        "datasets": {
            "DS1": {
                "name": "WELFake",
                "role": "test",
                "text_type": "Article",
                "in_article_transfer": True,
                "kaggle_slug": "saurabhshahane/fake-news-classification",
                "kaggle_version": "1",
                "input_dirname": "fake-news-classification",
                "input_path": "TBD",
                "label_column": "label",
                "text_columns": {"title": "title", "body": "text"},
                "source_field_available": False,
                "license_note": "test",
            }
        },
        "version_tags": {"vRAW": "raw"},
        "llm_contamination": {"probe": "x", "report_to": "y", "rule": "z"},
    }
    (cfg_dir / "datasets.yaml").write_text(yaml.dump(datasets_yaml), encoding="utf-8")

    hashes_path = tmp_path / "SNAPSHOT_HASHES.txt"
    manifest_path = tmp_path / "manifest.json"
    snaps = acquire_snapshots(
        input_root=input_root,
        hashes_path=hashes_path,
        manifest_path=manifest_path,
        config_dir=cfg_dir,
        dataset_ids=["DS1"],
    )
    assert len(snaps) == 1
    assert snaps[0].status == "ok"
    assert snaps[0].sha256
    text = hashes_path.read_text(encoding="utf-8")
    assert "DS1" in text and snaps[0].sha256 in text
    assert manifest_path.exists()

    # Re-run → identical hash
    snaps2 = acquire_snapshots(
        input_root=input_root,
        hashes_path=hashes_path,
        manifest_path=manifest_path,
        config_dir=cfg_dir,
        dataset_ids=["DS1"],
    )
    assert snaps2[0].sha256 == snaps[0].sha256


def test_ds4_does_not_falsely_match_isot(tmp_path: Path):
    """Regression: generic 'fake'/'news' tokens must not map COVID → ISOT."""
    isot = tmp_path / "datasets" / "clmentbisaillon" / "fake-and-real-news-dataset"
    isot.mkdir(parents=True)
    (isot / "True.csv").write_text("x", encoding="utf-8")
    attached = list_input_datasets(tmp_path)
    entry = {
        "name": "COVID-19 Fake News",
        "kaggle_slug": "TBD",
        "input_dirname": "TBD",
        "input_path": "TBD",
    }
    assert resolve_input_path("DS4", entry, tmp_path, attached) is None
    # DS2 still resolves to ISOT via slug/dirname.
    ds2_entry = {
        "name": "ISOT",
        "kaggle_slug": "clmentbisaillon/fake-and-real-news-dataset",
        "input_dirname": "fake-and-real-news-dataset",
        "input_path": "TBD",
    }
    assert resolve_input_path("DS2", ds2_entry, tmp_path, attached) == isot


def test_write_snapshot_hashes_marks_missing(tmp_path: Path):
    path = tmp_path / "hashes.txt"
    snaps = [
        DatasetSnapshot(
            dataset_id="DS9",
            name="Missing",
            input_path="",
            sha256="",
            n_files=0,
            total_bytes=0,
            kaggle_slug="TBD",
            kaggle_version="TBD",
            license_note="TBD",
            status="missing",
            notes="not attached",
        )
    ]
    write_snapshot_hashes(snaps, path)
    assert "MISSING" in path.read_text(encoding="utf-8")
