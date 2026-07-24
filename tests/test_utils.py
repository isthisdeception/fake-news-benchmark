"""Tests for fnb.utils: hashing, io, logging, run registry."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from fnb.utils import hashing
from fnb.utils.io import (
    append_csv_row,
    load_indices,
    read_json,
    save_indices,
    write_json,
)
from fnb.utils.logging_utils import setup_run_logging
from fnb.utils.run_registry import (
    RUNS_CSV_FIELDS,
    collect_environment,
    finish_run,
    make_experiment_id,
    make_run_id,
    start_run,
)


# --- hashing -----------------------------------------------------------------
def test_sha256_bytes_str_stable():
    assert hashing.sha256_bytes(b"abc") == hashing.sha256_bytes(b"abc")
    assert hashing.sha256_str("abc") == hashing.sha256_bytes(b"abc")
    assert len(hashing.sha256_str("abc")) == 64


def test_sha256_json_order_independent():
    a = hashing.sha256_json({"x": 1, "y": 2})
    b = hashing.sha256_json({"y": 2, "x": 1})
    assert a == b


def test_sha256_file_matches_bytes(tmp_path: Path):
    p = tmp_path / "f.bin"
    p.write_bytes(b"hello world")
    assert hashing.sha256_file(p) == hashing.sha256_bytes(b"hello world")


def test_sha256_file_missing_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        hashing.sha256_file(tmp_path / "nope.bin")


# --- io ----------------------------------------------------------------------
def test_json_roundtrip(tmp_path: Path):
    obj = {"a": 1, "b": [1, 2, 3]}
    p = write_json(obj, tmp_path / "d" / "x.json")
    assert read_json(p) == obj


def test_indices_roundtrip(tmp_path: Path):
    idx = [5, 3, 9, 0, 42]
    p = save_indices(idx, tmp_path / "s.idx")
    assert load_indices(p) == idx


def test_append_csv_row_writes_header_once(tmp_path: Path):
    p = tmp_path / "runs.csv"
    append_csv_row(p, {"a": 1, "b": 2}, fieldnames=["a", "b"])
    append_csv_row(p, {"a": 3, "b": 4}, fieldnames=["a", "b"])
    with p.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    assert rows[0] == ["a", "b"]  # single header
    assert rows[1] == ["1", "2"]
    assert rows[2] == ["3", "4"]
    assert len(rows) == 3


# --- logging -----------------------------------------------------------------
def test_setup_run_logging_creates_file_and_no_duplicate_handlers(tmp_path: Path):
    logger = setup_run_logging("run_abc", tmp_path, logger_name="fnb.test.log")
    n1 = len(logger.handlers)
    logger.info("hello")
    logger2 = setup_run_logging("run_abc", tmp_path, logger_name="fnb.test.log")
    assert logger2 is logger
    assert len(logger2.handlers) == n1  # idempotent
    assert (tmp_path / "run_abc.log").exists()
    for h in logger.handlers:
        h.flush()
    assert "hello" in (tmp_path / "run_abc.log").read_text(encoding="utf-8")


# --- run registry ------------------------------------------------------------
def test_run_id_and_experiment_id_format():
    rid = make_run_id("EXP-1", "BERT", "vBIN", "S-RAND", 13, timestamp="20260725T000000Z")
    assert rid == "EXP-1_BERT_vBIN_S-RAND_seed13__20260725T000000Z"
    eid = make_experiment_id("EXP-1", "BERT", "vBIN", "S-RAND")
    assert eid == "EXP-1_BERT_vBIN_S-RAND"


def test_collect_environment_has_keys():
    env = collect_environment()
    for key in ["python", "platform", "torch", "numpy", "cuda", "gpu"]:
        assert key in env


def test_start_and_finish_run_writes_registry_and_metadata(tmp_path: Path):
    ctx = start_run(
        "EXP-1", "BERT", "vBIN", "S-RAND", 42,
        base_dir=tmp_path,
        extra={"note": "smoke"},
    )
    artifact_dir = tmp_path / "artifacts" / ctx.run_id
    row = finish_run(ctx, status="completed", metrics={"macro_f1": 0.9}, artifact_dir=artifact_dir)

    # runs.csv row
    runs_csv = tmp_path / "results" / "runs.csv"
    assert runs_csv.exists()
    with runs_csv.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 1
    assert rows[0]["run_id"] == ctx.run_id
    assert rows[0]["seed"] == "42"
    assert rows[0]["protocol_version"] == "v1.0"
    assert set(RUNS_CSV_FIELDS).issubset(rows[0].keys())

    # log file
    assert (tmp_path / "logs" / f"{ctx.run_id}.log").exists()

    # metadata.json
    meta_path = artifact_dir / "metadata.json"
    assert meta_path.exists()
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["metrics"]["macro_f1"] == 0.9
    assert meta["seed"] == 42
    assert meta["extra"]["note"] == "smoke"
    assert row["status"] == "completed"
