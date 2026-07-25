"""Tests for EXP-P1a binarization (and later EXP-P1b preprocessing cases)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

from fnb.data.binarize import (
    apply_binary_map,
    binarize_all,
    binarize_dataframe,
    build_token_to_binary,
    load_raw_dataset,
    map_label,
    mapping_spec_for_dataset,
    normalize_label_token,
)

LABEL_SPACE = {"real": 0, "fake": 1}

LIAR_MAP = {
    "fake": ["pants-fire", "false", "barely-true"],
    "real": ["half-true", "mostly-true", "true"],
}


def _liar_token_map() -> dict[str, int]:
    return build_token_to_binary(
        label_space=LABEL_SPACE,
        fake_tokens=LIAR_MAP["fake"],
        real_tokens=LIAR_MAP["real"],
    )


def test_normalize_label_token_variants():
    assert normalize_label_token("Pants-Fire") == "pants-fire"
    assert normalize_label_token("pants_fire") == "pants-fire"
    assert normalize_label_token(0) == "0"
    assert normalize_label_token(1.0) == "1"
    assert normalize_label_token(None) is None
    assert normalize_label_token("") is None


def test_liar_frozen_mapping_tokens():
    """Protocol §4.3: pants-fire/false/barely-true→fake; half-true/mostly-true/true→real."""
    token_map = _liar_token_map()
    for tok in LIAR_MAP["fake"]:
        assert map_label(tok, token_map) == 1
    for tok in LIAR_MAP["real"]:
        assert map_label(tok, token_map) == 0
    assert map_label("pants_fire", token_map) == 1  # underscore variant
    assert map_label("unknown-label", token_map) is None


def test_liar_binarize_dataframe_counts_and_label_set():
    raw = pd.DataFrame(
        {
            "dataset_id": ["DS5"] * 8,
            "uid": [f"x{i}" for i in range(8)],
            "title": [pd.NA] * 8,
            "text": [f"stmt {i}" for i in range(8)],
            "label_raw": [
                "pants-fire",
                "false",
                "barely-true",
                "half-true",
                "mostly-true",
                "true",
                "waffle",  # ambiguous → drop
                None,  # missing → drop
            ],
            "source_file": ["train.tsv"] * 8,
        }
    )
    token_map = _liar_token_map()
    result = binarize_dataframe(
        raw,
        dataset_id="DS5",
        name="LIAR",
        token_to_binary=token_map,
        applied_mapping="liar-test",
        label_source="label",
        label_space=LABEL_SPACE,
    )
    assert result.report.n_raw == 8
    assert result.report.n_kept == 6
    assert result.report.n_dropped_ambiguous == 1
    assert result.report.n_dropped_missing_label == 1
    assert (
        result.report.n_kept
        + result.report.n_dropped_ambiguous
        + result.report.n_dropped_missing_label
        == result.report.n_raw
    )
    labels = set(result.dataframe["label"].unique().tolist())
    assert labels == {0, 1}
    assert result.report.n_fake == 3
    assert result.report.n_real == 3
    kept_raw = result.dataframe["label_raw"].tolist()
    kept_lab = result.dataframe["label"].tolist()
    for r, lab in zip(kept_raw, kept_lab, strict=True):
        expected = 1 if normalize_label_token(r) in {"pants-fire", "false", "barely-true"} else 0
        assert lab == expected


def test_welfake_polarity_matches_global():
    """Kaggle WELFake: 0=real, 1=fake → identity into global {real=0, fake=1}."""
    entry = {
        "name": "WELFake",
        "source_binary_map": {"fake": [1, "1"], "real": [0, "0"]},
    }
    token_map, applied, _ = mapping_spec_for_dataset("DS1", entry, LABEL_SPACE)
    assert map_label(0, token_map) == 0
    assert map_label(1, token_map) == 1

    raw = pd.DataFrame(
        {
            "dataset_id": ["DS1", "DS1"],
            "uid": ["a", "b"],
            "title": ["t0", "t1"],
            "text": ["body0", "body1"],
            "label_raw": [0, 1],
            "source_file": ["WELFake_Dataset.csv"] * 2,
        }
    )
    result = binarize_dataframe(
        raw,
        dataset_id="DS1",
        name="WELFake",
        token_to_binary=token_map,
        applied_mapping=applied,
        label_source="label",
        label_space=LABEL_SPACE,
    )
    assert result.dataframe["label"].tolist() == [0, 1]


def test_apply_binary_map_flags_missing_and_ambiguous():
    token_map = build_token_to_binary(
        label_space=LABEL_SPACE, fake_tokens=["fake"], real_tokens=["real"]
    )
    series = pd.Series(["real", "fake", "maybe", None])
    label_bin, is_missing, is_ambiguous = apply_binary_map(series, token_map)
    assert label_bin.tolist()[0] == 0
    assert label_bin.tolist()[1] == 1
    assert bool(is_missing.iloc[3]) is True
    assert bool(is_ambiguous.iloc[2]) is True
    assert int(is_missing.sum()) == 1
    assert int(is_ambiguous.sum()) == 1


def _parquet_usable(tmp_path: Path) -> bool:
    try:
        probe = tmp_path / "_probe.parquet"
        pd.DataFrame({"x": [1]}).to_parquet(probe)
        return True
    except Exception:
        return False


def test_binarize_all_writes_vbin_and_report(tmp_path: Path):
    """End-to-end on tiny synthetic snapshots for DS1 + DS5."""
    if not _parquet_usable(tmp_path):
        pytest.skip("parquet engine unavailable in this environment (Kaggle has pyarrow)")

    input_root = tmp_path / "input"
    ds1 = input_root / "datasets" / "saurabhshahane" / "fake-news-classification"
    ds1.mkdir(parents=True)
    pd.DataFrame(
        {
            "title": ["A", "B", "C"],
            "text": ["a", "b", "c"],
            "label": [0, 1, 9],  # 9 → ambiguous drop
        }
    ).to_csv(ds1 / "WELFake_Dataset.csv", index=False)

    ds5 = input_root / "datasets" / "isthisdeception" / "liar-dataset"
    ds5.mkdir(parents=True)

    def liar_row(i: int, lab: str, stmt: str) -> str:
        fillers = "\t".join(["x"] * 11)
        return f"id{i}\t{lab}\t{stmt}\t{fillers}"

    rows = [
        liar_row(0, "pants-fire", "s0"),
        liar_row(1, "true", "s1"),
        liar_row(2, "half-true", "s2"),
        liar_row(3, "not-a-label", "s3"),
    ]
    (ds5 / "train.tsv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    (ds5 / "valid.tsv").write_text("", encoding="utf-8")
    (ds5 / "test.tsv").write_text("", encoding="utf-8")

    cfg_dir = tmp_path / "configs"
    cfg_dir.mkdir()
    datasets_yaml = {
        "label_space": {"real": 0, "fake": 1},
        "article_transfer_set": ["DS1"],
        "domain_shift_probe": [],
        "short_statement_track": ["DS5"],
        "datasets": {
            "DS1": {
                "name": "WELFake",
                "role": "test",
                "text_type": "Article",
                "in_article_transfer": True,
                "kaggle_slug": "saurabhshahane/fake-news-classification",
                "kaggle_version": "1",
                "input_dirname": "fake-news-classification",
                "input_path": str(ds1),
                "label_column": "label",
                "text_columns": {"title": "title", "body": "text"},
                "source_binary_map": {"fake": [1, "1"], "real": [0, "0"]},
                "source_field_available": False,
                "license_note": "test",
            },
            "DS5": {
                "name": "LIAR",
                "role": "test",
                "text_type": "Short statement",
                "in_article_transfer": False,
                "kaggle_slug": "isthisdeception/liar-dataset",
                "kaggle_version": "1",
                "input_dirname": "liar-dataset",
                "input_path": str(ds5),
                "label_column": "label",
                "text_columns": {"statement": "statement"},
                "tsv_files": ["train.tsv", "valid.tsv", "test.tsv"],
                "tsv_has_header": False,
                "liar_binary_map": LIAR_MAP,
                "license_note": "test",
            },
        },
        "version_tags": {"vBIN": "bin"},
        "llm_contamination": {"probe": "x", "report_to": "y", "rule": "z"},
    }
    (cfg_dir / "datasets.yaml").write_text(yaml.dump(datasets_yaml), encoding="utf-8")

    out_dir = tmp_path / "processed"
    report_path = tmp_path / "label_mapping_report.csv"
    results = binarize_all(
        input_root=input_root,
        output_dir=out_dir,
        report_path=report_path,
        config_dir=cfg_dir,
        dataset_ids=["DS1", "DS5"],
    )
    assert len(results) == 2
    assert (out_dir / "DS1_vBIN.parquet").exists()
    assert (out_dir / "DS5_vBIN.parquet").exists()
    assert report_path.exists()

    report = pd.read_csv(report_path)
    assert set(report["dataset_id"]) == {"DS1", "DS5"}
    ds1_row = report.loc[report["dataset_id"] == "DS1"].iloc[0]
    assert ds1_row["n_raw"] == 3
    assert ds1_row["n_kept"] == 2
    assert ds1_row["n_dropped_ambiguous"] == 1

    ds5_row = report.loc[report["dataset_id"] == "DS5"].iloc[0]
    assert ds5_row["n_kept"] == 3
    assert ds5_row["n_dropped_ambiguous"] == 1

    ds5_df = pd.read_parquet(out_dir / "DS5_vBIN.parquet")
    assert set(ds5_df["label"].unique().tolist()) == {0, 1}


def test_binarize_all_report_without_parquet(tmp_path: Path):
    """Report + counts work even when parquet write is skipped (output_dir=None)."""
    input_root = tmp_path / "input"
    ds1 = input_root / "fake-news-classification"
    ds1.mkdir(parents=True)
    pd.DataFrame({"title": ["A", "B"], "text": ["a", "b"], "label": [0, 1]}).to_csv(
        ds1 / "WELFake_Dataset.csv", index=False
    )

    cfg_dir = tmp_path / "configs"
    cfg_dir.mkdir()
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
                "input_path": str(ds1),
                "label_column": "label",
                "text_columns": {"title": "title", "body": "text"},
                "source_binary_map": {"fake": [1, "1"], "real": [0, "0"]},
                "source_field_available": False,
                "license_note": "test",
            }
        },
        "version_tags": {"vBIN": "bin"},
        "llm_contamination": {"probe": "x", "report_to": "y", "rule": "z"},
    }
    (cfg_dir / "datasets.yaml").write_text(yaml.dump(datasets_yaml), encoding="utf-8")

    from fnb.config import load_config_raw
    from fnb.data.binarize import binarize_dataset, write_label_mapping_report

    raw_cfg = load_config_raw("datasets", cfg_dir)
    entry = raw_cfg["datasets"]["DS1"]
    result = binarize_dataset(
        "DS1",
        entry=entry,
        label_space=raw_cfg["label_space"],
        root=ds1,
        output_dir=None,
    )
    assert result.report.n_kept == 2
    assert set(result.dataframe["label"].tolist()) == {0, 1}
    report_path = tmp_path / "label_mapping_report.csv"
    write_label_mapping_report([result.report], report_path)
    assert report_path.exists()


def test_load_ds2_isot_from_filenames(tmp_path: Path):
    root = tmp_path / "isot"
    root.mkdir()
    pd.DataFrame({"title": ["t"], "text": ["real body"]}).to_csv(root / "True.csv", index=False)
    pd.DataFrame({"title": ["f"], "text": ["fake body"]}).to_csv(root / "Fake.csv", index=False)
    entry = {
        "text_columns": {"title": "title", "body": "text"},
        "source_binary_map": {"fake": ["fake"], "real": ["real"]},
    }
    raw = load_raw_dataset("DS2", entry, root)
    assert set(raw["label_raw"].unique()) == {"real", "fake"}
    token_map, _, _ = mapping_spec_for_dataset("DS2", entry, LABEL_SPACE)
    result = binarize_dataframe(
        raw,
        dataset_id="DS2",
        name="ISOT",
        token_to_binary=token_map,
        applied_mapping="isot",
        label_source="filename",
        label_space=LABEL_SPACE,
    )
    assert set(result.dataframe["label"].tolist()) == {0, 1}


def test_global_label_space_enforced():
    with pytest.raises(ValueError):
        build_token_to_binary(
            label_space={"real": 1, "fake": 0},
            fake_tokens=["fake"],
            real_tokens=["real"],
        )
