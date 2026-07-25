"""Tests for classical TF-IDF trainers (S16)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from fnb.models.classical import (
    CLASSICAL_MODEL_IDS,
    build_tfidf_vectorizer,
    train_classical,
)


def _write_tiny_configs(cfg_dir: Path) -> None:
    """Tiny grids + min_df=1 so synthetic tests stay fast and valid."""
    cfg_dir.mkdir(parents=True, exist_ok=True)
    classical = {
        "tfidf": {
            "ngram_range": [1, 2],
            "max_features": 5000,
            "min_df": 1,
            "fit_on": "train_only",
        },
        "selection": {"metric": "val_macro_f1", "refit": True},
        "class_weight": "balanced",
        "models": {
            "LR": {
                "estimator": "sklearn.linear_model.LogisticRegression",
                "grid": {
                    "C": [0.1, 1.0],
                    "penalty": ["l2"],
                    "solver": ["liblinear"],
                    "class_weight": ["balanced"],
                    "max_iter": [500],
                },
            },
            "SVM": {
                "estimator": "sklearn.svm.LinearSVC",
                "calibration": "platt",
                "grid": {
                    "C": [0.1, 1.0],
                    "class_weight": ["balanced"],
                    "max_iter": [2000],
                },
            },
            "RF": {
                "estimator": "sklearn.ensemble.RandomForestClassifier",
                "grid": {
                    "n_estimators": [20],
                    "max_depth": [5],
                    "min_samples_leaf": [1],
                    "class_weight": ["balanced"],
                },
            },
            "LGBM": {
                "estimator": "lightgbm.LGBMClassifier",
                "grid": {
                    "n_estimators": [20],
                    "num_leaves": [15],
                    "learning_rate": [0.1],
                    "class_weight": ["balanced"],
                },
            },
        },
    }
    preprocessing = {
        "classical_track": {
            "version_tag": "vCLEAN-C",
            "steps": ["lowercase"],
            "tfidf": {
                "ngram_range": [1, 2],
                "max_features": 5000,
                "min_df": 1,
                "fit_on": "train_only",
            },
            "record_versions": [],
        },
        "neural_track": {
            "version_tag": "vCLEAN-N",
            "steps": ["unicode_nfc"],
            "stopword_removal": False,
            "lemmatization": False,
        },
        "full_document_exposure": {
            "method": "sliding_window",
            "window_tokens": 512,
            "stride_tokens": 128,
            "window_logits_pooling": "mean",
            "title_only_max_tokens": 64,
            "granularity_conditions": ["title"],
            "granularity_applicable_datasets": ["DS1"],
        },
        "class_imbalance": {
            "method": "inverse_frequency_class_weighted_cross_entropy",
            "smote": False,
            "test_distribution": "natural",
        },
    }
    (cfg_dir / "classical_grids.yaml").write_text(yaml.dump(classical), encoding="utf-8")
    (cfg_dir / "preprocessing.yaml").write_text(yaml.dump(preprocessing), encoding="utf-8")


def _synthetic_split() -> tuple[list[str], np.ndarray, list[str], np.ndarray, list[str]]:
    # Separable bag-of-words classes; unique leak token appears ONLY in test.
    real_train = [f"real news report government policy update {i}" for i in range(12)]
    fake_train = [f"fake viral hoax conspiracy rumor claim {i}" for i in range(12)]
    texts_train = real_train + fake_train
    y_train = np.array([0] * 12 + [1] * 12)

    texts_val = [
        "real news report government brief",
        "fake viral hoax conspiracy alert",
        "real policy update government",
        "fake rumor claim viral",
        "real news government report",
        "fake hoax conspiracy claim",
    ]
    y_val = np.array([0, 1, 0, 1, 0, 1])

    texts_test = [
        "real news report government",
        "fake viral hoax rumor",
        "zzzzleakonlyintestoken real news report",  # unique token must NOT enter vocab
    ]
    return texts_train, y_train, texts_val, y_val, texts_test


def test_classical_model_ids():
    assert CLASSICAL_MODEL_IDS == ("LR", "SVM", "RF", "LGBM")


def test_tfidf_fit_on_must_be_train_only(tmp_path: Path):
    _write_tiny_configs(tmp_path)
    # Corrupt fit_on
    raw = yaml.safe_load((tmp_path / "classical_grids.yaml").read_text(encoding="utf-8"))
    raw["tfidf"]["fit_on"] = "all"
    (tmp_path / "classical_grids.yaml").write_text(yaml.dump(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="train_only"):
        build_tfidf_vectorizer(config_dir=tmp_path)


@pytest.mark.parametrize("model_id", ["LR", "SVM", "RF", "LGBM"])
def test_each_classical_model_trains_and_predicts_proba(model_id: str, tmp_path: Path):
    if model_id == "LGBM":
        pytest.importorskip("lightgbm")

    cfg_dir = tmp_path / "configs"
    _write_tiny_configs(cfg_dir)
    artifacts = tmp_path / "artifacts"
    texts_train, y_train, texts_val, y_val, texts_test = _synthetic_split()

    result = train_classical(
        model_id,
        texts_train,
        y_train,
        texts_val,
        y_val,
        seed=13,
        texts_test=texts_test,
        artifacts_dir=artifacts,
        config_dir=cfg_dir,
    )

    assert result.best_params is not None
    assert 0.0 <= result.val_macro_f1 <= 1.0
    assert result.y_pred_val is not None and result.y_prob_val is not None
    assert result.y_pred_test is not None and result.y_prob_test is not None
    assert result.y_pred_val.shape == (len(texts_val),)
    assert result.y_prob_val.shape == (len(texts_val),)
    assert np.all((result.y_prob_val >= 0.0) & (result.y_prob_val <= 1.0))
    assert np.all((result.y_prob_test >= 0.0) & (result.y_prob_test <= 1.0))
    # Platt / predict_proba path works for SVM too
    assert hasattr(result.estimator, "predict_proba")
    assert (artifacts / "tfidf_13.pkl").is_file()
    assert result.tfidf_path is not None


def test_tfidf_vocabulary_excludes_test_only_token(tmp_path: Path):
    cfg_dir = tmp_path / "configs"
    _write_tiny_configs(cfg_dir)
    texts_train, y_train, texts_val, y_val, texts_test = _synthetic_split()

    result = train_classical(
        "LR",
        texts_train,
        y_train,
        texts_val,
        y_val,
        seed=42,
        texts_test=texts_test,
        artifacts_dir=tmp_path / "artifacts",
        config_dir=cfg_dir,
    )
    vocab = result.vectorizer.vocabulary_
    assert "zzzzleakonlyintestoken" not in vocab
    # Train tokens should be present
    assert any(tok.startswith("government") or tok == "government" for tok in vocab)
    assert any("hoax" in tok or tok == "hoax" for tok in vocab)


def test_smote_config_true_is_rejected(tmp_path: Path):
    cfg_dir = tmp_path / "configs"
    _write_tiny_configs(cfg_dir)
    raw = yaml.safe_load((cfg_dir / "preprocessing.yaml").read_text(encoding="utf-8"))
    raw["class_imbalance"]["smote"] = True
    (cfg_dir / "preprocessing.yaml").write_text(yaml.dump(raw), encoding="utf-8")
    texts_train, y_train, texts_val, y_val, _ = _synthetic_split()
    with pytest.raises(RuntimeError, match="SMOTE"):
        train_classical(
            "LR",
            texts_train,
            y_train,
            texts_val,
            y_val,
            seed=13,
            artifacts_dir=tmp_path / "artifacts",
            config_dir=cfg_dir,
        )
