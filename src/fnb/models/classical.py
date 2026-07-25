"""Classical TF-IDF trainers: LR / SVM / RF / LGBM (protocol §5–§7).

Pipeline:

1. Build TF-IDF from ``configs/classical_grids.yaml`` (unigrams+bigrams,
   ``max_features=20000``, ``min_df=5``).
2. **Fit the vectorizer on TRAIN texts only** (P003 anti-leakage).
3. Expand the per-model grid; select by **validation macro-F1**.
4. Refit the winner on TRAIN (``selection.refit: true``).
5. Emit ``(y_pred, y_prob)`` for val/test; SVM probabilities via Platt scaling
   (``CalibratedClassifierCV``, method=``sigmoid``).
6. Serialize the fitted vectorizer to ``artifacts/tfidf_{seed}.pkl``.

SMOTE is prohibited. Test sets keep the natural class distribution.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.svm import LinearSVC

from fnb.config import load_config, load_config_raw

logger = logging.getLogger("fnb.models.classical")

CLASSICAL_MODEL_IDS: tuple[str, ...] = ("LR", "SVM", "RF", "LGBM")
DEFAULT_ARTIFACTS_DIR = Path("artifacts")


@dataclass
class ClassicalTrainResult:
    """Fitted classical model + vectorizer and held-out predictions."""

    model_id: str
    seed: int
    vectorizer: TfidfVectorizer
    estimator: Any
    best_params: dict[str, Any]
    val_macro_f1: float
    grid_scores: list[dict[str, Any]] = field(default_factory=list)
    y_pred_val: np.ndarray | None = None
    y_prob_val: np.ndarray | None = None
    y_pred_test: np.ndarray | None = None
    y_prob_test: np.ndarray | None = None
    tfidf_path: Path | None = None
    model_path: Path | None = None


def combine_texts(
    texts: Any = None,
    *,
    titles: Any = None,
    bodies: Any = None,
) -> list[str]:
    """Build title+body strings for TF-IDF (classical track).

    Prefer ``titles``/``bodies`` when both are provided; otherwise treat
    ``texts`` as the already-combined document string.
    """
    if titles is not None or bodies is not None:
        n = len(titles) if titles is not None else len(bodies)
        t = (
            pd.Series(titles).fillna("").astype(str)
            if titles is not None
            else pd.Series([""] * n)
        )
        b = (
            pd.Series(bodies).fillna("").astype(str)
            if bodies is not None
            else pd.Series([""] * n)
        )
        return (t + " " + b).str.strip().tolist()
    if texts is None:
        raise ValueError("provide texts= or titles=/bodies=")
    return pd.Series(texts).fillna("").astype(str).tolist()


def build_tfidf_vectorizer(
    *,
    config_dir: str | Path | None = None,
    ngram_range: tuple[int, int] | None = None,
    max_features: int | None = None,
    min_df: int | None = None,
) -> TfidfVectorizer:
    """Construct an unfitted TF-IDF vectorizer from ``classical_grids.yaml``."""
    cfg = load_config("classical_grids", config_dir)
    tf = cfg.tfidf
    if str(tf.fit_on) != "train_only":
        raise ValueError(
            f"classical TF-IDF must be fit_on='train_only' (got {tf.fit_on!r})"
        )
    ng = tuple(ngram_range) if ngram_range is not None else tuple(tf.ngram_range)
    return TfidfVectorizer(
        ngram_range=(int(ng[0]), int(ng[1])),
        max_features=int(tf.max_features if max_features is None else max_features),
        min_df=int(tf.min_df if min_df is None else min_df),
        dtype=np.float32,
    )


def save_tfidf(vectorizer: TfidfVectorizer, path: str | Path) -> Path:
    """Serialize a fitted TF-IDF vectorizer (``artifacts/tfidf_{seed}.pkl``)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, p)
    return p


def load_tfidf(path: str | Path) -> TfidfVectorizer:
    return joblib.load(Path(path))


def _expand_grid(grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    if not grid:
        return [{}]
    keys = list(grid.keys())
    values = [list(grid[k]) for k in keys]
    return [dict(zip(keys, combo, strict=True)) for combo in product(*values)]


def _assert_smote_prohibited(config_dir: str | Path | None) -> None:
    raw = load_config_raw("preprocessing", config_dir)
    smote = (raw.get("class_imbalance") or {}).get("smote", False)
    if smote:
        raise RuntimeError("SMOTE is PROHIBITED (protocol §7); configs disagree")


def _make_base_estimator(model_id: str, params: dict[str, Any], *, seed: int) -> Any:
    p = dict(params)
    if model_id == "LR":
        return LogisticRegression(random_state=seed, **p)
    if model_id == "SVM":
        # Probabilities added via CalibratedClassifierCV (Platt).
        return LinearSVC(random_state=seed, dual="auto", **p)
    if model_id == "RF":
        return RandomForestClassifier(random_state=seed, n_jobs=-1, **p)
    if model_id == "LGBM":
        try:
            from lightgbm import LGBMClassifier
        except ImportError as exc:  # pragma: no cover
            raise ImportError("lightgbm is required for model_id='LGBM'") from exc
        return LGBMClassifier(random_state=seed, verbosity=-1, **p)
    raise ValueError(f"unknown classical model_id: {model_id!r}; expected {CLASSICAL_MODEL_IDS}")


def _wrap_estimator(model_id: str, estimator: Any, *, seed: int, model_cfg: dict[str, Any]) -> Any:
    if model_id != "SVM":
        return estimator
    calibration = str(model_cfg.get("calibration", "platt")).lower()
    if calibration not in {"platt", "sigmoid"}:
        raise ValueError(f"SVM calibration must be Platt/sigmoid; got {calibration!r}")
    # method='sigmoid' == Platt scaling (§8)
    return CalibratedClassifierCV(estimator, method="sigmoid", cv=3)


def _predict_proba_positive(estimator: Any, x: Any) -> np.ndarray:
    if not hasattr(estimator, "predict_proba"):
        raise RuntimeError("estimator lacks predict_proba (required for ECE)")
    proba = np.asarray(estimator.predict_proba(x), dtype=float)
    if proba.ndim != 2 or proba.shape[1] < 2:
        raise RuntimeError(f"unexpected predict_proba shape: {proba.shape}")
    # Column corresponding to class label 1 (fake)
    classes = list(getattr(estimator, "classes_", [0, 1]))
    if 1 in classes:
        return proba[:, classes.index(1)]
    return proba[:, -1]


def predict_classical(
    vectorizer: TfidfVectorizer,
    estimator: Any,
    texts: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Transform texts with a fitted vectorizer and predict labels + P(fake=1)."""
    x = vectorizer.transform(texts)
    y_pred = np.asarray(estimator.predict(x), dtype=int)
    y_prob = _predict_proba_positive(estimator, x)
    return y_pred, y_prob


def train_classical(
    model_id: str,
    texts_train: list[str],
    y_train: Any,
    texts_val: list[str],
    y_val: Any,
    *,
    seed: int,
    texts_test: list[str] | None = None,
    artifacts_dir: str | Path = DEFAULT_ARTIFACTS_DIR,
    config_dir: str | Path | None = None,
    save_model: bool = True,
) -> ClassicalTrainResult:
    """Train one classical model with train-only TF-IDF + val macro-F1 grid search.

    Never fits the vectorizer on val/test. Never rebalances the test set.
    """
    model_id = str(model_id).upper()
    if model_id not in CLASSICAL_MODEL_IDS:
        raise ValueError(f"model_id must be one of {CLASSICAL_MODEL_IDS}; got {model_id!r}")

    _assert_smote_prohibited(config_dir)
    cfg = load_config("classical_grids", config_dir)
    raw = load_config_raw("classical_grids", config_dir)
    models_raw = raw.get("models") or {}
    if model_id not in models_raw:
        raise KeyError(f"{model_id} missing from classical_grids.yaml models")
    model_cfg = dict(models_raw[model_id])
    grid = dict(model_cfg.get("grid") or {})
    combos = _expand_grid(grid)
    if str(cfg.selection.metric) != "val_macro_f1":
        raise ValueError(
            f"selection.metric must be 'val_macro_f1' (got {cfg.selection.metric!r})"
        )

    y_tr = np.asarray(y_train, dtype=int).ravel()
    y_va = np.asarray(y_val, dtype=int).ravel()
    if len(texts_train) != len(y_tr) or len(texts_val) != len(y_va):
        raise ValueError("texts/labels length mismatch for train or val")
    if len(texts_train) == 0 or len(texts_val) == 0:
        raise ValueError("train and val must be non-empty")

    vectorizer = build_tfidf_vectorizer(config_dir=config_dir)
    # TRAIN ONLY — never include val/test (anti-leakage).
    x_train = vectorizer.fit_transform(texts_train)
    x_val = vectorizer.transform(texts_val)

    best_score = -1.0
    best_params: dict[str, Any] | None = None
    grid_scores: list[dict[str, Any]] = []

    for params in combos:
        est = _make_base_estimator(model_id, params, seed=seed)
        est = _wrap_estimator(model_id, est, seed=seed, model_cfg=model_cfg)
        est.fit(x_train, y_tr)
        pred_va = np.asarray(est.predict(x_val), dtype=int)
        score = float(f1_score(y_va, pred_va, average="macro", zero_division=0))
        grid_scores.append({"params": dict(params), "val_macro_f1": score})
        if score > best_score:
            best_score = score
            best_params = dict(params)

    if best_params is None:
        raise RuntimeError(f"{model_id}: grid search produced no candidates")

    logger.info(
        "%s seed=%s selected params=%s val_macro_f1=%.4f (%d grid points)",
        model_id,
        seed,
        best_params,
        best_score,
        len(combos),
    )

    # Refit winner on TRAIN only.
    final_est = _make_base_estimator(model_id, best_params, seed=seed)
    final_est = _wrap_estimator(model_id, final_est, seed=seed, model_cfg=model_cfg)
    if bool(cfg.selection.refit):
        final_est.fit(x_train, y_tr)
    else:  # pragma: no cover - frozen config always refits
        final_est.fit(x_train, y_tr)

    y_pred_val, y_prob_val = predict_classical(vectorizer, final_est, texts_val)
    y_pred_test = y_prob_test = None
    if texts_test is not None:
        y_pred_test, y_prob_test = predict_classical(vectorizer, final_est, texts_test)

    artifacts = Path(artifacts_dir)
    tfidf_path = save_tfidf(vectorizer, artifacts / f"tfidf_{seed}.pkl")
    model_path = None
    if save_model:
        model_path = artifacts / f"classical_{model_id}_seed{seed}.pkl"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(final_est, model_path)

    return ClassicalTrainResult(
        model_id=model_id,
        seed=seed,
        vectorizer=vectorizer,
        estimator=final_est,
        best_params=best_params,
        val_macro_f1=best_score,
        grid_scores=grid_scores,
        y_pred_val=y_pred_val,
        y_prob_val=y_prob_val,
        y_pred_test=y_pred_test,
        y_prob_test=y_prob_test,
        tfidf_path=tfidf_path,
        model_path=model_path,
    )


__all__ = [
    "CLASSICAL_MODEL_IDS",
    "ClassicalTrainResult",
    "build_tfidf_vectorizer",
    "combine_texts",
    "load_tfidf",
    "predict_classical",
    "save_tfidf",
    "train_classical",
]
