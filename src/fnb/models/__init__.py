"""fnb.models — classical, encoder, source-only, and QLoRA LLM wrappers."""

from __future__ import annotations

from .classical import (
    CLASSICAL_MODEL_IDS,
    ClassicalTrainResult,
    build_tfidf_vectorizer,
    combine_texts,
    predict_classical,
    save_tfidf,
    train_classical,
)
from .encoder import (
    EncoderClassifier,
    build_encoder,
    inverse_frequency_class_weights,
    mean_pool_logits_by_doc,
    resolve_encoder_name,
)

__all__ = [
    "CLASSICAL_MODEL_IDS",
    "ClassicalTrainResult",
    "EncoderClassifier",
    "build_encoder",
    "build_tfidf_vectorizer",
    "combine_texts",
    "inverse_frequency_class_weights",
    "mean_pool_logits_by_doc",
    "predict_classical",
    "resolve_encoder_name",
    "save_tfidf",
    "train_classical",
]
