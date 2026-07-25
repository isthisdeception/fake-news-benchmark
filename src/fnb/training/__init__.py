"""fnb.training — datasets, custom loop, scheduler, early stopping."""

from __future__ import annotations

from .datasets import (
    WindowedTextDataset,
    expected_num_windows,
    load_sequence_params,
    sliding_window_spans,
    tokenize_document_windows,
)
from .early_stopping import EarlyStopping
from .loop import TrainResult, evaluate_macro_f1, train
from .scheduler import build_warmup_cosine_scheduler, warmup_cosine_lambda

__all__ = [
    "EarlyStopping",
    "TrainResult",
    "WindowedTextDataset",
    "build_warmup_cosine_scheduler",
    "evaluate_macro_f1",
    "expected_num_windows",
    "load_sequence_params",
    "sliding_window_spans",
    "tokenize_document_windows",
    "train",
    "warmup_cosine_lambda",
]
