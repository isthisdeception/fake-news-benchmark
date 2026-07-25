"""HuggingFace encoder wrapper + window-logit pooling (protocol §5, §6.1).

Does **not** use ``transformers.Trainer``. Forward returns per-window logits;
document-level predictions mean-pool **logits** (not probabilities) across
windows sharing the same ``doc_idx``. Softmax is applied **after** pooling
(identical path for validation and test inference).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from fnb.config import load_config


def resolve_encoder_name(
    model_id: str,
    *,
    config_dir: str | Path | None = None,
) -> str:
    """Map frozen model ID (e.g. ``BERT``) → HF hub name from ``encoder.yaml``."""
    cfg = load_config("encoder", config_dir)
    key = str(model_id).upper()
    if key not in cfg.models:
        raise KeyError(f"unknown encoder model_id {model_id!r}; known={list(cfg.models)}")
    return str(cfg.models[key])


def inverse_frequency_class_weights(
    labels: Sequence[int] | np.ndarray,
    *,
    n_classes: int = 2,
) -> torch.Tensor:
    """Inverse-frequency class weights (§7): ``N / (K * n_c)``."""
    y = np.asarray(labels, dtype=int).ravel()
    if y.size == 0:
        raise ValueError("labels must be non-empty")
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    # Avoid div-by-zero for missing classes
    counts = np.maximum(counts, 1.0)
    n = float(y.size)
    weights = n / (float(n_classes) * counts)
    return torch.tensor(weights, dtype=torch.float32)


def mean_pool_logits_by_doc(
    logits: torch.Tensor,
    doc_idx: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean-pool window logits per document (§6.1).

    Parameters
    ----------
    logits:
        ``[n_windows, n_classes]``
    doc_idx:
        ``[n_windows]`` document ids

    Returns
    -------
    unique_docs:
        Sorted unique document ids ``[n_docs]``
    pooled:
        Mean logits ``[n_docs, n_classes]`` aligned with ``unique_docs``
    """
    if logits.ndim != 2:
        raise ValueError(f"logits must be 2-d; got {tuple(logits.shape)}")
    if doc_idx.ndim != 1 or doc_idx.shape[0] != logits.shape[0]:
        raise ValueError("doc_idx must be 1-d with length == n_windows")

    docs = doc_idx.detach().to(dtype=torch.long, device=logits.device)
    unique = torch.unique(docs, sorted=True)
    pooled = [logits[docs == d].mean(dim=0) for d in unique]
    return unique, torch.stack(pooled, dim=0)


class EncoderClassifier(nn.Module):
    """HF encoder + linear classification head (binary fake/real)."""

    def __init__(
        self,
        model_name_or_path: str,
        *,
        num_labels: int = 2,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        # Local import keeps this module free of Trainer symbols.
        from transformers import AutoConfig, AutoModel

        if pretrained:
            self.encoder = AutoModel.from_pretrained(model_name_or_path)
            hidden = int(self.encoder.config.hidden_size)
        else:
            config = AutoConfig.from_pretrained(model_name_or_path)
            self.encoder = AutoModel.from_config(config)
            hidden = int(config.hidden_size)
        self.classifier = nn.Linear(hidden, num_labels)
        self.num_labels = int(num_labels)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # CLS / first-token pooling (BERT-style); works for RoBERTa/Distil/ALBERT too.
        cls = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls)
        result: dict[str, Any] = {"logits": logits}
        if labels is not None:
            # Unweighted CE here; the training loop applies class weights.
            result["loss"] = F.cross_entropy(logits, labels.long())
        return result


def build_encoder(
    model_id: str,
    *,
    num_labels: int = 2,
    config_dir: str | Path | None = None,
    pretrained: bool = True,
) -> EncoderClassifier:
    """Construct an :class:`EncoderClassifier` for a frozen model ID."""
    name = resolve_encoder_name(model_id, config_dir=config_dir)
    return EncoderClassifier(name, num_labels=num_labels, pretrained=pretrained)


@dataclass(frozen=True)
class EncoderPredictResult:
    """Per-document predictions after mean-pooling window logits."""

    doc_idx: np.ndarray  # [n_docs] sorted document ids
    y_pred: np.ndarray  # [n_docs]
    y_prob: np.ndarray  # [n_docs, n_classes] softmax over pooled logits
    logits: np.ndarray  # [n_docs, n_classes] mean-pooled logits


def _forward_window_logits(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run ``model`` over a window loader; return ``(logits, doc_idx)`` on CPU."""
    model.eval()
    all_logits: list[torch.Tensor] = []
    all_docs: list[torch.Tensor] = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            doc_idx = batch["doc_idx"].to(device)
            out = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = out["logits"] if isinstance(out, dict) else out
            all_logits.append(logits.detach().cpu())
            all_docs.append(doc_idx.detach().cpu())
    if not all_logits:
        empty_logits = torch.zeros((0, 2), dtype=torch.float32)
        empty_docs = torch.zeros((0,), dtype=torch.long)
        return empty_logits, empty_docs
    return torch.cat(all_logits, dim=0), torch.cat(all_docs, dim=0)


def predict_from_loader(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: str | torch.device | None = None,
) -> EncoderPredictResult:
    """Document-level ``(y_pred, y_prob)`` from a window ``DataLoader``.

    Pools **logits** per ``doc_idx`` (same helper as training validation), then
    applies softmax. Never pools probabilities.
    """
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = model.to(dev)
    logits, docs = _forward_window_logits(model, loader, device=dev)
    if logits.shape[0] == 0:
        return EncoderPredictResult(
            doc_idx=np.zeros((0,), dtype=np.int64),
            y_pred=np.zeros((0,), dtype=np.int64),
            y_prob=np.zeros((0, 2), dtype=np.float64),
            logits=np.zeros((0, 2), dtype=np.float64),
        )
    unique_docs, pooled = mean_pool_logits_by_doc(logits, docs)
    probs = F.softmax(pooled, dim=-1)
    y_pred = pooled.argmax(dim=-1)
    return EncoderPredictResult(
        doc_idx=unique_docs.numpy().astype(np.int64, copy=False),
        y_pred=y_pred.numpy().astype(np.int64, copy=False),
        y_prob=probs.numpy().astype(np.float64, copy=False),
        logits=pooled.numpy().astype(np.float64, copy=False),
    )


def predict(
    model: nn.Module,
    dataset: Dataset,
    *,
    batch_size: int = 16,
    device: str | torch.device | None = None,
    collate_fn: Any | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Predict per document from a :class:`WindowedTextDataset` (or compatible).

    Returns
    -------
    y_pred, y_prob
        ``y_pred`` shape ``[n_docs]``; ``y_prob`` shape ``[n_docs, n_classes]``
        (softmax over **mean-pooled logits**). Document order follows sorted
        ``doc_idx`` (0..n-1 when the dataset covers all documents).
    """
    if collate_fn is None:
        from fnb.training.datasets import collate_windows

        collate_fn = collate_windows
    loader = DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        collate_fn=collate_fn,
    )
    result = predict_from_loader(model, loader, device=device)
    return result.y_pred, result.y_prob


__all__ = [
    "EncoderClassifier",
    "EncoderPredictResult",
    "build_encoder",
    "inverse_frequency_class_weights",
    "mean_pool_logits_by_doc",
    "predict",
    "predict_from_loader",
    "resolve_encoder_name",
]
