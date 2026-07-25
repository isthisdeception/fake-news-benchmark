"""HuggingFace encoder wrapper + window-logit pooling (protocol §5, §6.1).

Does **not** use ``transformers.Trainer``. Forward returns per-window logits;
document-level predictions mean-pool **logits** (not probabilities) across
windows sharing the same ``doc_idx``.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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


__all__ = [
    "EncoderClassifier",
    "build_encoder",
    "inverse_frequency_class_weights",
    "mean_pool_logits_by_doc",
    "resolve_encoder_name",
]
