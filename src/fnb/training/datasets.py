"""Sliding-window encoder datasets for ``vCLEAN-N`` (protocol §6.1).

Transformers must see the **full document**: tokenize without single-pass
truncation-to-512, then emit overlapping windows

* ``window_tokens = 512``
* ``stride_tokens = 128``
* window logits are mean-pooled per document at inference (pooling itself is
  elsewhere; this module preserves ``doc_idx`` ↔ window mapping)

Title-only (RQ-E / HP-TITLE): cap at ``title_only_max_length = 64`` (single
window; no classical stopword/lemmatization here).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch
from torch.utils.data import Dataset

from fnb.config import load_config

Granularity = Literal["title_content", "title", "content"]


@dataclass(frozen=True)
class SequenceParams:
    """Frozen sequence / window settings from ``configs/encoder.yaml``."""

    max_length: int
    window_tokens: int
    stride_tokens: int
    window_logits_pooling: str
    title_only_max_length: int


def load_sequence_params(config_dir: str | Path | None = None) -> SequenceParams:
    """Read sliding-window constants from ``encoder.yaml`` (§6.1)."""
    cfg = load_config("encoder", config_dir)
    sw = cfg.sequence.sliding_window
    return SequenceParams(
        max_length=int(cfg.sequence.max_length),
        window_tokens=int(sw.window_tokens),
        stride_tokens=int(sw.stride_tokens),
        window_logits_pooling=str(sw.window_logits_pooling),
        title_only_max_length=int(cfg.sequence.title_only_max_length),
    )


def sliding_window_spans(
    n_content_tokens: int,
    *,
    window_content: int,
    stride: int,
) -> list[tuple[int, int]]:
    """Return ``(start, end)`` spans covering all content tokens (full document).

    ``window_content`` is the max number of *content* tokens per window
    (i.e. ``window_tokens - special_tokens``). Never drops the tail.
    """
    if window_content < 1:
        raise ValueError(f"window_content must be >= 1; got {window_content}")
    if stride < 1:
        raise ValueError(f"stride must be >= 1; got {stride}")
    if n_content_tokens <= 0:
        return [(0, 0)]
    if n_content_tokens <= window_content:
        return [(0, n_content_tokens)]

    spans: list[tuple[int, int]] = []
    start = 0
    while start < n_content_tokens:
        end = min(start + window_content, n_content_tokens)
        spans.append((start, end))
        if end >= n_content_tokens:
            break
        start += stride
    return spans


def expected_num_windows(
    n_content_tokens: int,
    *,
    window_tokens: int,
    stride_tokens: int,
    num_special_tokens: int = 2,
) -> int:
    """Number of windows for a document with ``n_content_tokens`` content ids."""
    window_content = window_tokens - num_special_tokens
    if window_content < 1:
        raise ValueError(
            f"window_tokens={window_tokens} too small for {num_special_tokens} special tokens"
        )
    return len(
        sliding_window_spans(
            n_content_tokens, window_content=window_content, stride=stride_tokens
        )
    )


def _content_token_ids(tokenizer: Any, text: str) -> list[int]:
    """Tokenize text to content ids (no special tokens, no truncation)."""
    # Prefer encode API; never pass max_length here — full document required.
    if hasattr(tokenizer, "encode"):
        ids = tokenizer.encode(
            text,
            add_special_tokens=False,
            truncation=False,
        )
        return list(ids)
    encoded = tokenizer(text, add_special_tokens=False, truncation=False)
    return list(encoded["input_ids"])


def _pad_to_length(
    input_ids: list[int],
    *,
    length: int,
    pad_id: int,
) -> tuple[list[int], list[int]]:
    if len(input_ids) > length:
        input_ids = input_ids[:length]
    attn = [1] * len(input_ids) + [0] * (length - len(input_ids))
    input_ids = input_ids + [pad_id] * (length - len(input_ids))
    return input_ids, attn


def tokenize_document_windows(
    text: str,
    tokenizer: Any,
    *,
    window_tokens: int = 512,
    stride_tokens: int = 128,
    title_only: bool = False,
    title_only_max_length: int = 64,
) -> list[dict[str, list[int]]]:
    """Tokenize one document into window dicts ``{input_ids, attention_mask}``.

    Full-document mode uses sliding windows. Title-only mode emits **one**
    window capped at ``title_only_max_length`` (protocol §6.1 / HP-TITLE).
    """
    pad_id = int(getattr(tokenizer, "pad_token_id", None) or 0)
    cls_id = getattr(tokenizer, "cls_token_id", None)
    sep_id = getattr(tokenizer, "sep_token_id", None)
    # RoBERTa uses bos/eos instead of cls/sep
    if cls_id is None:
        cls_id = getattr(tokenizer, "bos_token_id", None)
    if sep_id is None:
        sep_id = getattr(tokenizer, "eos_token_id", None)
    if cls_id is None or sep_id is None:
        # Fallback: no specials (tests / weird tokenizers)
        special_prefix: list[int] = []
        special_suffix: list[int] = []
    else:
        special_prefix = [int(cls_id)]
        special_suffix = [int(sep_id)]
    n_special = len(special_prefix) + len(special_suffix)

    if title_only:
        max_len = int(title_only_max_length)
        content = _content_token_ids(tokenizer, text)
        budget = max_len - n_special
        if budget < 0:
            raise ValueError(f"title_only_max_length={max_len} smaller than specials")
        content = content[:budget]
        ids = special_prefix + content + special_suffix
        ids, attn = _pad_to_length(ids, length=max_len, pad_id=pad_id)
        return [{"input_ids": ids, "attention_mask": attn}]

    content = _content_token_ids(tokenizer, text)
    window_content = window_tokens - n_special
    if window_content < 1:
        raise ValueError(
            f"window_tokens={window_tokens} too small for {n_special} special tokens"
        )
    spans = sliding_window_spans(
        len(content), window_content=window_content, stride=stride_tokens
    )
    windows: list[dict[str, list[int]]] = []
    for start, end in spans:
        chunk = content[start:end]
        ids = special_prefix + chunk + special_suffix
        ids, attn = _pad_to_length(ids, length=window_tokens, pad_id=pad_id)
        windows.append({"input_ids": ids, "attention_mask": attn})
    return windows


@dataclass
class WindowRecord:
    """One window instance linked back to its source document."""

    input_ids: list[int]
    attention_mask: list[int]
    label: int
    doc_idx: int
    window_idx: int


class WindowedTextDataset(Dataset):
    """Torch dataset yielding per-window encoder inputs + document index.

    Each ``__getitem__`` returns::

        {
          "input_ids": LongTensor [W],
          "attention_mask": LongTensor [W],
          "labels": LongTensor [],
          "doc_idx": LongTensor [],
          "window_idx": LongTensor [],
        }

    Use ``doc_to_window_indices`` (or the ``doc_idx`` field) to mean-pool window
    logits back to documents at validation/inference.
    """

    def __init__(
        self,
        texts: Sequence[str],
        labels: Sequence[int],
        tokenizer: Any,
        *,
        window_tokens: int | None = None,
        stride_tokens: int | None = None,
        title_only: bool = False,
        title_only_max_length: int | None = None,
        config_dir: str | Path | None = None,
        granularity: Granularity = "title_content",
    ) -> None:
        if len(texts) != len(labels):
            raise ValueError("texts and labels must have the same length")

        params = load_sequence_params(config_dir)
        self.window_tokens = int(params.window_tokens if window_tokens is None else window_tokens)
        self.stride_tokens = int(params.stride_tokens if stride_tokens is None else stride_tokens)
        self.title_only_max_length = int(
            params.title_only_max_length
            if title_only_max_length is None
            else title_only_max_length
        )
        # Title-only mode: explicit flag or granularity == "title"
        self.title_only = bool(title_only or granularity == "title")
        if self.window_tokens != params.max_length and not self.title_only:
            # Keep a soft check: frozen max_length should match window_tokens.
            pass

        self.tokenizer = tokenizer
        self.texts = list(texts)
        self.labels = [int(y) for y in labels]

        self.windows: list[WindowRecord] = []
        self.doc_to_window_indices: dict[int, list[int]] = {}

        for doc_idx, (text, label) in enumerate(zip(self.texts, self.labels, strict=True)):
            win_dicts = tokenize_document_windows(
                text,
                tokenizer,
                window_tokens=self.window_tokens,
                stride_tokens=self.stride_tokens,
                title_only=self.title_only,
                title_only_max_length=self.title_only_max_length,
            )
            idxs: list[int] = []
            for w_i, w in enumerate(win_dicts):
                rec = WindowRecord(
                    input_ids=w["input_ids"],
                    attention_mask=w["attention_mask"],
                    label=label,
                    doc_idx=doc_idx,
                    window_idx=w_i,
                )
                idxs.append(len(self.windows))
                self.windows.append(rec)
            self.doc_to_window_indices[doc_idx] = idxs

        if not self.windows:
            raise ValueError("dataset produced zero windows")

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        rec = self.windows[index]
        return {
            "input_ids": torch.tensor(rec.input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(rec.attention_mask, dtype=torch.long),
            "labels": torch.tensor(rec.label, dtype=torch.long),
            "doc_idx": torch.tensor(rec.doc_idx, dtype=torch.long),
            "window_idx": torch.tensor(rec.window_idx, dtype=torch.long),
        }

    @property
    def n_documents(self) -> int:
        return len(self.texts)

    def windows_for_doc(self, doc_idx: int) -> list[int]:
        """Return dataset indices of all windows belonging to ``doc_idx``."""
        return list(self.doc_to_window_indices[doc_idx])


def collate_windows(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Default collate for :class:`WindowedTextDataset` batches."""
    return {
        key: torch.stack([b[key] for b in batch], dim=0)
        if batch[0][key].ndim > 0
        else torch.stack([b[key] for b in batch], dim=0)
        for key in batch[0]
    }


__all__ = [
    "SequenceParams",
    "WindowRecord",
    "WindowedTextDataset",
    "collate_windows",
    "expected_num_windows",
    "load_sequence_params",
    "sliding_window_spans",
    "tokenize_document_windows",
]
