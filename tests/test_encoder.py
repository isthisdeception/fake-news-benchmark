"""Tests for encoder sliding-window tokenization (S17 / protocol §6.1)."""

from __future__ import annotations

from pathlib import Path

import torch
import yaml

from fnb.training.datasets import (
    WindowedTextDataset,
    expected_num_windows,
    load_sequence_params,
    sliding_window_spans,
    tokenize_document_windows,
)


class _StubTokenizer:
    """Whitespace tokenizer with CLS/SEP/PAD — no HF download required."""

    def __init__(self) -> None:
        self.cls_token_id = 101
        self.sep_token_id = 102
        self.pad_token_id = 0
        self._vocab: dict[str, int] = {}
        self._next = 1000

    def encode(self, text: str, *, add_special_tokens: bool = True, truncation: bool = False):
        del truncation  # never truncate in full-document encode path
        ids = []
        for tok in str(text).split():
            if tok not in self._vocab:
                self._vocab[tok] = self._next
                self._next += 1
            ids.append(self._vocab[tok])
        if add_special_tokens:
            return [self.cls_token_id, *ids, self.sep_token_id]
        return ids


def _write_encoder_config(cfg_dir: Path) -> None:
    cfg_dir.mkdir(parents=True, exist_ok=True)
    data = {
        "models": {"BERT": "bert-base-uncased"},
        "optimizer": {"name": "adamw", "weight_decay": 0.01, "learning_rate": 2.0e-5},
        "scheduler": {
            "name": "linear_warmup_cosine",
            "warmup_ratio": 0.06,
            "decay_to": 0.0,
        },
        "training": {
            "batch_size": 16,
            "max_epochs": 10,
            "early_stopping": {
                "monitor": "val_macro_f1",
                "patience": 3,
                "mode": "max",
            },
        },
        "sequence": {
            "max_length": 512,
            "sliding_window": {
                "window_tokens": 512,
                "stride_tokens": 128,
                "window_logits_pooling": "mean",
            },
            "title_only_max_length": 64,
        },
        "loss": {
            "type": "class_weighted_cross_entropy",
            "class_weighting": "inverse_frequency",
            "smote": False,
        },
    }
    (cfg_dir / "encoder.yaml").write_text(yaml.dump(data), encoding="utf-8")


def test_load_sequence_params_from_frozen_config():
    params = load_sequence_params()  # repo configs/encoder.yaml
    assert params.window_tokens == 512
    assert params.stride_tokens == 128
    assert params.title_only_max_length == 64
    assert params.window_logits_pooling == "mean"


def test_expected_windows_long_document_512_128():
    # Content tokens only; window_tokens=512 ⇒ 510 content slots (CLS+SEP).
    n_content = 2000
    n_win = expected_num_windows(n_content, window_tokens=512, stride_tokens=128)
    spans = sliding_window_spans(n_content, window_content=510, stride=128)
    assert n_win == len(spans)
    assert n_win > 1
    # Full coverage: first start 0, last end == n_content
    assert spans[0][0] == 0
    assert spans[-1][1] == n_content
    # Manual: starts 0,128,..., until covered
    assert n_win == expected_num_windows(2000, window_tokens=512, stride_tokens=128)


def test_short_document_yields_one_window():
    tok = _StubTokenizer()
    text = "short document only a few tokens"
    windows = tokenize_document_windows(
        text, tok, window_tokens=512, stride_tokens=128
    )
    assert len(windows) == 1
    assert len(windows[0]["input_ids"]) == 512
    assert sum(windows[0]["attention_mask"]) == len(text.split()) + 2  # CLS+SEP


def test_long_document_window_count_matches_formula():
    tok = _StubTokenizer()
    # 1200 content tokens
    text = " ".join([f"tok{i}" for i in range(1200)])
    windows = tokenize_document_windows(
        text, tok, window_tokens=512, stride_tokens=128
    )
    assert len(windows) == expected_num_windows(
        1200, window_tokens=512, stride_tokens=128, num_special_tokens=2
    )
    assert len(windows) > 1
    for w in windows:
        assert len(w["input_ids"]) == 512
        assert len(w["attention_mask"]) == 512


def test_title_only_caps_at_64():
    tok = _StubTokenizer()
    text = " ".join([f"word{i}" for i in range(200)])
    windows = tokenize_document_windows(
        text,
        tok,
        window_tokens=512,
        stride_tokens=128,
        title_only=True,
        title_only_max_length=64,
    )
    assert len(windows) == 1
    assert len(windows[0]["input_ids"]) == 64
    # At most 62 content tokens + CLS + SEP
    assert sum(windows[0]["attention_mask"]) <= 64
    assert sum(windows[0]["attention_mask"]) == 64  # fully packed from long title


def test_dataset_preserves_document_window_mapping(tmp_path: Path):
    cfg_dir = tmp_path / "configs"
    _write_encoder_config(cfg_dir)
    tok = _StubTokenizer()
    texts = [
        "short one",
        " ".join([f"long{i}" for i in range(800)]),
        "medium length document with several tokens here",
    ]
    labels = [0, 1, 0]
    ds = WindowedTextDataset(
        texts,
        labels,
        tok,
        config_dir=cfg_dir,
    )
    assert ds.n_documents == 3
    # Doc 0 short → 1 window; doc 1 long → many
    assert len(ds.windows_for_doc(0)) == 1
    assert len(ds.windows_for_doc(1)) == expected_num_windows(
        800, window_tokens=512, stride_tokens=128
    )
    assert len(ds.windows_for_doc(1)) > 1

    # Every window carries the right doc_idx; mapping is exhaustive/disjoint
    seen: set[int] = set()
    for doc_idx in range(3):
        idxs = ds.windows_for_doc(doc_idx)
        for i in idxs:
            item = ds[i]
            assert int(item["doc_idx"]) == doc_idx
            assert int(item["labels"]) == labels[doc_idx]
            assert item["input_ids"].dtype == torch.long
            seen.add(i)
    assert seen == set(range(len(ds)))


def test_title_only_dataset_mode(tmp_path: Path):
    cfg_dir = tmp_path / "configs"
    _write_encoder_config(cfg_dir)
    tok = _StubTokenizer()
    texts = [" ".join([f"t{i}" for i in range(100)])]
    ds = WindowedTextDataset(
        texts,
        [1],
        tok,
        config_dir=cfg_dir,
        title_only=True,
    )
    assert len(ds) == 1
    assert ds[0]["input_ids"].numel() == 64
