"""Tests for encoder sliding-window tokenization + predict pooling (S17/S19)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from fnb.models.encoder import mean_pool_logits_by_doc, predict, predict_from_loader
from fnb.training.datasets import (
    WindowedTextDataset,
    collate_windows,
    expected_num_windows,
    load_sequence_params,
    sliding_window_spans,
    tokenize_document_windows,
)
from fnb.training.loop import evaluate_macro_f1
from fnb.utils.seeding import set_global_seed


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


# --- S19: predict + window logit pooling -------------------------------------


class _TinyEncoder(nn.Module):
    def __init__(self, vocab_size: int = 8000, hidden: int = 16, num_labels: int = 2) -> None:
        super().__init__()
        self.emb = nn.Embedding(vocab_size, hidden, padding_idx=0)
        self.fc = nn.Linear(hidden, num_labels)

    def forward(self, input_ids, attention_mask=None, labels=None):
        if attention_mask is None:
            attention_mask = (input_ids != 0).long()
        mask = attention_mask.unsqueeze(-1).float()
        h = self.emb(input_ids)
        pooled = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        return {"logits": self.fc(pooled)}


def test_pooled_logit_equals_mean_of_window_logits():
    # k=3 windows for one document → pooled == mean
    window_logits = torch.tensor(
        [[1.0, 0.0], [3.0, 2.0], [5.0, 4.0]],
        dtype=torch.float32,
    )
    docs = torch.tensor([7, 7, 7], dtype=torch.long)
    unique, pooled = mean_pool_logits_by_doc(window_logits, docs)
    assert unique.tolist() == [7]
    assert pooled[0].tolist() == pytest.approx(
        window_logits.mean(dim=0).tolist()
    )


def test_predict_probs_sum_to_one_and_matches_val_path(tmp_path: Path):
    cfg_dir = tmp_path / "configs"
    _write_encoder_config(cfg_dir)
    tok = _StubTokenizer()
    texts = [
        "short real news",
        " ".join([f"longfake{i}" for i in range(600)]),
    ]
    labels = [0, 1]
    ds = WindowedTextDataset(texts, labels, tok, config_dir=cfg_dir)
    assert len(ds.windows_for_doc(1)) > 1

    set_global_seed(42)
    model = _TinyEncoder()
    y_pred, y_prob = predict(model, ds, batch_size=4, device="cpu")
    assert y_pred.shape == (2,)
    assert y_prob.shape == (2, 2)
    np.testing.assert_allclose(y_prob.sum(axis=1), np.ones(2), atol=1e-6)

    loader = DataLoader(ds, batch_size=4, shuffle=False, collate_fn=collate_windows)
    from_loader = predict_from_loader(model, loader, device="cpu")
    np.testing.assert_array_equal(y_pred, from_loader.y_pred)
    np.testing.assert_allclose(y_prob, from_loader.y_prob, atol=1e-6)

    # Same argmax path as training validation
    f1 = evaluate_macro_f1(model, loader, device=torch.device("cpu"))
    assert 0.0 <= f1 <= 1.0
    # Manual F1 from predict matches evaluate_macro_f1
    from sklearn.metrics import f1_score

    f1_manual = f1_score(labels, y_pred, average="macro", zero_division=0)
    assert f1 == pytest.approx(f1_manual)


def test_predict_deterministic_under_fixed_seed(tmp_path: Path):
    cfg_dir = tmp_path / "configs"
    _write_encoder_config(cfg_dir)
    tok = _StubTokenizer()
    texts = ["alpha beta gamma", " ".join([f"w{i}" for i in range(400)])]
    ds = WindowedTextDataset(texts, [0, 1], tok, config_dir=cfg_dir)

    def _run(seed: int):
        set_global_seed(seed)
        model = _TinyEncoder()
        # Freeze weights after seeded init
        return predict(model, ds, batch_size=2, device="cpu")

    a_pred, a_prob = _run(13)
    b_pred, b_prob = _run(13)
    c_pred, c_prob = _run(99)
    np.testing.assert_array_equal(a_pred, b_pred)
    np.testing.assert_allclose(a_prob, b_prob, atol=1e-7)
    # Different seed → different init → typically different probs
    assert not np.allclose(a_prob, c_prob) or not np.array_equal(a_pred, c_pred)
