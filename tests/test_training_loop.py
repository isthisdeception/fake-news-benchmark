"""Tests for custom encoder training loop (S18) — no HF Trainer."""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from fnb.models.encoder import inverse_frequency_class_weights, mean_pool_logits_by_doc
from fnb.training.datasets import WindowedTextDataset, collate_windows
from fnb.training.early_stopping import EarlyStopping
from fnb.training.loop import evaluate_macro_f1, train
from fnb.training.scheduler import build_warmup_cosine_scheduler, warmup_cosine_lambda


class _StubTokenizer:
    def __init__(self) -> None:
        self.cls_token_id = 101
        self.sep_token_id = 102
        self.pad_token_id = 0
        self._vocab: dict[str, int] = {}
        self._next = 10

    def encode(self, text: str, *, add_special_tokens: bool = True, truncation: bool = False):
        del truncation
        ids = []
        for tok in str(text).split():
            if tok not in self._vocab:
                self._vocab[tok] = self._next
                self._next += 1
            ids.append(self._vocab[tok])
        if add_special_tokens:
            return [self.cls_token_id, *ids, self.sep_token_id]
        return ids


class TinyEncoder(nn.Module):
    """Minimal stand-in for EncoderClassifier (no HF download)."""

    def __init__(self, vocab_size: int = 5000, hidden: int = 32, num_labels: int = 2) -> None:
        super().__init__()
        self.emb = nn.Embedding(vocab_size, hidden, padding_idx=0)
        self.fc = nn.Linear(hidden, num_labels)

    def forward(self, input_ids, attention_mask=None, labels=None):
        if attention_mask is None:
            attention_mask = (input_ids != 0).long()
        mask = attention_mask.unsqueeze(-1).float()
        h = self.emb(input_ids)
        pooled = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        logits = self.fc(pooled)
        return {"logits": logits}


def _write_encoder_cfg(cfg_dir: Path) -> None:
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
            "batch_size": 4,
            "max_epochs": 10,
            "early_stopping": {
                "monitor": "val_macro_f1",
                "patience": 3,
                "mode": "max",
            },
        },
        "sequence": {
            "max_length": 32,
            "sliding_window": {
                "window_tokens": 32,
                "stride_tokens": 8,
                "window_logits_pooling": "mean",
            },
            "title_only_max_length": 16,
        },
        "loss": {
            "type": "class_weighted_cross_entropy",
            "class_weighting": "inverse_frequency",
            "smote": False,
        },
    }
    (cfg_dir / "encoder.yaml").write_text(yaml.dump(data), encoding="utf-8")


def _loaders(cfg_dir: Path, window_tokens: int = 32):
    tok = _StubTokenizer()
    # Separable bag-of-words docs
    real = [f"real news government policy update {i}" for i in range(8)]
    fake = [f"fake viral hoax conspiracy rumor {i}" for i in range(8)]
    texts = real + fake
    labels = [0] * 8 + [1] * 8
    # tiny val
    val_texts = [
        "real news government policy",
        "fake viral hoax conspiracy",
        "real policy update government",
        "fake rumor conspiracy viral",
    ]
    val_labels = [0, 1, 0, 1]

    train_ds = WindowedTextDataset(
        texts,
        labels,
        tok,
        window_tokens=window_tokens,
        stride_tokens=8,
        config_dir=cfg_dir,
    )
    val_ds = WindowedTextDataset(
        val_texts,
        val_labels,
        tok,
        window_tokens=window_tokens,
        stride_tokens=8,
        config_dir=cfg_dir,
    )
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, collate_fn=collate_windows)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, collate_fn=collate_windows)
    return train_loader, val_loader, labels


def test_no_trainer_imported_in_training_stack():
    root = Path(__file__).resolve().parents[1] / "src" / "fnb"
    files = [
        root / "training" / "loop.py",
        root / "training" / "scheduler.py",
        root / "training" / "early_stopping.py",
        root / "models" / "encoder.py",
    ]
    banned = ("Trainer", "transformers.Trainer")
    for path in files:
        text = path.read_text(encoding="utf-8")
        # AST check: no import of Trainer symbol
        tree = ast.parse(text)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    assert alias.name != "Trainer", f"{path} imports Trainer"
                    assert alias.asname != "Trainer", f"{path} imports Trainer"
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert "Trainer" not in alias.name
        for token in banned:
            # Allow mentioning the prohibition in comments/docstrings only if
            # not an import line. Soft check: no 'import ... Trainer'
            for line in text.splitlines():
                stripped = line.strip()
                if stripped.startswith("#") or stripped.startswith('"""') or stripped.startswith("'''"):
                    continue
                if stripped.startswith("from ") or stripped.startswith("import "):
                    assert "Trainer" not in stripped, f"{path}: {stripped}"


def test_class_weights_match_inverse_frequency():
    labels = [0, 0, 0, 1]
    w = inverse_frequency_class_weights(labels, n_classes=2).numpy()
    # N=4, K=2 → w0=4/(2*3)=2/3, w1=4/(2*1)=2
    assert w[0] == pytest.approx(4 / (2 * 3))
    assert w[1] == pytest.approx(4 / (2 * 1))


def test_mean_pool_logits_by_doc():
    logits = torch.tensor([[1.0, 0.0], [3.0, 2.0], [0.0, 4.0]], dtype=torch.float32)
    docs = torch.tensor([0, 0, 1], dtype=torch.long)
    unique, pooled = mean_pool_logits_by_doc(logits, docs)
    assert unique.tolist() == [0, 1]
    assert pooled[0].tolist() == pytest.approx([2.0, 1.0])
    assert pooled[1].tolist() == pytest.approx([0.0, 4.0])


def test_scheduler_decays_to_near_zero():
    model = TinyEncoder()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    n_steps = 100
    sched = build_warmup_cosine_scheduler(
        opt, num_training_steps=n_steps, warmup_ratio=0.06, decay_to=0.0
    )
    lrs = []
    for _ in range(n_steps):
        opt.step()
        sched.step()
        lrs.append(opt.param_groups[0]["lr"])
    assert lrs[0] > 0
    assert lrs[-1] == pytest.approx(0.0, abs=1e-8)
    # Mid warmup should be below peak
    assert max(lrs) == pytest.approx(1e-3, rel=1e-6)
    assert warmup_cosine_lambda(
        n_steps - 1, num_warmup_steps=int(0.06 * n_steps), num_training_steps=n_steps
    ) == pytest.approx(0.0, abs=1e-8)


def test_early_stopping_fires_after_patience_3():
    es = EarlyStopping(patience=3, mode="max")
    # Improve once, then stall
    assert es.step(0.5).improved
    assert not es.step(0.4).should_stop
    assert not es.step(0.4).should_stop
    st = es.step(0.4)
    assert st.should_stop
    assert st.bad_epochs == 3


def test_train_loss_decreases_on_tiny_overfit(tmp_path: Path):
    cfg_dir = tmp_path / "configs"
    _write_encoder_cfg(cfg_dir)
    train_loader, val_loader, labels = _loaders(cfg_dir)
    model = TinyEncoder()
    result = train(
        model,
        train_loader,
        val_loader,
        seed=13,
        train_labels=labels,
        config_dir=cfg_dir,
        device="cpu",
        checkpoint_dir=tmp_path / "ckpt",
        max_epochs=8,
        learning_rate=5e-2,  # higher LR for fast overfit on tiny net
    )
    assert len(result.history) >= 2
    assert result.history[-1].train_loss < result.history[0].train_loss
    assert result.checkpoint_path is not None
    assert result.checkpoint_path.is_file()
    assert result.class_weights == pytest.approx(
        inverse_frequency_class_weights(labels).tolist()
    )
    # evaluate path runs
    f1 = evaluate_macro_f1(model, val_loader, device=torch.device("cpu"))
    assert 0.0 <= f1 <= 1.0
