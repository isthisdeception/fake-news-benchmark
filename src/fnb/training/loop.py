"""Hand-coded encoder training loop (protocol §7 / HP-ENCODER).

**No** ``transformers.Trainer`` and **no** Accelerate Trainer. Settings are
read from ``configs/encoder.yaml``:

* AdamW ``weight_decay=0.01``, ``lr=2e-5``
* batch size 16
* linear warmup 6% of steps → cosine decay to 0
* max 10 epochs; early stop on **val macro-F1**, patience 3
* inverse-frequency class-weighted cross-entropy; SMOTE prohibited

Validation mean-pools **window logits** per document (§6.1) before metrics.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from fnb.config import load_config, load_config_raw
from fnb.models.encoder import inverse_frequency_class_weights, predict_from_loader
from fnb.training.early_stopping import EarlyStopping
from fnb.training.scheduler import build_warmup_cosine_scheduler
from fnb.utils.seeding import set_global_seed

logger = logging.getLogger("fnb.training.loop")


@dataclass
class EpochLog:
    epoch: int
    train_loss: float
    val_macro_f1: float
    lr: float
    early_stop_improved: bool
    early_stop_bad_epochs: int
    early_stop: bool


@dataclass
class TrainResult:
    """Outcome of :func:`train`."""

    best_val_macro_f1: float
    best_epoch: int
    checkpoint_path: Path | None
    history: list[EpochLog] = field(default_factory=list)
    class_weights: list[float] = field(default_factory=list)
    stopped_early: bool = False


def _assert_no_smote(config_dir: str | Path | None) -> None:
    enc = load_config("encoder", config_dir)
    if bool(enc.loss.smote):
        raise RuntimeError("SMOTE is PROHIBITED (protocol §7); encoder.yaml disagrees")
    try:
        raw = load_config_raw("preprocessing", config_dir)
    except FileNotFoundError:
        return
    if (raw.get("class_imbalance") or {}).get("smote", False):
        raise RuntimeError("SMOTE is PROHIBITED (protocol §7); preprocessing.yaml disagrees")


def _move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch.items()}


@torch.no_grad()
def evaluate_macro_f1(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
) -> float:
    """Validation macro-F1 via the shared :func:`predict_from_loader` path (§6.1)."""
    # Collect one gold label per document (windows of a doc share the label).
    doc_to_label: dict[int, int] = {}
    for batch in loader:
        docs = batch["doc_idx"].detach().cpu().tolist()
        labs = batch["labels"].detach().cpu().tolist()
        for d, lab in zip(docs, labs, strict=True):
            doc_to_label.setdefault(int(d), int(lab))

    pred = predict_from_loader(model, loader, device=device)
    if pred.doc_idx.size == 0:
        return float("nan")
    y_true = np.asarray([doc_to_label[int(d)] for d in pred.doc_idx], dtype=int)
    return float(f1_score(y_true, pred.y_pred, average="macro", zero_division=0))


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    seed: int,
    train_labels: Sequence[int] | np.ndarray,
    config_dir: str | Path | None = None,
    device: str | torch.device | None = None,
    checkpoint_dir: str | Path | None = None,
    max_epochs: int | None = None,
    learning_rate: float | None = None,
) -> TrainResult:
    """Train ``model`` with the frozen HP-ENCODER loop; return best checkpoint.

    ``train_labels`` are **document-level** train labels used for class weights.
    """
    cfg = load_config("encoder", config_dir)
    _assert_no_smote(config_dir)
    set_global_seed(seed)

    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = model.to(dev)

    epochs = int(cfg.training.max_epochs if max_epochs is None else max_epochs)
    lr = float(cfg.optimizer.learning_rate if learning_rate is None else learning_rate)
    wd = float(cfg.optimizer.weight_decay)
    patience = int(cfg.training.early_stopping.patience)
    mode = str(cfg.training.early_stopping.mode)
    if str(cfg.training.early_stopping.monitor) != "val_macro_f1":
        raise ValueError(
            f"early_stopping.monitor must be 'val_macro_f1'; got "
            f"{cfg.training.early_stopping.monitor!r}"
        )

    class_weights = inverse_frequency_class_weights(train_labels, n_classes=2).to(dev)
    steps_per_epoch = max(1, len(train_loader))
    num_training_steps = steps_per_epoch * epochs

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = build_warmup_cosine_scheduler(
        optimizer,
        num_training_steps=num_training_steps,
        warmup_ratio=float(cfg.scheduler.warmup_ratio),
        decay_to=float(cfg.scheduler.decay_to),
    )
    stopper = EarlyStopping(patience=patience, mode=mode)  # type: ignore[arg-type]

    ckpt_dir = Path(checkpoint_dir) if checkpoint_dir is not None else None
    if ckpt_dir is not None:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path: Path | None = None
    best_f1 = -1.0
    best_epoch = -1
    history: list[EpochLog] = []

    for epoch in range(epochs):
        model.train()
        running = 0.0
        n_batches = 0
        for batch in train_loader:
            batch = _move_batch(batch, dev)
            optimizer.zero_grad(set_to_none=True)
            out = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )
            logits = out["logits"] if isinstance(out, dict) else out
            loss = F.cross_entropy(logits, batch["labels"].long(), weight=class_weights)
            loss.backward()
            optimizer.step()
            scheduler.step()
            running += float(loss.item())
            n_batches += 1

        train_loss = running / max(1, n_batches)
        val_f1 = evaluate_macro_f1(model, val_loader, device=dev)
        current_lr = float(optimizer.param_groups[0]["lr"])
        es = stopper.step(val_f1)

        if es.improved:
            best_f1 = float(val_f1)
            best_epoch = epoch
            if ckpt_dir is not None:
                best_path = ckpt_dir / "best.pt"
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "epoch": epoch,
                        "val_macro_f1": best_f1,
                        "seed": seed,
                    },
                    best_path,
                )

        log = EpochLog(
            epoch=epoch,
            train_loss=train_loss,
            val_macro_f1=float(val_f1),
            lr=current_lr,
            early_stop_improved=es.improved,
            early_stop_bad_epochs=es.bad_epochs,
            early_stop=es.should_stop,
        )
        history.append(log)
        logger.info(
            "epoch=%d loss=%.4f val_macro_f1=%.4f lr=%.2e improved=%s bad=%d stop=%s",
            epoch,
            train_loss,
            val_f1,
            current_lr,
            es.improved,
            es.bad_epochs,
            es.should_stop,
        )
        if es.should_stop:
            break

    # Restore best weights when a checkpoint was written
    if best_path is not None and best_path.is_file():
        state = torch.load(best_path, map_location=dev, weights_only=False)
        model.load_state_dict(state["model_state_dict"])

    return TrainResult(
        best_val_macro_f1=best_f1 if best_f1 >= 0 else float("nan"),
        best_epoch=best_epoch,
        checkpoint_path=best_path,
        history=history,
        class_weights=class_weights.detach().cpu().tolist(),
        stopped_early=bool(history and history[-1].early_stop),
    )


__all__ = [
    "EpochLog",
    "TrainResult",
    "evaluate_macro_f1",
    "train",
]
