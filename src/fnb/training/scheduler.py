"""LR schedule: linear warmup → cosine decay to 0 (protocol §7 / HP-ENCODER)."""

from __future__ import annotations

import math
from typing import Any

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def warmup_cosine_lambda(
    current_step: int,
    *,
    num_warmup_steps: int,
    num_training_steps: int,
    decay_to: float = 0.0,
) -> float:
    """Multiplier for base LR at ``current_step`` (0-indexed).

    * Linear warmup from 0 → 1 over ``num_warmup_steps``.
    * Cosine decay from 1 → ``decay_to`` over the remaining steps.
    """
    if num_training_steps <= 0:
        raise ValueError("num_training_steps must be positive")
    if current_step < 0:
        raise ValueError("current_step must be >= 0")

    if num_warmup_steps > 0 and current_step < num_warmup_steps:
        return float(current_step + 1) / float(max(1, num_warmup_steps))

    # After the final training step index, LR is exactly decay_to (§7 → 0).
    if current_step >= num_training_steps - 1:
        return float(decay_to)

    # Progress in [0, 1) over the cosine phase.
    cosine_steps = max(1, num_training_steps - num_warmup_steps - 1)
    progress = float(current_step - num_warmup_steps) / float(cosine_steps)
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return float(decay_to + (1.0 - decay_to) * cosine)


def build_warmup_cosine_scheduler(
    optimizer: Optimizer,
    *,
    num_training_steps: int,
    warmup_ratio: float = 0.06,
    decay_to: float = 0.0,
) -> LambdaLR:
    """Build a ``LambdaLR`` matching ``configs/encoder.yaml`` scheduler block."""
    if not (0.0 <= warmup_ratio < 1.0):
        raise ValueError(f"warmup_ratio must be in [0, 1); got {warmup_ratio}")
    num_warmup_steps = int(num_training_steps * warmup_ratio)

    def lr_lambda(step: int) -> float:
        return warmup_cosine_lambda(
            step,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            decay_to=decay_to,
        )

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def scheduler_hparams_from_config(cfg: Any) -> dict[str, float]:
    """Extract scheduler knobs from a loaded ``EncoderConfig``."""
    return {
        "warmup_ratio": float(cfg.scheduler.warmup_ratio),
        "decay_to": float(cfg.scheduler.decay_to),
    }


__all__ = [
    "build_warmup_cosine_scheduler",
    "scheduler_hparams_from_config",
    "warmup_cosine_lambda",
]
