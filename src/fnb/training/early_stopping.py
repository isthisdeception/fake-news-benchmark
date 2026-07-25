"""Early stopping on validation macro-F1 (protocol §7; patience 3)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Mode = Literal["max", "min"]


@dataclass
class EarlyStoppingState:
    """Snapshot after one ``step`` call."""

    improved: bool
    should_stop: bool
    best_score: float | None
    bad_epochs: int


class EarlyStopping:
    """Stop when ``monitor`` fails to improve for ``patience`` epochs.

    Frozen defaults (``configs/encoder.yaml``): monitor val macro-F1, patience 3,
    mode ``max``.
    """

    def __init__(
        self,
        *,
        patience: int = 3,
        mode: Mode = "max",
        min_delta: float = 0.0,
    ) -> None:
        if patience < 1:
            raise ValueError(f"patience must be >= 1; got {patience}")
        if mode not in ("max", "min"):
            raise ValueError(f"mode must be 'max' or 'min'; got {mode!r}")
        self.patience = int(patience)
        self.mode = mode
        self.min_delta = float(min_delta)
        self.best_score: float | None = None
        self.bad_epochs = 0
        self.should_stop = False

    def _is_improvement(self, score: float) -> bool:
        if self.best_score is None:
            return True
        if self.mode == "max":
            return score > self.best_score + self.min_delta
        return score < self.best_score - self.min_delta

    def step(self, score: float) -> EarlyStoppingState:
        """Update state with a new validation score."""
        improved = self._is_improvement(float(score))
        if improved:
            self.best_score = float(score)
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
            if self.bad_epochs >= self.patience:
                self.should_stop = True
        return EarlyStoppingState(
            improved=improved,
            should_stop=self.should_stop,
            best_score=self.best_score,
            bad_epochs=self.bad_epochs,
        )


__all__ = ["EarlyStopping", "EarlyStoppingState"]
