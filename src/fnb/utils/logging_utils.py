"""Structured per-run logging (reproducibility §9).

``setup_run_logging(run_id)`` configures a console handler plus a file handler
at ``logs/{run_id}.log`` with a fixed format and level, and returns a logger.
Handlers are idempotent per (logger, file) so re-calling within a session does
not duplicate log lines.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%dT%H:%M:%S"
DEFAULT_LEVEL = logging.INFO


def _has_file_handler(logger: logging.Logger, path: Path) -> bool:
    target = str(path.resolve())
    for h in logger.handlers:
        if isinstance(h, logging.FileHandler) and getattr(h, "_fnb_path", None) == target:
            return True
    return False


def _has_stream_handler(logger: logging.Logger) -> bool:
    return any(
        isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
        for h in logger.handlers
    )


def setup_run_logging(
    run_id: str,
    logs_dir: str | Path = "logs",
    *,
    level: int = DEFAULT_LEVEL,
    logger_name: str = "fnb",
    console: bool = True,
) -> logging.Logger:
    """Configure and return a logger writing to ``logs/{run_id}.log`` (+ console).

    Args:
        run_id: unique run identifier; names the log file.
        logs_dir: directory for log files (created if missing).
        level: logging level (default INFO).
        logger_name: base logger name.
        console: also emit to stderr if True.

    Returns:
        The configured ``logging.Logger``.
    """
    logs_path = Path(logs_dir)
    logs_path.mkdir(parents=True, exist_ok=True)
    log_file = logs_path / f"{run_id}.log"

    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    if not _has_file_handler(logger, log_file):
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(formatter)
        fh._fnb_path = str(log_file.resolve())  # type: ignore[attr-defined]
        logger.addHandler(fh)

    if console and not _has_stream_handler(logger):
        ch = logging.StreamHandler(sys.stderr)
        ch.setLevel(level)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    logger.propagate = False
    return logger


def get_logger(name: str = "fnb") -> logging.Logger:
    """Return a named logger (child of the ``fnb`` logger)."""
    return logging.getLogger(name)
