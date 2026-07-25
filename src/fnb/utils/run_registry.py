"""Run registry: run_id, environment capture, ``results/runs.csv`` + ``metadata.json``.

Implements the experiment-tracking + logging provenance required by the
reproducibility checklist (§9, §10): each run gets a unique ``run_id`` (with a
parent ``experiment_id`` tying seed-replicates together), a structured log file,
an append-only row in ``results/runs.csv``, and a per-artifact ``metadata.json``
capturing git commit, resolved config hashes, seed, library/CUDA versions,
hardware, dataset-version tag, timestamps, and wall-clock.

Everything degrades gracefully: git and heavy libraries are probed defensively
so this works in a lightweight authoring environment as well as on Kaggle.
"""

from __future__ import annotations

import platform
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .io import append_csv_row, ensure_dir, write_json
from .logging_utils import setup_run_logging

PROTOCOL_VERSION = "v1.0"

# Stable column order for results/runs.csv (append-only).
RUNS_CSV_FIELDS = [
    "run_id",
    "experiment_id",
    "experiment",
    "model",
    "dataset_version",
    "split",
    "seed",
    "protocol_version",
    "git_commit",
    "git_dirty",
    "config_hashes",
    "python",
    "platform",
    "torch",
    "cuda",
    "cudnn",
    "transformers",
    "scikit_learn",
    "numpy",
    "gpu",
    "start_time",
    "end_time",
    "wall_clock_s",
    "status",
    "notes",
]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _sanitize(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", str(text)).strip("-")


def make_run_id(
    experiment: str,
    model: str,
    dataset_version: str,
    split: str,
    seed: int,
    *,
    timestamp: str | None = None,
) -> str:
    """Build a unique run id from the identifying tuple + a UTC timestamp."""
    ts = timestamp or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    base = "_".join([experiment, model, dataset_version, split, f"seed{seed}"])
    return f"{_sanitize(base)}__{ts}"


def make_experiment_id(experiment: str, model: str, dataset_version: str, split: str) -> str:
    """Parent id tying seed-replicates of the same cell together."""
    return _sanitize("_".join([experiment, model, dataset_version, split]))


def _module_version(name: str) -> str:
    try:
        mod = __import__(name)
        return str(getattr(mod, "__version__", "unknown"))
    except Exception:  # pragma: no cover - defensive
        return "not-installed"


def find_git_root(start: str | Path | None = None) -> Path | None:
    """Walk upward from ``start`` (default: cwd) until a ``.git`` directory is found.

    On Kaggle, runs often pass ``base_dir=/kaggle/working`` while the clone lives
    at ``/kaggle/working/fake-news-benchmark``. Callers should also try the cwd
    (the clone) when resolving provenance.
    """
    cur = Path(start or Path.cwd()).resolve()
    for candidate in (cur, *cur.parents):
        if (candidate / ".git").exists():
            return candidate
    return None


def git_commit_sha(repo_dir: str | Path | None = None) -> str:
    root = find_git_root(repo_dir) if repo_dir else find_git_root()
    if root is None and repo_dir is not None:
        # Fall back: repo may be cwd (clone) even if base_dir is /kaggle/working.
        root = find_git_root(Path.cwd())
    if root is None:
        return "unknown"
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(root),
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except Exception:  # pragma: no cover - defensive
        return "unknown"


def git_is_dirty(repo_dir: str | Path | None = None) -> bool:
    root = find_git_root(repo_dir) if repo_dir else find_git_root()
    if root is None and repo_dir is not None:
        root = find_git_root(Path.cwd())
    if root is None:
        return False
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(root),
            capture_output=True,
            text=True,
            check=True,
        )
        return bool(out.stdout.strip())
    except Exception:  # pragma: no cover - defensive
        return False


def collect_environment() -> dict[str, Any]:
    """Capture python/library/CUDA/hardware versions for provenance."""
    env: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor(),
        "torch": _module_version("torch"),
        "transformers": _module_version("transformers"),
        "scikit_learn": _module_version("sklearn"),
        "numpy": _module_version("numpy"),
        "cuda": "unknown",
        "cudnn": "unknown",
        "gpu": "cpu",
    }
    try:  # pragma: no cover - torch present on Kaggle
        import torch

        env["cuda"] = str(torch.version.cuda)
        cudnn_ver = torch.backends.cudnn.version()
        env["cudnn"] = str(cudnn_ver) if cudnn_ver else "unknown"
        if torch.cuda.is_available():
            env["gpu"] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    return env


@dataclass
class RunContext:
    """In-memory handle for an active run; finalize with :func:`finish_run`."""

    run_id: str
    experiment_id: str
    experiment: str
    model: str
    dataset_version: str
    split: str
    seed: int
    protocol_version: str
    start_time: str
    _start_perf: float
    logger: Any
    results_dir: Path
    logs_dir: Path
    config_hashes: dict[str, str]
    environment: dict[str, Any]
    git_commit: str
    git_dirty: bool
    extra: dict[str, Any] = field(default_factory=dict)


def start_run(
    experiment: str,
    model: str,
    dataset_version: str,
    split: str,
    seed: int,
    *,
    config_names: list[str] | None = None,
    base_dir: str | Path | None = None,
    repo_dir: str | Path | None = None,
    results_dir: str | Path = "results",
    logs_dir: str | Path = "logs",
    protocol_version: str = PROTOCOL_VERSION,
    extra: dict[str, Any] | None = None,
) -> RunContext:
    """Begin a run: create the log file, capture environment/git/config hashes.

    ``base_dir`` is where OUTPUTS go (results/, logs/); on Kaggle set it to
    ``/kaggle/working``. ``repo_dir`` is where the CODE lives (the git repo) and
    defaults to the current working directory, so the git commit is captured
    even when outputs are written elsewhere.

    Returns a :class:`RunContext` to pass to :func:`finish_run`.
    """
    base = Path(base_dir) if base_dir else Path.cwd()
    results_path = base / results_dir
    logs_path = base / logs_dir
    ensure_dir(results_path)
    ensure_dir(logs_path)

    # Resolve the git root: explicit repo_dir, then cwd, then common Kaggle clone
    # path under base (/kaggle/working/fake-news-benchmark).
    if repo_dir is not None:
        repo_root = find_git_root(repo_dir)
    else:
        repo_root = (
            find_git_root(Path.cwd())
            or find_git_root(base)
            or find_git_root(base / "fake-news-benchmark")
        )

    run_id = make_run_id(experiment, model, dataset_version, split, seed)
    experiment_id = make_experiment_id(experiment, model, dataset_version, split)
    logger = setup_run_logging(run_id, logs_path)

    config_hashes: dict[str, str] = {}
    if config_names:
        from fnb.config import config_hash

        for name in config_names:
            try:
                config_hashes[name] = config_hash(name)
            except Exception:  # pragma: no cover - defensive
                config_hashes[name] = "unavailable"

    ctx = RunContext(
        run_id=run_id,
        experiment_id=experiment_id,
        experiment=experiment,
        model=model,
        dataset_version=dataset_version,
        split=split,
        seed=seed,
        protocol_version=protocol_version,
        start_time=_now_iso(),
        _start_perf=time.perf_counter(),
        logger=logger,
        results_dir=results_path,
        logs_dir=logs_path,
        config_hashes=config_hashes,
        environment=collect_environment(),
        git_commit=git_commit_sha(repo_root),
        git_dirty=git_is_dirty(repo_root),
        extra=dict(extra or {}),
    )
    logger.info(
        "run start run_id=%s experiment_id=%s seed=%s git=%s dirty=%s",
        ctx.run_id, ctx.experiment_id, ctx.seed, ctx.git_commit[:12], ctx.git_dirty,
    )
    return ctx


def write_metadata(directory: str | Path, metadata: dict[str, Any]) -> Path:
    """Write ``metadata.json`` into an artifact directory."""
    return write_json(metadata, Path(directory) / "metadata.json")


def finish_run(
    ctx: RunContext,
    *,
    status: str = "completed",
    metrics: dict[str, Any] | None = None,
    artifact_dir: str | Path | None = None,
    notes: str = "",
) -> dict[str, Any]:
    """Finalize a run: append a row to ``results/runs.csv`` and write ``metadata.json``.

    Returns the row dict that was written.
    """
    import json as _json

    wall = round(time.perf_counter() - ctx._start_perf, 3)
    end_time = _now_iso()
    env = ctx.environment

    row = {
        "run_id": ctx.run_id,
        "experiment_id": ctx.experiment_id,
        "experiment": ctx.experiment,
        "model": ctx.model,
        "dataset_version": ctx.dataset_version,
        "split": ctx.split,
        "seed": ctx.seed,
        "protocol_version": ctx.protocol_version,
        "git_commit": ctx.git_commit,
        "git_dirty": ctx.git_dirty,
        "config_hashes": _json.dumps(ctx.config_hashes, sort_keys=True),
        "python": env.get("python"),
        "platform": env.get("platform"),
        "torch": env.get("torch"),
        "cuda": env.get("cuda"),
        "cudnn": env.get("cudnn"),
        "transformers": env.get("transformers"),
        "scikit_learn": env.get("scikit_learn"),
        "numpy": env.get("numpy"),
        "gpu": env.get("gpu"),
        "start_time": ctx.start_time,
        "end_time": end_time,
        "wall_clock_s": wall,
        "status": status,
        "notes": notes,
    }
    append_csv_row(ctx.results_dir / "runs.csv", row, fieldnames=RUNS_CSV_FIELDS)

    metadata = {
        **{k: row[k] for k in RUNS_CSV_FIELDS},
        "config_hashes": ctx.config_hashes,
        "environment": ctx.environment,
        "metrics": metrics or {},
        "extra": ctx.extra,
    }
    write_metadata(artifact_dir or ctx.results_dir, metadata)

    ctx.logger.info(
        "run finish run_id=%s status=%s wall_clock_s=%s", ctx.run_id, status, wall
    )
    return row
