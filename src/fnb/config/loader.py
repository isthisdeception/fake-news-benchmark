"""Load, hash, and validate the frozen ``configs/*.yaml`` mirrors.

``load_config(name)`` reads a YAML file, logs a stable SHA-256 of its bytes
(reproducibility §8), validates it against the typed schema in
:mod:`fnb.config.schema`, and returns the validated pydantic model. A missing
or malformed field raises a clear ``pydantic.ValidationError``; unknown keys in
a structured config are rejected.

Config-directory resolution order:
    1. explicit ``config_dir`` argument,
    2. ``FNB_CONFIG_DIR`` environment variable,
    3. ``./configs`` relative to the current working directory (the Kaggle
       repo-root case),
    4. ``configs/`` next to the repository root inferred from this file.
"""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel

from .schema import CONFIG_MODELS

logger = logging.getLogger("fnb.config")

# repo root = .../FakeNewsBenchmark  (this file: src/fnb/config/loader.py)
_REPO_ROOT = Path(__file__).resolve().parents[3]


def resolve_config_dir(config_dir: str | os.PathLike[str] | None = None) -> Path:
    """Return the directory holding the ``*.yaml`` config mirrors."""
    if config_dir is not None:
        return Path(config_dir)
    env = os.environ.get("FNB_CONFIG_DIR")
    if env:
        return Path(env)
    cwd_configs = Path.cwd() / "configs"
    if cwd_configs.is_dir():
        return cwd_configs
    return _REPO_ROOT / "configs"


def config_path(name: str, config_dir: str | os.PathLike[str] | None = None) -> Path:
    """Return the path to ``<config_dir>/<name>.yaml``."""
    return resolve_config_dir(config_dir) / f"{name}.yaml"


def config_hash(name: str, config_dir: str | os.PathLike[str] | None = None) -> str:
    """Return the SHA-256 hex digest of a config file's raw bytes (stable)."""
    path = config_path(name, config_dir)
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def available_configs() -> list[str]:
    """Return the sorted names of configs with a registered schema."""
    return sorted(CONFIG_MODELS)


def load_config_raw(
    name: str, config_dir: str | os.PathLike[str] | None = None
) -> dict[str, Any]:
    """Load a config YAML as a plain dict (no schema validation)."""
    path = config_path(name, config_dir)
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Config {name!r} did not parse to a mapping (got {type(data).__name__}).")
    return data


def load_config(
    name: str,
    config_dir: str | os.PathLike[str] | None = None,
    *,
    validate: bool = True,
) -> BaseModel | dict[str, Any]:
    """Load, hash-log, and validate a named config.

    Args:
        name: config stem, e.g. ``"encoder"`` (no ``.yaml``).
        config_dir: optional override for the configs directory.
        validate: if ``False``, return the raw dict instead of a typed model.

    Returns:
        The validated pydantic model (or the raw dict when ``validate=False``).

    Raises:
        FileNotFoundError: the config file does not exist.
        KeyError: no schema is registered for ``name``.
        pydantic.ValidationError: the config fails schema validation.
    """
    path = config_path(name, config_dir)
    if not path.is_file():
        raise FileNotFoundError(
            f"Config file not found: {path}. Known configs: {available_configs()}"
        )

    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    logger.info("loaded config name=%s sha256=%s path=%s", name, digest, path)

    data = yaml.safe_load(raw.decode("utf-8"))
    if not validate:
        return data

    model_cls = CONFIG_MODELS.get(name)
    if model_cls is None:
        raise KeyError(
            f"No schema registered for config {name!r}. Known: {available_configs()}"
        )
    return model_cls.model_validate(data)
