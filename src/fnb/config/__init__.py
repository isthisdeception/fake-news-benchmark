"""Config loading, hashing, and schema validation for the frozen ``configs/*.yaml``.

Public API:
    load_config(name)      -> validated pydantic model (or raw dict if validate=False)
    load_config_raw(name)  -> plain dict, no validation
    config_hash(name)      -> stable SHA-256 of the config file
    config_path(name)      -> resolved path to the YAML file
    resolve_config_dir()   -> the configs directory in use
    available_configs()    -> names with a registered schema
"""

from __future__ import annotations

from .loader import (
    available_configs,
    config_hash,
    config_path,
    load_config,
    load_config_raw,
    resolve_config_dir,
)
from .schema import CONFIG_MODELS, PROTOCOL_VERSION

__all__ = [
    "load_config",
    "load_config_raw",
    "config_hash",
    "config_path",
    "resolve_config_dir",
    "available_configs",
    "CONFIG_MODELS",
    "PROTOCOL_VERSION",
]
