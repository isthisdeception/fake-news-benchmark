"""Global determinism & seeding (reproducibility §1; protocol determinism block).

``set_global_seed(seed)`` seeds every RNG source used in the project and enables
deterministic algorithms, matching the frozen determinism policy in
``configs/protocol.yaml``:

    * ``PYTHONHASHSEED``               = seed
    * ``CUBLAS_WORKSPACE_CONFIG``      = ":4096:8"   (set BEFORE CUDA init)
    * ``random`` / ``numpy`` / ``torch`` / ``torch.cuda`` seeded
    * ``transformers.set_seed`` (if available)
    * ``torch.use_deterministic_algorithms(True)``
    * ``cudnn.deterministic = True`` / ``cudnn.benchmark = False``

It returns a :class:`SeedBundle` carrying the seed, a torch ``Generator`` (for
DataLoaders), and a ``worker_init_fn`` so DataLoader workers are seeded too.

``torch`` and ``transformers`` are imported lazily and treated as optional, so
this module (and its non-torch behaviour) works even where they are absent
(e.g. a lightweight authoring environment); on Kaggle both are present.
"""

from __future__ import annotations

import logging
import os
import random
from dataclasses import dataclass
from typing import Any, Callable

logger = logging.getLogger("fnb.seeding")

# Frozen determinism constants (mirror of configs/protocol.yaml -> determinism).
CUBLAS_WORKSPACE_CONFIG = ":4096:8"
_UINT32 = 2**32


@dataclass
class SeedBundle:
    """Result of :func:`set_global_seed`.

    Attributes:
        seed: the base seed applied.
        generator: a torch ``Generator`` seeded with ``seed`` (or ``None`` if
            torch is unavailable). Pass to ``DataLoader(generator=...)``.
        worker_init_fn: pass to ``DataLoader(worker_init_fn=...)`` to seed each
            worker deterministically.
    """

    seed: int
    generator: Any | None
    worker_init_fn: Callable[[int], None]


def seed_worker(worker_id: int) -> None:
    """DataLoader ``worker_init_fn``: seed numpy/random per worker deterministically.

    Uses torch's per-worker initial seed (derived from the base generator seed),
    so each worker gets a distinct-but-reproducible stream (PyTorch-recommended).
    """
    try:
        import torch

        worker_seed = torch.initial_seed() % _UINT32
    except ImportError:  # pragma: no cover - torch present on Kaggle
        worker_seed = worker_id

    import numpy as np

    np.random.seed(worker_seed)
    random.seed(worker_seed)


def set_global_seed(seed: int, *, deterministic: bool = True) -> SeedBundle:
    """Seed all RNG sources and enable deterministic algorithms.

    Args:
        seed: the global seed (protocol seeds are {13,21,42,87,100} / {13,42,100}).
        deterministic: if True (default), also set the deterministic toggles and
            ``CUBLAS_WORKSPACE_CONFIG``.

    Returns:
        A :class:`SeedBundle` with the seed, a torch generator, and a worker init fn.
    """
    if not isinstance(seed, int):
        raise TypeError(f"seed must be int, got {type(seed).__name__}")

    # --- Environment first (CUBLAS must be set before CUDA context init) ------
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = CUBLAS_WORKSPACE_CONFIG

    # --- stdlib + numpy -------------------------------------------------------
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:  # pragma: no cover
        logger.warning("numpy not available; skipping numpy seeding")

    # --- torch (optional) -----------------------------------------------------
    generator: Any | None = None
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        generator = torch.Generator()
        generator.manual_seed(seed)
    except ImportError:  # pragma: no cover - torch present on Kaggle
        logger.warning("torch not available; skipping torch seeding")

    # --- transformers (optional) ---------------------------------------------
    try:
        import transformers

        transformers.set_seed(seed)
    except ImportError:  # pragma: no cover - transformers present on Kaggle
        pass

    logger.info("set_global_seed seed=%d deterministic=%s", seed, deterministic)
    return SeedBundle(seed=seed, generator=generator, worker_init_fn=seed_worker)
