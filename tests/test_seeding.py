"""Tests for fnb.utils.seeding: determinism across repeated seeded runs."""

from __future__ import annotations

import os
import random

import numpy as np
import pytest

from fnb.utils.seeding import CUBLAS_WORKSPACE_CONFIG, SeedBundle, set_global_seed


def test_returns_seedbundle_with_worker_init_fn():
    bundle = set_global_seed(42)
    assert isinstance(bundle, SeedBundle)
    assert bundle.seed == 42
    assert callable(bundle.worker_init_fn)


def test_env_vars_set():
    set_global_seed(87)
    assert os.environ["PYTHONHASHSEED"] == "87"
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == CUBLAS_WORKSPACE_CONFIG == ":4096:8"


def test_numpy_reproducible_same_seed():
    set_global_seed(13)
    a = np.random.rand(5)
    set_global_seed(13)
    b = np.random.rand(5)
    assert np.array_equal(a, b)


def test_numpy_differs_for_different_seed():
    set_global_seed(13)
    a = np.random.rand(5)
    set_global_seed(100)
    b = np.random.rand(5)
    assert not np.array_equal(a, b)


def test_python_random_reproducible_same_seed():
    set_global_seed(21)
    a = [random.random() for _ in range(5)]
    set_global_seed(21)
    b = [random.random() for _ in range(5)]
    assert a == b


def test_non_int_seed_raises():
    with pytest.raises(TypeError):
        set_global_seed("42")  # type: ignore[arg-type]


def test_torch_reproducible_and_flags():
    torch = pytest.importorskip("torch")

    bundle = set_global_seed(42)
    a = torch.randn(4)
    set_global_seed(42)
    b = torch.randn(4)
    assert torch.equal(a, b)

    # deterministic toggles
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False

    # generator seeded and reproducible
    assert bundle.generator is not None
    g1 = torch.randn(3, generator=bundle.generator.manual_seed(42))
    g2 = torch.randn(3, generator=bundle.generator.manual_seed(42))
    assert torch.equal(g1, g2)


def test_torch_differs_for_different_seed():
    torch = pytest.importorskip("torch")
    set_global_seed(13)
    a = torch.randn(4)
    set_global_seed(100)
    b = torch.randn(4)
    assert not torch.equal(a, b)
