"""Tests for fnb.config: loading, hashing, schema validation."""

from __future__ import annotations

import copy

import pytest
from pydantic import ValidationError

from fnb.config import (
    PROTOCOL_VERSION,
    available_configs,
    config_hash,
    load_config,
    load_config_raw,
)
from fnb.config.schema import EncoderConfig

ALL_CONFIGS = [
    "protocol",
    "datasets",
    "preprocessing",
    "dedup",
    "classical_grids",
    "encoder",
    "llm_qlora",
    "adversarial",
    "metrics",
    "stats",
]


def test_registry_covers_all_ten_configs():
    assert available_configs() == sorted(ALL_CONFIGS)


@pytest.mark.parametrize("name", ALL_CONFIGS)
def test_every_config_loads_and_validates(name):
    model = load_config(name)
    assert model is not None


def test_protocol_version_is_frozen_v1():
    proto = load_config("protocol")
    assert proto.protocol_version == PROTOCOL_VERSION == "v1.0"


def test_protocol_seeds_match_protocol():
    proto = load_config("protocol")
    assert proto.seeds.primary == [13, 21, 42, 87, 100]
    assert proto.seeds.secondary == [13, 42, 100]
    assert proto.compute_budget.total_gpu_hours == 120
    assert proto.compute_budget.rq_f_cap_gpu_hours == 20


def test_encoder_frozen_values_and_accessors():
    enc = load_config("encoder")
    assert isinstance(enc, EncoderConfig)
    # convenience accessors
    assert enc.lr == 2e-5
    assert enc.batch_size == 16
    assert enc.max_len == 512
    # nested frozen values
    assert enc.optimizer.weight_decay == 0.01
    assert enc.scheduler.warmup_ratio == 0.06
    assert enc.training.max_epochs == 10
    assert enc.training.early_stopping.patience == 3
    assert enc.sequence.sliding_window.window_tokens == 512
    assert enc.sequence.sliding_window.stride_tokens == 128
    assert enc.sequence.title_only_max_length == 64
    assert enc.loss.smote is False


def test_dedup_stats_llm_frozen_values():
    dedup = load_config("dedup")
    assert dedup.within_dataset.near_duplicate_threshold == 0.95
    assert dedup.across_dataset.threshold == 0.90
    assert dedup.sensitivity.thresholds == [0.90, 0.95, 0.98]

    stats = load_config("stats")
    assert stats.bootstrap.n_resamples == 10000
    assert stats.multiple_comparisons.q == 0.05

    llm = load_config("llm_qlora")
    assert (llm.qlora.r, llm.qlora.alpha, llm.qlora.dropout) == (64, 16, 0.2)
    assert llm.qlora.quant_type == "nf4"
    assert llm.qlora.double_quant is True
    assert llm.lr == 2e-4

    adv = load_config("adversarial")
    assert adv.semantic_constraint.min_similarity == 0.90


def test_config_hash_is_stable_and_hex():
    h1 = config_hash("encoder")
    h2 = config_hash("encoder")
    assert h1 == h2
    assert len(h1) == 64
    int(h1, 16)  # valid hex


def test_missing_config_raises_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_config("does_not_exist")


def test_missing_required_field_raises():
    data = load_config_raw("encoder")
    mutated = copy.deepcopy(data)
    del mutated["optimizer"]["learning_rate"]
    with pytest.raises(ValidationError):
        EncoderConfig.model_validate(mutated)


def test_unknown_extra_key_is_rejected():
    data = load_config_raw("encoder")
    mutated = copy.deepcopy(data)
    mutated["totally_unexpected_key"] = 123
    with pytest.raises(ValidationError):
        EncoderConfig.model_validate(mutated)


def test_wrong_protocol_version_rejected():
    from fnb.config.schema import ProtocolConfig

    data = load_config_raw("protocol")
    mutated = copy.deepcopy(data)
    mutated["protocol_version"] = "v9.9"
    with pytest.raises(ValidationError):
        ProtocolConfig.model_validate(mutated)
