"""Typed schemas validating each ``configs/*.yaml`` frozen mirror.

Built on pydantic v2. Structured, numeric-bearing configs (encoder, dedup,
stats, llm_qlora, protocol) are validated strictly with ``extra="forbid"`` so a
typo or an unknown key is rejected rather than silently ignored. Descriptive,
free-form sections (per-dataset lineage, per-model grids, metric lists) use
open ``dict`` fields because their inner shape is intentionally flexible /
partly resolved later (e.g. Kaggle slugs filled in EXP-P0).

The public entry point is the ``CONFIG_MODELS`` registry mapping a config name
to its model class; ``fnb.config.load_config`` uses it.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, field_validator

PROTOCOL_VERSION = "v1.0"


class _Strict(BaseModel):
    """Base model that forbids unknown keys."""

    model_config = ConfigDict(extra="forbid")


# --------------------------------------------------------------------------- #
# Shared sub-models
# --------------------------------------------------------------------------- #
class _EarlyStopping(_Strict):
    monitor: str
    patience: int
    mode: str


# --------------------------------------------------------------------------- #
# protocol.yaml
# --------------------------------------------------------------------------- #
class _ComputeBudget(_Strict):
    total_gpu_hours: int
    rq_f_cap_gpu_hours: int
    accelerator: str
    cut_order_if_over_budget: list[str]


class _Seeds(_Strict):
    primary: list[int]
    secondary: list[int]
    reporting: str


class _Determinism(_Strict):
    use_deterministic_algorithms: bool
    cudnn_deterministic: bool
    cudnn_benchmark: bool
    cublas_workspace_config: str
    set_pythonhashseed: bool


class ProtocolConfig(_Strict):
    protocol_version: str
    status: str
    frozen_date: str
    primary_endpoint: str
    research_questions: dict[str, dict[str, str]]
    compute_budget: _ComputeBudget
    seeds: _Seeds
    determinism: _Determinism
    out_of_scope: list[str]

    @field_validator("protocol_version")
    @classmethod
    def _check_version(cls, v: str) -> str:
        if v != PROTOCOL_VERSION:
            raise ValueError(f"protocol_version must be {PROTOCOL_VERSION!r}, got {v!r}")
        return v


# --------------------------------------------------------------------------- #
# datasets.yaml  (per-dataset entries kept flexible: TBD placeholders)
# --------------------------------------------------------------------------- #
class DatasetsConfig(_Strict):
    label_space: dict[str, int]
    article_transfer_set: list[str]
    domain_shift_probe: list[str]
    short_statement_track: list[str]
    datasets: dict[str, dict[str, Any]]
    version_tags: dict[str, Any]
    llm_contamination: dict[str, Any]


# --------------------------------------------------------------------------- #
# preprocessing.yaml
# --------------------------------------------------------------------------- #
class _Tfidf(_Strict):
    ngram_range: list[int]
    max_features: int
    min_df: int
    fit_on: str


class _ClassicalTrack(_Strict):
    version_tag: str
    steps: list[str]
    tfidf: _Tfidf
    record_versions: list[str]


class _NeuralTrack(_Strict):
    version_tag: str
    steps: list[str]
    stopword_removal: bool
    lemmatization: bool


class _FullDocExposure(_Strict):
    method: str
    window_tokens: int
    stride_tokens: int
    window_logits_pooling: str
    title_only_max_tokens: int
    granularity_conditions: list[str]
    granularity_applicable_datasets: list[str]


class _ClassImbalance(_Strict):
    method: str
    smote: bool
    test_distribution: str


class PreprocessingConfig(_Strict):
    classical_track: _ClassicalTrack
    neural_track: _NeuralTrack
    full_document_exposure: _FullDocExposure
    class_imbalance: _ClassImbalance


# --------------------------------------------------------------------------- #
# dedup.yaml
# --------------------------------------------------------------------------- #
class _WithinDataset(_Strict):
    remove_exact_duplicates: bool
    near_duplicate_metric: str
    near_duplicate_threshold: float
    version_tag: str


class _AcrossDataset(_Strict):
    metric: str
    threshold: float
    side_removed: str
    report_to: str
    version_tag: str


class _DedupSensitivity(_Strict):
    thresholds: list[float]
    dataset: str
    version_tags: list[str]


class DedupConfig(_Strict):
    within_dataset: _WithinDataset
    across_dataset: _AcrossDataset
    sensitivity: _DedupSensitivity


# --------------------------------------------------------------------------- #
# classical_grids.yaml  (per-model estimator/grid kept flexible)
# --------------------------------------------------------------------------- #
class _Selection(_Strict):
    metric: str
    refit: bool


class ClassicalGridsConfig(_Strict):
    tfidf: _Tfidf
    selection: _Selection
    class_weight: str
    models: dict[str, dict[str, Any]]


# --------------------------------------------------------------------------- #
# encoder.yaml
# --------------------------------------------------------------------------- #
class _AdamW(_Strict):
    name: str
    weight_decay: float
    learning_rate: float


class _Scheduler(_Strict):
    name: str
    warmup_ratio: float
    decay_to: float


class _EncoderTraining(_Strict):
    batch_size: int
    max_epochs: int
    early_stopping: _EarlyStopping


class _SlidingWindow(_Strict):
    window_tokens: int
    stride_tokens: int
    window_logits_pooling: str


class _Sequence(_Strict):
    max_length: int
    sliding_window: _SlidingWindow
    title_only_max_length: int


class _EncoderLoss(_Strict):
    type: str
    class_weighting: str
    smote: bool


class EncoderConfig(_Strict):
    models: dict[str, str]
    optimizer: _AdamW
    scheduler: _Scheduler
    training: _EncoderTraining
    sequence: _Sequence
    loss: _EncoderLoss

    # Convenience accessors for the most-referenced frozen constants.
    @property
    def lr(self) -> float:
        return self.optimizer.learning_rate

    @property
    def batch_size(self) -> int:
        return self.training.batch_size

    @property
    def max_len(self) -> int:
        return self.sequence.max_length


# --------------------------------------------------------------------------- #
# llm_qlora.yaml
# --------------------------------------------------------------------------- #
class _Qlora(_Strict):
    r: int
    alpha: int
    dropout: float
    quant_type: str
    double_quant: bool


class _LLMTraining(_Strict):
    effective_batch_size: int
    max_epochs: int
    early_stopping: _EarlyStopping


class _Contamination(_Strict):
    probe: str
    report_to: str
    exclude_flagged_from_zero_shot_claims: bool


class _LLMBudget(_Strict):
    rq_f_cap_gpu_hours: int


class LLMQloraConfig(_Strict):
    models: dict[str, str]
    qlora: _Qlora
    optimizer: _AdamW
    scheduler: _Scheduler
    training: _LLMTraining
    inference_variants: list[str]
    contamination: _Contamination
    budget: _LLMBudget

    @property
    def lr(self) -> float:
        return self.optimizer.learning_rate


# --------------------------------------------------------------------------- #
# adversarial.yaml
# --------------------------------------------------------------------------- #
class _SemanticConstraint(_Strict):
    metric: str
    min_similarity: float
    enforced_via: str


class _ApplyTo(_Strict):
    dataset: str
    split: str
    models: str


class AdversarialConfig(_Strict):
    allowed_attacks: list[str]
    prohibited_attacks: list[str]
    semantic_constraint: _SemanticConstraint
    apply_to: _ApplyTo
    report: list[str]


# --------------------------------------------------------------------------- #
# metrics.yaml
# --------------------------------------------------------------------------- #
class _MetricsCore(_Strict):
    primary: str
    reported_alongside: list[str]
    aggregation: str


class _ECE(_Strict):
    bins: int
    probabilistic_models_only: bool
    probability_source: dict[str, str]
    exclude: list[str]


class _TransferMetrics(_Strict):
    delta_f1: str
    dispersion: str


class _EfficiencyMetrics(_Strict):
    measured_on: str
    metrics: list[str]


class MetricsConfig(_Strict):
    core: _MetricsCore
    ece: _ECE
    transfer: _TransferMetrics
    robust: list[str]
    efficiency: _EfficiencyMetrics


# --------------------------------------------------------------------------- #
# stats.yaml
# --------------------------------------------------------------------------- #
class _Bootstrap(_Strict):
    n_resamples: int
    stratified: bool
    ci: float
    alternative: str


class _McNemar(_Strict):
    paired_predictions: bool


class _EffectSize(_Strict):
    metric: str


class _MultipleComparisons(_Strict):
    method: str
    q: float
    applied_across: str


class _Ranking(_Strict):
    omnibus: str
    post_hoc: str
    scope: str


class StatsConfig(_Strict):
    confirmatory_family: list[str]
    bootstrap: _Bootstrap
    mcnemar: _McNemar
    effect_size: _EffectSize
    multiple_comparisons: _MultipleComparisons
    ranking: _Ranking


# --------------------------------------------------------------------------- #
# Registry: config name -> model class
# --------------------------------------------------------------------------- #
CONFIG_MODELS: dict[str, type[BaseModel]] = {
    "protocol": ProtocolConfig,
    "datasets": DatasetsConfig,
    "preprocessing": PreprocessingConfig,
    "dedup": DedupConfig,
    "classical_grids": ClassicalGridsConfig,
    "encoder": EncoderConfig,
    "llm_qlora": LLMQloraConfig,
    "adversarial": AdversarialConfig,
    "metrics": MetricsConfig,
    "stats": StatsConfig,
}

__all__ = ["CONFIG_MODELS", "PROTOCOL_VERSION", *[m.__name__ for m in CONFIG_MODELS.values()]]
