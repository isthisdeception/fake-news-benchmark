# Reproducibility Checklist — Fake News Detection Benchmark

Author role: Senior NLP researcher
Consistent with `docs/final_protocol.md` (v1.0, FROZEN) §9 (seeds) and §15 (reproducibility &
artifacts). Cross-refs: `docs/experiment_matrix.md`, `docs/evaluation_protocol.md`.

> Purpose: a single, auditable checklist that must be **fully green before any result is
> reported**. Every result row records `protocol_version=v1.0` plus the identifiers below so any
> run can be reconstructed bit-for-bit (within hardware nondeterminism bounds). Directly remedies
> the corpus reproducibility gaps: missing schedulers (P002–P005, P009, P010), missing batch
> sizes (P003–P005, P011, P012), and missing dataset sizes/hyperparameters (P011, P012).

Legend: ☐ = to verify per run/release · **[FROZEN]** = fixed by protocol, must not diverge.

---

## 1. Random Seeds

- ☐ **[FROZEN]** Primary endpoints (RQ-A/B/C) run all **5 seeds `{13, 21, 42, 87, 100}`** (§9).
- ☐ **[FROZEN]** Secondary endpoints (RQ-D/E/F) run **3 seeds `{13, 42, 100}`** (§9).
- ☐ Stochastic classical models (RandomForest, LightGBM) run all mandated seeds; deterministic
  ones (LogReg, LinearSVM) record the seed used for the split.
- ☐ Each seed sets **all** RNG sources in one place (`set_global_seed(seed)`):
  `random.seed`, `numpy.random.seed`, `torch.manual_seed`, `torch.cuda.manual_seed_all`,
  `PYTHONHASHSEED`, HF `transformers.set_seed`, and the DataLoader `generator` + `worker_init_fn`.
- ☐ **[FROZEN]** Deterministic algorithms enabled (§15): `torch.use_deterministic_algorithms(True)`,
  `torch.backends.cudnn.deterministic=True`, `cudnn.benchmark=False`,
  `CUBLAS_WORKSPACE_CONFIG=:4096:8`.
- ☐ Every result row stores its `seed`; aggregates are **mean ± std over seeds** (never best-run).
- ☐ Any irreducible nondeterminism (e.g., some CUDA kernels) is documented in `ENVIRONMENT.md`.

## 2. Library Versions

- ☐ **[FROZEN]** Pin and record exact versions in `ENVIRONMENT.md` (§15): Python, PyTorch,
  Transformers, scikit-learn, TextAttack, CUDA (+ cuDNN, driver).
- ☐ Also pin: `tokenizers`, `datasets`, `accelerate`, `peft`, `bitsandbytes` (QLoRA),
  `numpy`, `scipy`, `pandas`, `nltk`, `lightgbm`, `sentence-transformers`/USE (attack constraint),
  `statsmodels` (BH-FDR), `scikit-posthocs` (Nemenyi), `matplotlib`.
- ☐ Exact environment lock committed: `requirements.txt` (pinned `==`) **and** a frozen lock
  (`pip freeze > requirements.lock.txt` or `environment.yml`).
- ☐ NLTK data versions recorded (stopwords, WordNet) with download date.
- ☐ A `pip check` / import-smoke test passes in CI; container image digest (if used) recorded.

## 3. Hardware

- ☐ **[FROZEN]** All **efficiency/timing** metrics measured on a **single NVIDIA T4 (16 GB),
  Kaggle** (§14); runs on any other GPU are excluded from efficiency tables.
- ☐ Record per run: accelerator model, VRAM, CPU, RAM, and Kaggle/base image.
- ☐ Accuracy-quality runs may use other GPUs, but the GPU used is recorded per run and never mixed
  into efficiency tables.
- ☐ Mixed precision / TF32 settings recorded (affect numerics and timing).
- ☐ **[FROZEN]** Compute budget tracked against ≤ 120 T4 GPU-h total; RQ-F ≤ 20 GPU-h (§1);
  per-run wall-clock logged for the budget ledger.

## 4. Dataset Version

- ☐ **[FROZEN]** SHA-256 content hash per dataset committed to `data/SNAPSHOT_HASHES.txt`; FakeNewsNet
  cached at snapshot time and experiments use ONLY the cached snapshot (§4.6).
- ☐ Every result records its **dataset-version tag** (`vRAW / vBIN / vCLEAN-C / vCLEAN-N / vDEDUP /
  vXDEDUP(train→test)`; DS1 sensitivity `vDEDUP@{0.90,0.95,0.98}`) — see matrix §0.2.
- ☐ **[FROZEN]** Binary label mapping `{real=0, fake=1}` recorded; LIAR 6-way→binary map applied;
  ambiguous/dropped rows counted in `results/label_mapping_report.csv` (§4.3).
- ☐ **[FROZEN]** Dedup recorded: within-dataset exact + near-dup (cosine ≥ 0.95) counts in
  `results/dedup_counts.csv`; cross-dataset overlap (cosine ≥ 0.90) in
  `results/interdataset_overlap.csv` (§4.5).
- ☐ **[FROZEN]** Absolute sizes + class balances per dataset/split in `results/dataset_stats.csv`
  (§4.4; fixes P011's omission).
- ☐ Dataset licenses recorded; no redistribution beyond permitted terms (§16).

## 5. Train / Validation / Test Split

- ☐ **[FROZEN]** Stratified **70/15/15** train/val/test; split performed **before** any resampling
  or vectorizer fitting (§4.4, §6).
- ☐ Split **indices** (not re-derived splits) committed per dataset × split-type × seed:
  `data/splits/{DSx}_{S-RAND|S-SRC|S-TEMP}_{seed}.idx`.
- ☐ `S-SRC` (source-disjoint) provided for DS1 iff source recoverable; otherwise recorded as
  "not applicable" with a note (§11).
- ☐ `S-TEMP` (temporal, train-past/test-future) applied ONLY where reliable timestamps exist;
  else reported "not applicable" per dataset (§4.4).
- ☐ **[FROZEN]** Test set always uses the **natural class distribution**; no SMOTE, no rebalanced
  test set (§7).
- ☐ Cross-dataset transfer uses `vXDEDUP(train→test)` test indices (overlap removed) — recorded.

## 6. Tokenizer Version

- ☐ Record the exact tokenizer/vectorizer per model family:
  - Classical: **TF-IDF** (unigrams+bigrams, `max_features=20000`, `min_df=5`), **fit on TRAIN
    only**; the fitted vectorizer is serialized and committed (`artifacts/tfidf_{seed}.pkl`) (§6).
  - Encoders: model-native subword tokenizer; record HF model id + **tokenizer revision/commit
    hash** + `tokenizers` version; NFC + URL/HTML strip only, **no** stopword/lemmatization (§6).
  - LLMs: base-model tokenizer + chat template recorded; revision pinned.
- ☐ **[FROZEN]** Max length = 512 via sliding window (window 512, stride 128, logits mean-pooled);
  title-only = 64 (§6.1). Windowing params recorded.
- ☐ NLTK stopword list + WordNet lemmatizer versions recorded (classical track).

## 7. Model Checkpoints

- ☐ Every trained model saved with a reproducible name:
  `artifacts/{model}_{dataset-version}_{split}_{seed}/` + a `metadata.json`
  (protocol version, dataset hash, seed, git commit, library versions, train wall-clock).
- ☐ Record base-model **revision/commit hash** for every HF encoder and LLM (not just the name),
  so `bert-base-uncased`@rev is unambiguous.
- ☐ **[FROZEN]** LLMs: 4-bit **QLoRA** adapters saved (`r=64, alpha=16, dropout=0.2`, NF4,
  double-quant); base weights referenced by revision, adapters committed/released (§5, §7).
- ☐ `best_encoder` selection frozen to `results/best_encoder.txt` (consumed by RQ-C/D).
- ☐ Efficiency artifacts: `QUANT-BEST` (dynamic INT8) checkpoint + on-disk size recorded (§14).
- ☐ Checkpoint integrity hashes recorded; final checkpoint chosen by **early stopping on
  validation macro-F1** (patience 3) — the selected step is logged.

## 8. Configuration Files

- ☐ **[FROZEN]** All settings read from `configs/` **mirrors** of the protocol — code must not
  hard-code divergent values (§0). Required config files:
  - `configs/protocol.yaml` (version, seeds, thresholds, budget)
  - `configs/classical_grids.yaml` (per-model grids; §7)
  - `configs/encoder.yaml` (AdamW wd=0.01, LR 2e-5, batch 16, warmup 6%→cosine, epochs≤10,
    patience 3, max_len 512, window/stride, class-weighted CE, SMOTE=false)
  - `configs/llm_qlora.yaml` (LR 2e-4, r/alpha/dropout, NF4, grad-accum → effective batch 16;
    zero-shot + 5-shot prompt templates)
  - `configs/preprocessing.yaml` (classical vs neural tracks)
  - `configs/dedup.yaml` (0.95 within, 0.90 cross, DS1 sensitivity {0.90,0.95,0.98})
  - `configs/adversarial.yaml` (label-preserving attacks; negation/antonym prohibited; USE ≥ 0.90)
  - `configs/metrics.yaml`, `configs/stats.yaml` (bootstrap n=10,000; McNemar; Cliff's δ; BH-FDR q=0.05)
- ☐ Every config committed and referenced by path + git commit in each run's `metadata.json`.
- ☐ A config-hash is logged per run so config drift is detectable.
- ☐ CI check: no experiment reads a value that is absent from / disagrees with `configs/`.

## 9. Logging

- ☐ Structured per-run log capturing: git commit SHA (clean working tree enforced), full resolved
  config, all seeds, library + CUDA versions, hardware, dataset-version tag, split indices path,
  tokenizer revision, start/end timestamps, and wall-clock.
- ☐ Per-epoch training log: loss, validation macro-F1, LR (schedule trace), early-stopping
  decision and chosen step.
- ☐ Deterministic-mode confirmation logged (`use_deterministic_algorithms`, `CUBLAS_WORKSPACE_CONFIG`).
- ☐ Metrics written to the frozen result CSVs (evaluation protocol) with mean ± std and provenance
  columns; logs and CSVs are append-only and committed.
- ☐ Warnings/exceptions captured; any "not applicable" (temporal/source-disjoint) explicitly logged
  (§4.4, §11) — never silently skipped.
- ☐ Console + file logging with fixed log level; logs stored under `logs/{run_id}.log`.

## 10. Experiment Tracking

- ☐ Unique `run_id` per (experiment, model, dataset-version, split, seed); parent `experiment_id`
  ties seed-replicates together.
- ☐ Tracker records params (config), metrics (per evaluation protocol), artifacts (checkpoints,
  splits, perturbation sets), and environment; offline-capable (Kaggle-friendly), e.g. MLflow or
  a committed `results/runs.csv` registry — tool choice recorded in `ENVIRONMENT.md`.
- ☐ Each run linked to its git commit and config hash; re-runs with identical inputs must land in
  the same cell of the experiment matrix.
- ☐ Confirmatory-family runs tagged so `EXP-G4` can assemble them and apply **BH-FDR at q=0.05**
  reproducibly.
- ☐ Compute-budget ledger updated per run (GPU-h) against the ≤120 h / ≤20 h caps (§1).
- ☐ Final release bundle (§15): code, `configs/*.yaml`, split indices, perturbation sets, result
  CSVs, checkpoints/adapters, `ENVIRONMENT.md`, `data/SNAPSHOT_HASHES.txt`. **No proprietary /
  API-gated models** in reported results (§15).

---

## Pre-report Gate (all must be ☑ before any number is published)

1. ☐ Seeds, deterministic mode, and mean±std aggregation verified (§1).
2. ☐ `ENVIRONMENT.md` complete: pinned libs, CUDA, hardware (§2, §3).
3. ☐ Dataset hashes, version tags, dedup/overlap counts, and `dataset_stats.csv` present (§4).
4. ☐ Split indices committed; natural-distribution test sets confirmed (§5).
5. ☐ Tokenizer/vectorizer revisions + windowing params recorded (§6).
6. ☐ Checkpoints + base-model revisions + QLoRA adapters saved with metadata (§7).
7. ☐ All values sourced from `configs/`; config hashes logged (§8).
8. ☐ Logs complete and committed; no silent "N/A" (§9).
9. ☐ Every run in the tracker with git commit + config hash; budget ledger reconciled (§10).
10. ☐ `protocol_version=v1.0` stamped on every result row.
