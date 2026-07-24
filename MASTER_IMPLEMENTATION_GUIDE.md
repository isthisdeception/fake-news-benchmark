# MASTER IMPLEMENTATION GUIDE — Fake News Detection Benchmark

> STATUS: AUTHORITATIVE IMPLEMENTATION MANUAL. This document is the project's
> constitution for implementation. It is subordinate to and must never contradict
> `docs/final_protocol.md` (v1.0, FROZEN). It adds NO new research decision, phase,
> dataset, model, metric, or hyperparameter. It only specifies HOW to build the code
> that executes the already-frozen protocol and experiment matrix.
>
> Governing sources (do not re-open):
> - `docs/final_protocol.md` (v1.0, FROZEN) — the immutable research protocol.
> - `docs/experiment_matrix.md` — EXP-P0 … EXP-G4, the single source of truth for experiments.
> - `docs/evaluation_protocol.md` — how results are measured/tested (R1–R10).
> - `docs/reproducibility_checklist.md` — the pre-report gate.
> - `literature/dataset_selection_report.md` — dataset justification.
>
> Rule of use: complete the milestones in order. There is exactly ONE next step at any
> time (the lowest-numbered incomplete step). Do not start a step until every prior step's
> "Definition of Done" is satisfied. Do not redesign; if a genuine protocol conflict is
> found, STOP and request a protocol version bump per `final_protocol.md` §0 — do not
> silently deviate.

---

## Execution Model (Cursor authoring + Kaggle execution) — READ FIRST

This project uses a **two-environment split**. It governs every "run" and "verify" instruction
in this manual.

- **Cursor (authoring only):** used to GENERATE and edit code, configs, and docs. Code is NOT
  executed for real here. After each step's prompt produces code, commit it and push to a Git
  remote (e.g., GitHub).
- **Kaggle (execution only):** the single **NVIDIA T4 (16 GB)** notebook environment is where ALL
  code actually runs — data pipeline, training, evaluation, tests, and figure generation. This is
  consistent with `final_protocol.md` §14 (efficiency/timing pinned to a fixed Kaggle T4).
- **You (the loop):** run cells on Kaggle, then copy the produced outputs (CSV contents, logs,
  test results, verification reports) back into this Cursor chat so the next task can proceed.

**Concrete Kaggle conventions used throughout this guide:**
- **Getting the code onto Kaggle:** in the notebook's first cell, `git clone` the repo (or add it
  as a Kaggle *Utility Script*/*Dataset*), then `pip install -e .`. Enable "Internet" in notebook
  settings for `pip`/Hugging Face downloads.
- **Datasets:** DS1–DS5 are attached via **Add Data → Kaggle Datasets** and read from
  `/kaggle/input/<dataset-slug>/` (READ-ONLY). Never download to a local machine. Snapshot hashing
  (EXP-P0) is computed inside Kaggle over the attached `/kaggle/input/` files, and the recorded
  hash pins the exact Kaggle dataset version used.
- **Outputs:** all artifacts are written under `/kaggle/working/` (writable). At session end, save
  `/kaggle/working/` as a **versioned notebook output** or a new **Kaggle Dataset** so results,
  checkpoints, and adapters persist (Kaggle sessions are ephemeral). The next experiment attaches
  the previous output as an input dataset to resume.
- **Session/quota limits:** Kaggle sessions are time-boxed (≈9–12 h) with a weekly GPU quota. The
  ≤120 T4 GPU-h budget (protocol §1) is therefore spread across **multiple Kaggle sessions**; each
  experiment (EXP-*) is designed to run within one session and to persist its outputs so the run is
  resumable. Long sweeps (e.g., EXP-1's 5 encoders × 5 seeds) are split across sessions per model/seed.
- **Wherever this manual says "run … locally" or `pytest …` / `python scripts/…`,** it means "run
  that command in a Kaggle notebook cell and paste the output back into Cursor." Unit tests
  (`pytest`) run in a lightweight CPU Kaggle session; GPU experiments run in a T4 session.
- **`notebooks/kaggle_runner.ipynb`** is the thin wrapper that (1) clones + installs the package,
  (2) attaches input datasets, (3) dispatches a chosen stage/experiment via `scripts/`, and
  (4) saves `/kaggle/working/` outputs. Each milestone extends it with the cells for that milestone.

---

## Table of Contents

1. Project Goal
2. Final Project Structure
3. Implementation Roadmap (Milestones)
4. Detailed Step-by-Step Guide
5. Cursor Prompts (one per step)
6. Validation Checklist (per milestone)
7. Git Workflow (per milestone)
8. Final Verification
9. Experiment Phase
10. Paper Phase

---

# 1. Project Goal

Build a **reproducible, leakage-audited benchmarking framework** in Python + PyTorch +
Hugging Face Transformers (with a **custom training loop, NOT the HF `Trainer`**) that
executes every experiment in `docs/experiment_matrix.md` on the frozen dataset set
(WELFake primary, plus ISOT, FakeNewsNet, COVID-19, LIAR) and produces the exact result
artifacts listed in the matrix's Deliverables Index (§12).

The implementation objective, precisely:

- **Primary endpoint:** a reproducible, leakage-audited measurement of cross-dataset
  generalization in fake news detection (protocol §1).
- **Deliver working code** for: data snapshotting/hashing, binarization, dual-track
  preprocessing, within- and cross-dataset deduplication, seed-controlled splits, dataset
  statistics, classical + encoder + decoder-LLM + source-only models, a custom PyTorch
  training loop with sliding-window full-document exposure, the frozen metric suite, the
  frozen statistical stack (paired bootstrap, McNemar, Cliff's delta, BH-FDR,
  Friedman/Nemenyi), adversarial evaluation, efficiency profiling, calibration, error
  analysis, and interpretability.
- **Every result row** must record `protocol_version=v1.0`, dataset-version tag, split
  type, model ID, and seed, and every reported number must be **mean ± std across the
  mandated seeds** (never best-run-only).
- **Every value** that the protocol freezes must be read from `configs/` mirrors, never
  hard-coded (protocol §0; reproducibility §8).

Success = the framework can regenerate all CSVs in `docs/experiment_matrix.md` §12 within
the ≤120 T4 GPU-hour budget, passing the reproducibility pre-report gate.

---

# 2. Final Project Structure

```
FakeNewsBenchmark/
├── MASTER_IMPLEMENTATION_GUIDE.md   # this manual (implementation constitution)
├── README.md                        # setup, quickstart, how to run each milestone
├── ENVIRONMENT.md                   # pinned Python/PyTorch/Transformers/CUDA + hardware (repro §2,§3)
├── requirements.txt                 # pinned '==' dependencies
├── requirements.lock.txt            # full `pip freeze` lock
├── pyproject.toml                   # package metadata + tool config (ruff/black/pytest)
│
├── configs/                         # FROZEN protocol mirrors — code reads values ONLY from here (§0)
│   ├── protocol.yaml                # version, seeds, thresholds, compute budget
│   ├── datasets.yaml                # DS1–DS5 ids, roles, source paths, label columns
│   ├── preprocessing.yaml           # classical vs neural tracks (§6)
│   ├── dedup.yaml                   # within 0.95, cross 0.90, DS1 sensitivity {0.90,0.95,0.98}
│   ├── classical_grids.yaml         # per-model grids (§7)
│   ├── encoder.yaml                 # AdamW wd, LR 2e-5, batch 16, warmup 6%→cosine, epochs≤10, patience 3, max_len 512, window/stride, class-weighted CE, SMOTE=false
│   ├── llm_qlora.yaml               # LR 2e-4, r/alpha/dropout, NF4, grad-accum→eff batch 16, zero/5-shot templates
│   ├── adversarial.yaml             # label-preserving attacks; negation/antonym prohibited; USE ≥ 0.90
│   ├── metrics.yaml                 # METRICS-CORE + ECE bins
│   └── stats.yaml                   # bootstrap n=10000, McNemar, Cliff's δ, BH-FDR q=0.05
│
├── data/
│   ├── raw/                         # committed raw snapshots (DS1–DS5), incl. cached FakeNewsNet
│   ├── processed/                   # {DSx}_{vBIN,vCLEAN-C,vCLEAN-N,vDEDUP,...}.parquet
│   ├── splits/                      # {DSx}_{S-RAND|S-SRC|S-TEMP}_{seed}.idx (indices only)
│   └── SNAPSHOT_HASHES.txt          # SHA-256 per dataset (§4.6)
│
├── src/fnb/                         # the installable package (import as `fnb`)
│   ├── __init__.py
│   ├── config/                      # config loading + schema validation + typed constants
│   │   ├── __init__.py
│   │   ├── loader.py                # load_config(name) with hash logging
│   │   └── schema.py                # dataclasses / pydantic models validating each YAML
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── seeding.py               # set_global_seed(), determinism toggles
│   │   ├── logging_utils.py         # structured per-run logging
│   │   ├── hashing.py               # SHA-256 of files/dataframes/configs
│   │   ├── run_registry.py          # run_id, metadata.json, results/runs.csv registry
│   │   └── io.py                    # parquet/csv/idx read-write helpers
│   ├── data/
│   │   ├── __init__.py
│   │   ├── acquire.py               # snapshot + hash datasets (EXP-P0)
│   │   ├── binarize.py              # frozen label mapping + report (EXP-P1a)
│   │   ├── preprocess.py            # classical + neural tracks (EXP-P1b)
│   │   ├── dedup.py                 # within-dataset exact+near dup (EXP-P2)
│   │   ├── splits.py                # S-RAND/S-SRC/S-TEMP (EXP-P3)
│   │   ├── stats.py                 # dataset_stats.csv (EXP-P4)
│   │   └── xdedup.py                # cross-dataset overlap removal (EXP-P5)
│   ├── models/
│   │   ├── __init__.py
│   │   ├── classical.py             # TF-IDF + LR/SVM/RF/LGBM + calibration
│   │   ├── encoder.py               # HF encoder wrapper + sliding-window head
│   │   ├── source_only.py           # SRCONLY shortcut baseline (RQ-B)
│   │   └── llm_qlora.py             # 4-bit QLoRA fine-tune + zero/5-shot inference
│   ├── training/
│   │   ├── __init__.py
│   │   ├── datasets.py              # torch Dataset + sliding-window tokenization
│   │   ├── loop.py                  # CUSTOM training loop (no HF Trainer)
│   │   ├── scheduler.py             # linear warmup 6% → cosine decay
│   │   └── early_stopping.py        # patience-3 on val macro-F1
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py               # METRICS-CORE, ECE, TRANSFER, ROBUST
│   │   └── stats.py                 # bootstrap, McNemar, Cliff's δ, BH-FDR, Friedman/Nemenyi
│   ├── adversarial/
│   │   ├── __init__.py
│   │   └── attacks.py               # TextAttack recipes + USE≥0.90 constraint
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── calibration.py           # ECE + reliability diagrams (EXP-G1)
│   │   ├── error_analysis.py        # confusion, taxonomy, length buckets (EXP-G2)
│   │   ├── interpretability.py      # Integrated Gradients + occlusion faithfulness (EXP-G3)
│   │   └── figures.py               # Pareto plots, tables, paper figures
│   └── experiments/
│       ├── __init__.py
│       ├── exp1_indist.py           # EXP-1 + best_encoder
│       ├── exp_p6_dedup_sens.py     # EXP-P6
│       ├── expA_transfer.py         # EXP-A1/A2/A3
│       ├── expB_leakage.py          # EXP-B1/B2/B3
│       ├── expC_adversarial.py      # EXP-C1
│       ├── expD_efficiency.py       # EXP-D1/D2
│       ├── expE_granularity.py      # EXP-E1
│       ├── expF_llm.py              # EXP-F0/F1/F2
│       └── expG_confirmatory.py     # EXP-G4 aggregation
│
├── scripts/                         # thin CLI entry points (argparse) calling src/fnb
│   ├── run_data_pipeline.py         # P0→P5 end-to-end
│   ├── run_experiment.py            # dispatch any EXP-* by id
│   └── build_paper_outputs.py       # tables + figures for the paper
│
├── notebooks/                       # Kaggle T4 execution notebooks (thin wrappers over scripts)
│   └── kaggle_runner.ipynb
│
├── tests/                           # unit tests (pytest), mirrors src/fnb
│   ├── test_config.py
│   ├── test_seeding.py
│   ├── test_preprocess.py
│   ├── test_dedup.py
│   ├── test_splits.py
│   ├── test_metrics.py
│   ├── test_stats.py
│   ├── test_classical.py
│   ├── test_encoder.py
│   └── test_training_loop.py
│
├── artifacts/                       # checkpoints, tfidf_{seed}.pkl, QLoRA adapters, metadata.json
├── results/                         # all frozen result CSVs (matrix §12) + plots
├── logs/                            # {run_id}.log structured logs
│
├── docs/                            # FROZEN planning docs (do not edit; only error_coding_scheme.md is added)
│   ├── final_protocol.md
│   ├── experiment_matrix.md
│   ├── evaluation_protocol.md
│   ├── research_questions.md
│   ├── hypotheses.md
│   ├── reproducibility_checklist.md
│   ├── pre_implementation_review.md
│   ├── experimental_design.md
│   └── error_coding_scheme.md       # created in EXP-G2 (§13)
│
└── literature/                      # FROZEN literature database (do not edit)
    ├── notes/ (P001–P012.md)
    ├── master_literature_database.csv
    ├── pattern_analysis.md
    ├── research_gap.md
    ├── reviewer_observations.md
    └── dataset_selection_report.md
```

**Folder purposes:**

- `configs/` — the ONLY place frozen constants live in machine-readable form; every module
  reads from here so no value is ever hard-coded (protocol §0). Editing a frozen value here
  is forbidden without a protocol version bump.
- `data/` — on Kaggle, raw bytes live in `/kaggle/input/` (attached datasets) and are NOT copied
  into the repo (`data/raw/.gitkeep` is a placeholder). The repo commits only lightweight,
  reproducible artifacts: `SNAPSHOT_HASHES.txt` (content-hash ledger + Kaggle slug/version
  manifest) and split indices (`splits/*.idx`). Large `processed/*.parquet` are produced under
  `/kaggle/working/` and persisted as a Kaggle output dataset (not committed to git).
- `src/fnb/` — the installable library, split by responsibility: `config`, `utils`,
  `data`, `models`, `training`, `evaluation`, `adversarial`, `analysis`, `experiments`.
- `scripts/` — argument-parsed entry points; contain no research logic, only orchestration.
- `notebooks/` — Kaggle T4 wrappers used for the actual GPU runs (efficiency numbers must
  come from a single fixed T4 per §14).
- `tests/` — unit tests guarding correctness of every deterministic component.
- `artifacts/`, `results/`, `logs/` — reproducible outputs; `results/` holds every CSV in
  the matrix Deliverables Index.
- `docs/`, `literature/` — frozen planning; read-only except adding `error_coding_scheme.md`.

---

# 3. Implementation Roadmap (Milestones)

The roadmap is strictly linear. Each milestone ends with a runnable project and depends
only on previous milestones.

| Milestone | Name | Steps | Depends on | Produces |
|:--:|------|:--:|:--:|------|
| **M1** | Project Foundation | S1–S5 | — | Installable package, configs, determinism, logging, run registry |
| **M2** | Data Pipeline (EXP-P0–P5) | S6–S12 | M1 | Snapshots+hashes, binarized/preprocessed/deduped datasets, splits, stats, cross-dedup |
| **M3** | Evaluation & Statistics Core | S13–S15 | M1 | Metric suite, statistical stack, result-row schema/writer |
| **M4** | Model & Training Infrastructure | S16–S19 | M2, M3 | Classical trainer, encoder+sliding-window, custom training loop, inference |
| **M5** | Primary Experiments (RQ-A/B/C) | S20–S24 | M4 | EXP-1, EXP-P6, EXP-A*, EXP-B*, EXP-C1 result CSVs |
| **M6** | Secondary Experiments (RQ-D/E/F) | S25–S27 | M5 | EXP-D*, EXP-E1, EXP-F* result CSVs |
| **M7** | Analysis & Interpretability | S28–S31 | M5 (M6 where applicable) | Calibration, error analysis, interpretability, confirmatory stats |
| **M8** | Paper Outputs & Final Verification | S32 | M5–M7 | Paper tables, figures, full reproducibility gate pass |

Priority (protocol §1): M5 + the primary parts of M7 (G1/G2/G4) are the primary endpoints
and are never cut. M6 (secondary D/E/F) runs only if within the ≤120 GPU-h budget; RQ-F is
capped at 20 GPU-h. If budget is threatened, cut secondary phases in order F → E → D.

---

# 4. Detailed Step-by-Step Guide

Difficulty scale: ★ trivial · ★★ easy · ★★★ moderate · ★★★★ hard · ★★★★★ very hard.
Time estimates assume one graduate student; GPU time is separate from coding time.

## Milestone 1 — Project Foundation (S1–S5)

### Step S1 — Repository scaffolding & installable package
- **Objective:** Create the folder tree of §2, an installable `fnb` package, dependency
  files, `README.md`, and `ENVIRONMENT.md` stub.
- **Files to Create:** `pyproject.toml`, `requirements.txt`, `requirements.lock.txt`
  (placeholder), `README.md`, `ENVIRONMENT.md`, `src/fnb/__init__.py`, all sub-package
  `__init__.py` files, empty `results/`, `artifacts/`, `logs/`, `data/{raw,processed,splits}/`
  with `.gitkeep`.
- **Files to Modify:** none.
- **Expected Output:** in a Kaggle notebook, `git clone <repo> && pip install -e .` succeeds and
  `python -c "import fnb; print(fnb.__version__)"` prints a version; a `paths` config abstracts
  `/kaggle/input` (read-only inputs) and `/kaggle/working` (outputs) so no path is hard-coded.
- **How to Verify It Works:** run the clone + `pip install -e .` + import cell on Kaggle; run
  `pytest -q` (0 tests collected but no import errors); paste the cell output back into Cursor.
- **Common Mistakes:** forgetting `src`-layout config in `pyproject.toml`; missing `__init__.py`;
  committing large data (raw bytes live only in `/kaggle/input/`, never in the repo); hard-coding
  absolute paths instead of a configurable `paths` block.
- **Estimated Difficulty:** ★★
- **Estimated Time:** 1–2 h
- **Definition of Done:** package installs+imports inside a Kaggle notebook, tree matches §2,
  `README.md` documents the Kaggle clone+install flow and the milestone run order, and a
  `notebooks/kaggle_runner.ipynb` skeleton (clone → install → attach data → dispatch → save
  `/kaggle/working/`) exists.

### Step S2 — Frozen config mirrors (YAML)
- **Objective:** Encode every FROZEN constant from `final_protocol.md` into the `configs/*.yaml`
  files listed in reproducibility §8, with exact values (no invented numbers).
- **Files to Create:** `configs/protocol.yaml`, `configs/datasets.yaml`,
  `configs/preprocessing.yaml`, `configs/dedup.yaml`, `configs/classical_grids.yaml`,
  `configs/encoder.yaml`, `configs/llm_qlora.yaml`, `configs/adversarial.yaml`,
  `configs/metrics.yaml`, `configs/stats.yaml`.
- **Files to Modify:** none.
- **Expected Output:** ten YAML files whose values match the protocol exactly (seeds
  `{13,21,42,87,100}`/`{13,42,100}`; encoder LR `2e-5`; batch `16`; warmup `0.06`; epochs `10`;
  patience `3`; max_len `512`; window `512`/stride `128`; title max_len `64`; dedup `0.95`/`0.90`
  + sensitivity `{0.90,0.95,0.98}`; QLoRA `r=64,alpha=16,dropout=0.2`, LR `2e-4`; bootstrap
  `n=10000`; BH-FDR `q=0.05`; USE `≥0.90`; budget `120`/`20` GPU-h; `protocol_version: v1.0`).
- **How to Verify It Works:** `python -c "import yaml,glob;[yaml.safe_load(open(f)) for f in glob.glob('configs/*.yaml')]"`
  parses all; a reviewer diff against `final_protocol.md` §4–§14 shows no divergence.
- **Common Mistakes:** transcribing a value that disagrees with the protocol (forbidden);
  hard-coding these later instead of reading YAML; omitting the dataset-version tag list.
- **Estimated Difficulty:** ★★
- **Estimated Time:** 2–3 h
- **Definition of Done:** all ten files parse; every value traceable to a protocol section;
  no value contradicts the protocol.

### Step S3 — Config loader & schema validation
- **Objective:** Provide `fnb.config.load_config(name)` that loads a YAML, validates it against
  a typed schema, logs a config hash, and exposes frozen constants as typed objects.
- **Files to Create:** `src/fnb/config/loader.py`, `src/fnb/config/schema.py`,
  `tests/test_config.py`.
- **Files to Modify:** `src/fnb/config/__init__.py` (exports).
- **Expected Output:** `load_config("encoder")` returns a validated object with `.lr == 2e-5`,
  `.batch_size == 16`, etc.; loading a config with a missing/invalid field raises a clear error.
- **How to Verify It Works:** `pytest tests/test_config.py` passes: valid configs load; a
  mutated fixture (e.g., LR removed) raises; `protocol_version` is asserted `== "v1.0"`.
- **Common Mistakes:** silent defaults masking missing frozen values; not logging the config
  hash (needed for repro §8); accepting unknown extra keys.
- **Estimated Difficulty:** ★★★
- **Estimated Time:** 3–4 h
- **Definition of Done:** loader validates all ten configs, rejects malformed ones, logs a
  stable per-config SHA-256, and tests pass.

### Step S4 — Determinism & global seeding
- **Objective:** Implement `set_global_seed(seed)` that seeds every RNG source and enables
  deterministic algorithms exactly as reproducibility §1 mandates.
- **Files to Create:** `src/fnb/utils/seeding.py`, `tests/test_seeding.py`.
- **Files to Modify:** `src/fnb/utils/__init__.py`.
- **Expected Output:** one call sets `random`, `numpy`, `torch`, `torch.cuda`,
  `PYTHONHASHSEED`, `transformers.set_seed`, enables `torch.use_deterministic_algorithms(True)`,
  `cudnn.deterministic=True`, `cudnn.benchmark=False`, sets `CUBLAS_WORKSPACE_CONFIG=:4096:8`,
  and returns a DataLoader `generator` + `worker_init_fn`.
- **How to Verify It Works:** `pytest tests/test_seeding.py` — two runs with the same seed
  produce identical tensors/`numpy` draws; different seeds differ; deterministic flags are set.
- **Common Mistakes:** setting `CUBLAS_WORKSPACE_CONFIG` after CUDA init (must be before);
  forgetting DataLoader `worker_init_fn`; not seeding HF.
- **Estimated Difficulty:** ★★★
- **Estimated Time:** 2–3 h
- **Definition of Done:** identical outputs across repeated seeded runs on CPU; deterministic
  toggles confirmed; tests pass.

### Step S5 — Logging, hashing & run registry
- **Objective:** Provide structured per-run logging, file/dataframe/config SHA-256 hashing, a
  `run_id` generator, and a `results/runs.csv` registry writing `metadata.json` per run
  (reproducibility §9, §10).
- **Files to Create:** `src/fnb/utils/logging_utils.py`, `src/fnb/utils/hashing.py`,
  `src/fnb/utils/run_registry.py`, `src/fnb/utils/io.py`.
- **Files to Modify:** `src/fnb/utils/__init__.py`.
- **Expected Output:** starting a run creates `logs/{run_id}.log` and appends a row to
  `results/runs.csv` with git commit SHA, resolved config hash, seed, library + CUDA versions,
  hardware, dataset-version tag, timestamps, wall-clock; `metadata.json` written per artifact dir.
- **How to Verify It Works:** call the registry in a smoke script; confirm log file, `runs.csv`
  row, and `metadata.json` appear with all required fields; hashing a file twice is stable.
- **Common Mistakes:** not enforcing a clean git tree; logging mutable config instead of the
  resolved+hashed one; non-append-only writes clobbering history.
- **Estimated Difficulty:** ★★★
- **Estimated Time:** 3–4 h
- **Definition of Done:** every future run can be reconstructed from its log + registry row;
  hashing is deterministic; git SHA captured.

## Milestone 2 — Data Pipeline (S6–S12)

### Step S6 — Dataset registration, snapshot & hashing (EXP-P0) — Kaggle input datasets
- **Objective:** Register DS1–DS5 as **Kaggle input datasets** (attached at `/kaggle/input/`),
  record each dataset's Kaggle slug + version, and compute a SHA-256 per dataset over the attached
  read-only files into `data/SNAPSHOT_HASHES.txt` (protocol §4.6). No local downloads: the pinned
  Kaggle dataset version IS the frozen snapshot; FakeNewsNet is used from its attached hydrated
  Kaggle dataset (record its slug/version + recovery fraction; do not re-crawl).
- **Files to Create:** `src/fnb/data/acquire.py`, `scripts/run_data_pipeline.py` (P0 stage).
- **Files to Modify:** `configs/datasets.yaml` (per-dataset Kaggle slug, version, `/kaggle/input/`
  path, label column, license note).
- **Expected Output:** `data/SNAPSHOT_HASHES.txt` (one SHA-256 per dataset, computed on Kaggle);
  a resolved manifest of Kaggle slug + version per dataset; FakeNewsNet source slug/version +
  recovery fraction recorded. (Raw bytes stay in `/kaggle/input/`; they are NOT copied into the
  git repo.)
- **How to Verify It Works:** in a Kaggle notebook, re-running acquisition over the same attached
  dataset versions reproduces identical hashes; attaching a different dataset version changes the
  hash; paste `SNAPSHOT_HASHES.txt` back into Cursor to commit it.
- **Common Mistakes:** copying large `/kaggle/input/` bytes into the repo (commit only hashes +
  the slug/version manifest); not pinning the Kaggle dataset *version*; hashing after any mutation;
  ignoring dataset licenses (§16).
- **Estimated Difficulty:** ★★★
- **Estimated Time:** 2–3 h authoring (dataset attach + hashing run on Kaggle)
- **Definition of Done:** all five datasets attached and pinned by Kaggle slug+version; SHA-256
  hashes recorded and stable across re-runs on the same versions; license notes captured.

### Step S7 — Binarization & label-mapping report (EXP-P1a)
- **Objective:** Apply the frozen global label space `{real=0, fake=1}` and the LIAR 6-way→binary
  map; drop and COUNT ambiguous rows (protocol §4.3).
- **Files to Create:** `src/fnb/data/binarize.py`, `tests/test_preprocess.py` (binarize cases).
- **Files to Modify:** `scripts/run_data_pipeline.py` (P1a stage).
- **Expected Output:** `{DSx}` at `vBIN`; `results/label_mapping_report.csv` with per-dataset
  dropped/ambiguous counts and the applied mapping.
- **How to Verify It Works:** unit test asserts LIAR mapping (`{pants-fire,false,barely-true}→fake`;
  `{half-true,mostly-true,true}→real`); label set is exactly `{0,1}`; counts sum correctly.
- **Common Mistakes:** inconsistent label polarity across datasets; silently dropping rows
  without counting; mislabeling LIAR classes.
- **Estimated Difficulty:** ★★
- **Estimated Time:** 3–4 h
- **Definition of Done:** every dataset has binary labels, report CSV written, tests pass.

### Step S8 — Dual-track preprocessing (EXP-P1b)
- **Objective:** Produce the two frozen preprocessing tracks (protocol §6): classical
  (`vCLEAN-C`) and neural (`vCLEAN-N`).
- **Files to Create:** `src/fnb/data/preprocess.py`.
- **Files to Modify:** `scripts/run_data_pipeline.py` (P1b), `tests/test_preprocess.py`.
- **Expected Output:** `data/processed/{DSx}_vCLEAN-C.parquet` and `{DSx}_vCLEAN-N.parquet`.
  Classical: NFC→lowercase→strip URLs/HTML/mentions/hashtags/symbols→NLTK stopwords→WordNet
  lemmatize. Neural: NFC→strip URLs/HTML only (no stopword/lemmatization; subword tokenization
  happens at model time).
- **How to Verify It Works:** unit tests on crafted strings verify each cleaning step for each
  track; neural track preserves case and stopwords; NLTK data versions recorded.
- **Common Mistakes:** applying stopword removal/lemmatization to the neural track (forbidden);
  fitting any vectorizer here (TF-IDF is fit later, on TRAIN only); not recording NLTK versions.
- **Estimated Difficulty:** ★★★
- **Estimated Time:** 4–5 h
- **Definition of Done:** both tracks produced for all datasets; step-by-step cleaning verified;
  NLTK stopword/WordNet versions recorded in `ENVIRONMENT.md`.

### Step S9 — Within-dataset deduplication (EXP-P2)
- **Objective:** Remove exact duplicates then near-duplicates (TF-IDF cosine ≥ 0.95) within each
  dataset; record counts (protocol §4.5).
- **Files to Create:** `src/fnb/data/dedup.py`, `tests/test_dedup.py`.
- **Files to Modify:** `scripts/run_data_pipeline.py` (P2).
- **Expected Output:** `data/processed/{DSx}_vDEDUP.parquet`; `results/dedup_counts.csv`
  (removed count and %-removed per dataset).
- **How to Verify It Works:** unit test with injected exact + near-duplicate rows confirms
  removal at threshold 0.95 and correct counts; original order/index preserved for survivors.
- **Common Mistakes:** hard-coding 0.95 instead of reading `configs/dedup.yaml`; deduping across
  the train/test boundary before splitting is fine here (within-dataset), but do NOT peek at
  labels; O(n²) memory blowup — use sparse TF-IDF + blocked cosine.
- **Estimated Difficulty:** ★★★★
- **Estimated Time:** 5–6 h
- **Definition of Done:** `vDEDUP` produced for DS1–DS5; counts CSV written; tests pass.

### Step S10 — Splits: random / source-disjoint / temporal (EXP-P3)
- **Objective:** Create stratified 70/15/15 splits, saving INDICES only, for `S-RAND` (all),
  `S-SRC` (DS1 iff source recoverable), `S-TEMP` (only where timestamps exist, else "N/A")
  (protocol §4.4). Split BEFORE any resampling/vectorizer fitting.
- **Files to Create:** `src/fnb/data/splits.py`, `tests/test_splits.py`.
- **Files to Modify:** `scripts/run_data_pipeline.py` (P3).
- **Expected Output:** `data/splits/{DSx}_{S-RAND|S-SRC|S-TEMP}_{seed}.idx` for the mandated
  seeds; a per-dataset source-recoverability + temporal-applicability note.
- **How to Verify It Works:** unit tests confirm 70/15/15 ratios, stratification preserves class
  balance, indices are disjoint and cover all rows, same seed → identical indices; `S-SRC` has
  no shared source across splits; unavailable regimes recorded "N/A".
- **Common Mistakes:** saving materialized splits instead of indices; leaking by fitting anything
  pre-split; silently skipping N/A instead of recording it.
- **Estimated Difficulty:** ★★★
- **Estimated Time:** 4–5 h
- **Definition of Done:** index files committed for all datasets×regimes×seeds; ratios/stratification
  verified; N/A cases explicitly recorded.

### Step S11 — Dataset statistics (EXP-P4)
- **Objective:** Record absolute sizes and class balances per dataset/split and text-length
  stats (protocol §4.4; fixes P011 omission).
- **Files to Create:** `src/fnb/data/stats.py`.
- **Files to Modify:** `scripts/run_data_pipeline.py` (P4), `tests/` (optional stats test).
- **Expected Output:** `results/dataset_stats.csv` with N per split, class ratio, mean/median
  article length, for `vBIN` and `vDEDUP`.
- **How to Verify It Works:** open CSV; counts match split index sizes; class ratios match the
  binarized data; lengths are plausible.
- **Common Mistakes:** computing stats on materialized text rather than the committed splits;
  mixing dataset versions in one row without tagging.
- **Estimated Difficulty:** ★★
- **Estimated Time:** 2–3 h
- **Definition of Done:** `dataset_stats.csv` complete for all datasets/splits/versions.

### Step S12 — Cross-dataset overlap removal (EXP-P5)
- **Objective:** For every directed transfer pair in {DS1,DS2,DS3} and DS→DS4, remove test-side
  items with TF-IDF cosine ≥ 0.90 to any training-side item; record counts (protocol §4.5).
- **Files to Create:** `src/fnb/data/xdedup.py`.
- **Files to Modify:** `scripts/run_data_pipeline.py` (P5), `tests/test_dedup.py` (xdedup cases).
- **Expected Output:** per-pair cleaned test indices `vXDEDUP(train→test)`;
  `results/interdataset_overlap.csv` with overlap counts per pair (also report WELFake↔ISOT
  residual size/class balance, since they share Reuters).
- **How to Verify It Works:** unit test injects a train↔test near-duplicate across two toy
  datasets and confirms test-side removal at 0.90; overlap counts are symmetric-aware per
  direction; residual class balance recorded.
- **Common Mistakes:** removing train-side instead of test-side items; recomputing TF-IDF
  inconsistently between within- and cross-dedup; not reporting residual balance after removal.
- **Estimated Difficulty:** ★★★★
- **Estimated Time:** 5–6 h
- **Definition of Done:** cleaned test indices for all transfer pairs; overlap CSV written;
  tests pass.

## Milestone 3 — Evaluation & Statistics Core (S13–S15)

### Step S13 — Metric suite (METRICS-CORE, ECE, TRANSFER, ROBUST)
- **Objective:** Implement the frozen metric suite (protocol §8; matrix §0.7): macro-F1
  (primary), accuracy, macro-precision, macro-recall, per-class F1 (fake/real), ROC-AUC,
  PR-AUC; ECE (15 bins, probabilistic models only); TRANSFER (ΔF1, std across targets);
  ROBUST (clean acc, acc-under-attack, ASR, mean similarity).
- **Files to Create:** `src/fnb/evaluation/metrics.py`, `tests/test_metrics.py`.
- **Files to Modify:** `src/fnb/evaluation/__init__.py`, `configs/metrics.yaml` (bin count).
- **Expected Output:** functions returning a dict of METRICS-CORE from `(y_true, y_pred, y_prob)`;
  `ece(y_true, y_prob, n_bins=15)`; `delta_f1(...)`; ROBUST aggregator. All reject best-run-only
  usage by design (they consume per-seed arrays and return mean ± std helpers).
- **How to Verify It Works:** `pytest tests/test_metrics.py` — known-answer tests against
  sklearn on fixed arrays; ECE matches a hand-computed toy example; macro-F1 is the default
  "winner" metric.
- **Common Mistakes:** using micro instead of macro averaging; computing ECE for models without
  valid probabilities (must be gated); PR-AUC/ROC-AUC on hard labels instead of scores.
- **Estimated Difficulty:** ★★★
- **Estimated Time:** 4–5 h
- **Definition of Done:** all metrics implemented, ECE gated to probabilistic models, tests pass.

### Step S14 — Statistical stack (STATS-CONF)
- **Objective:** Implement the frozen statistics (protocol §10; eval §R6): paired bootstrap on
  macro-F1 difference (10,000 stratified resamples, 95% CI of the difference, two-sided p),
  McNemar on paired predictions, Cliff's delta, Benjamini–Hochberg FDR at q=0.05 across the
  confirmatory family, and Friedman + Nemenyi for exploratory ranking.
- **Files to Create:** `src/fnb/evaluation/stats.py`, `tests/test_stats.py`.
- **Files to Modify:** `configs/stats.yaml`.
- **Expected Output:** `paired_bootstrap_diff(...)`, `mcnemar(...)`, `cliffs_delta(...)`,
  `bh_fdr(pvalues, q=0.05)`, `friedman_nemenyi(...)`; a `confirmatory_family` assembler that
  returns per-comparison `(adjusted_p, effect_size, ci_of_difference)`.
- **How to Verify It Works:** `pytest tests/test_stats.py` — bootstrap CI covers the true diff
  on synthetic data with known effect; BH-FDR matches `statsmodels`; McNemar matches a hand
  example; Cliff's delta signs are correct; seed makes the bootstrap reproducible.
- **Common Mistakes:** non-stratified resampling; forgetting BH-FDR is applied across the WHOLE
  confirmatory family (not per-test); reporting a raw delta without the required triple.
- **Estimated Difficulty:** ★★★★
- **Estimated Time:** 6–8 h
- **Definition of Done:** all tests implemented and verified against reference libraries; a
  confirmatory claim helper enforces the (adjusted p, effect size, CI) triple.

### Step S15 — Result-row schema & provenance writer
- **Objective:** Define the canonical result-row schema and a writer that stamps every row with
  `protocol_version=v1.0`, dataset-version tag, split type, model ID, seed, run_id, config hash,
  git SHA (eval §R4), and aggregates mean ± std across seeds.
- **Files to Create:** `src/fnb/evaluation/results_io.py` (or extend `utils/io.py`),
  `tests/test_results_io.py`.
- **Files to Modify:** `src/fnb/evaluation/__init__.py`.
- **Expected Output:** `write_result_rows(...)` appending to the correct `results/*.csv`;
  `aggregate_seeds(...)` producing mean ± std tables; schema validation on write.
- **How to Verify It Works:** writing sample rows yields a CSV with all provenance columns;
  aggregation matches manual mean ± std; a row missing `protocol_version` is rejected.
- **Common Mistakes:** omitting provenance columns; overwriting instead of appending; reporting
  best-run-only (must aggregate across mandated seeds).
- **Estimated Difficulty:** ★★★
- **Estimated Time:** 3–4 h
- **Definition of Done:** every result CSV shares one schema with full provenance; aggregation
  is mean ± std; tests pass.

## Milestone 4 — Model & Training Infrastructure (S16–S19)

### Step S16 — Classical model trainer (TF-IDF + LR/SVM/RF/LGBM)
- **Objective:** Implement the classical family (protocol §5) on `vCLEAN-C`: TF-IDF
  (unigrams+bigrams, `max_features=20000`, `min_df=5`) fit on TRAIN ONLY, per-model grid from
  `configs/classical_grids.yaml` selected by validation macro-F1, with probability outputs
  (SVM via Platt scaling) for ECE.
- **Files to Create:** `src/fnb/models/classical.py`, `tests/test_classical.py`.
- **Files to Modify:** `configs/classical_grids.yaml`.
- **Expected Output:** trainer producing predictions + probabilities on val/test; serialized
  `artifacts/tfidf_{seed}.pkl`; grid selection logged.
- **How to Verify It Works:** on a tiny synthetic dataset, LR/SVM/RF/LGBM train and predict;
  vectorizer is fit only on train indices (test never seen); Platt scaling yields probabilities.
- **Common Mistakes:** fitting TF-IDF on all data (leakage); using SMOTE (prohibited);
  rebalancing the test set; not saving the vectorizer.
- **Estimated Difficulty:** ★★★
- **Estimated Time:** 5–6 h
- **Definition of Done:** all four classical models train/predict with probabilities; TF-IDF
  train-only; grids read from config; tests pass.

### Step S17 — Encoder dataset & sliding-window tokenization
- **Objective:** Implement a torch `Dataset` on `vCLEAN-N` that tokenizes full documents into
  512-token windows with stride 128 (title-only condition: max 64), producing per-window inputs
  keyed to each document (protocol §6.1).
- **Files to Create:** `src/fnb/training/datasets.py`, `tests/test_encoder.py` (tokenization part).
- **Files to Modify:** `configs/encoder.yaml` (window/stride/max_len).
- **Expected Output:** dataset yielding windowed `input_ids`/`attention_mask` + a document index
  so window logits can be mean-pooled per document at inference.
- **How to Verify It Works:** unit test: a long document produces the expected number of windows
  at window=512/stride=128; a short document produces one window; title-only caps at 64.
- **Common Mistakes:** truncating to 512 instead of windowing (forbidden — transformers must see
  the full document); losing the document→window mapping; applying classical cleaning here.
- **Estimated Difficulty:** ★★★★
- **Estimated Time:** 5–6 h
- **Definition of Done:** windowing matches the frozen params; document mapping preserved; tests pass.

### Step S18 — Custom PyTorch training loop (NO HF Trainer)
- **Objective:** Implement the frozen training configuration (protocol §7) as a hand-written
  loop: AdamW (wd=0.01), LR 2e-5, batch 16, linear warmup 6% → cosine decay to 0, ≤10 epochs,
  early stopping on validation macro-F1 (patience 3), inverse-frequency class-weighted
  cross-entropy, SMOTE prohibited. Must NOT use `transformers.Trainer`.
- **Files to Create:** `src/fnb/training/loop.py`, `src/fnb/training/scheduler.py`,
  `src/fnb/training/early_stopping.py`, `tests/test_training_loop.py`.
- **Files to Modify:** `src/fnb/models/encoder.py` (model wrapper for HF encoders).
- **Expected Output:** `train(model, loaders, cfg, seed)` returning the checkpoint selected by
  best val macro-F1, with a per-epoch log (loss, val macro-F1, LR trace, early-stop decision).
- **How to Verify It Works:** `pytest tests/test_training_loop.py` — loss decreases on a tiny
  overfit set; scheduler reaches ~0 at end; early stopping fires after patience 3; class weights
  computed from train frequencies; no `Trainer` import anywhere (grep check in test).
- **Common Mistakes:** importing HF `Trainer` (forbidden by stack); wrong warmup fraction; not
  mean-pooling window logits during validation; using accuracy instead of macro-F1 for early stop.
- **Estimated Difficulty:** ★★★★★
- **Estimated Time:** 8–10 h
- **Definition of Done:** custom loop trains an encoder end-to-end on a small sample, honors all
  §7 settings from config, selects by val macro-F1, contains no HF Trainer; tests pass.

### Step S19 — Encoder inference & window logit pooling
- **Objective:** Implement prediction that mean-pools per-document window logits into a single
  document prediction + probability (softmax), reused by every downstream experiment.
- **Files to Create:** `src/fnb/models/encoder.py` (predict path), extend `tests/test_encoder.py`.
- **Files to Modify:** none.
- **Expected Output:** `predict(model, dataset)` returning `(y_pred, y_prob)` per document via
  mean-pooled window logits; deterministic under a fixed seed.
- **How to Verify It Works:** unit test: a document split into k windows yields one pooled logit
  equal to the mean of window logits; probabilities sum to 1; matches training-time validation path.
- **Common Mistakes:** pooling probabilities instead of logits; mismatched pooling between train
  validation and test inference; forgetting eval mode / `no_grad`.
- **Estimated Difficulty:** ★★★
- **Estimated Time:** 3–4 h
- **Definition of Done:** consistent windowed inference producing per-document `(y_pred, y_prob)`;
  identical pooling in validation and test; tests pass.

## Milestone 5 — Primary Experiments RQ-A/B/C (S20–S24)

### Step S20 — EXP-1 in-distribution sweep + best_encoder selection
- **Objective:** Run all 9 configs (4 classical + 5 encoders) on DS1 `vDEDUP`/`S-RAND`, 5 seeds;
  compute METRICS-CORE + ECE; run pairwise encoder STATS-CONF (H1); write the frozen
  `best_encoder` (deterministic tie-break: highest mean macro-F1, then lowest ECE, then fewest
  params).
- **Files to Create:** `src/fnb/experiments/exp1_indist.py`; add dispatch in `scripts/run_experiment.py`.
- **Files to Modify:** none.
- **Expected Output:** `results/ds1_indist.csv` (model×seed), `results/ds1_encoder_pairwise_stats.csv`,
  `results/best_encoder.txt`.
- **How to Verify It Works:** CSVs contain 9 models × 5 seeds with full provenance; pairwise stats
  carry (adjusted p, δ, CI); `best_encoder.txt` is reproducible under the tie-break rule.
- **Common Mistakes:** reporting best-run-only; selecting best_encoder by a non-frozen rule;
  running fewer than 5 seeds for a primary endpoint.
- **Estimated Difficulty:** ★★★★
- **Estimated Time:** 4–5 h coding (+~30 GPU-h)
- **Definition of Done:** all three artifacts produced; H1 pairwise stats present; best_encoder frozen.

### Step S21 — EXP-P6 dedup-threshold sensitivity (DS1)
- **Objective:** Re-run best encoder + LR on DS1 at dedup thresholds {0.90, 0.95, 0.98}, 3 seeds;
  report metric stability (protocol §4.5).
- **Files to Create:** `src/fnb/experiments/exp_p6_dedup_sens.py`; dispatch entry.
- **Files to Modify:** none.
- **Expected Output:** `results/dedup_sensitivity_ds1.csv` (threshold × model × seed).
- **How to Verify It Works:** three thresholds present; each reuses the dedup utility from S9 with
  the config value; metrics reported mean ± std.
- **Common Mistakes:** re-deriving dedup logic instead of reusing S9; hard-coding thresholds.
- **Estimated Difficulty:** ★★
- **Estimated Time:** 2–3 h coding (+~2 GPU-h)
- **Definition of Done:** sensitivity CSV complete for all three thresholds.

### Step S22 — EXP-A1/A2/A3 transfer & short-statement track
- **Objective:** Build the directed article-type transfer matrix over {DS1,DS2,DS3} using
  `vXDEDUP(train→test)` off-diagonal (reusing EXP-1 DS1 diagonal), the DS4 domain-shift probe,
  and the separate LIAR track; compute TRANSFER (ΔF1, std) and STATS-CONF ΔF1 (H2). LIAR is
  NEVER merged into article transfer (protocol §4.2).
- **Files to Create:** `src/fnb/experiments/expA_transfer.py`; dispatch entry.
- **Files to Modify:** none.
- **Expected Output:** `results/transfer_matrix.csv`, `results/transfer_deltaF1.csv`,
  `results/transfer_family_exploratory.csv`, `results/domain_shift_ds4.csv`,
  `results/liar_shortstatement.csv`.
- **How to Verify It Works:** transfer matrix has train×test×model×seed rows; off-diagonal tests
  use cross-deduped indices; ΔF1 carries the confirmatory triple; LIAR results are in a separate CSV.
- **Common Mistakes:** using non-cross-deduped test sets off-diagonal; mixing LIAR into article
  transfer; recomputing DS1 diagonal instead of reusing EXP-1.
- **Estimated Difficulty:** ★★★★
- **Estimated Time:** 5–6 h coding (+~28 GPU-h)
- **Definition of Done:** all five CSVs produced; H2 ΔF1 stats present; LIAR isolated.

### Step S23 — EXP-B1/B2/B3 leakage & source-only shortcut
- **Objective:** Quantify leakage on DS1: before-vs-after dedup (H3 part 1), random-vs-source-disjoint
  split (H3 part 2; N/A with note if source unrecoverable), and the source/metadata-only shortcut
  baseline vs full-text (H4). Full-text vs source-only is in the confirmatory family.
- **Files to Create:** `src/fnb/models/source_only.py`, `src/fnb/experiments/expB_leakage.py`;
  dispatch entry; extend `tests/`.
- **Files to Modify:** none.
- **Expected Output:** `results/leakage_analysis.csv` with the dedup delta, split-regime delta
  (or N/A note), and the source-only/full-text ratio + H4 ≥0.80 check.
- **How to Verify It Works:** CSV rows for all three contrasts; source-recoverability decision
  recorded; if source absent, SRCONLY falls back to surface features (length/punctuation/caps)
  and this is noted; STATS-CONF triple on the confirmatory contrast.
- **Common Mistakes:** claiming a source-disjoint result when source is unrecoverable; using
  article text in the SRCONLY baseline; skipping the honest N/A note.
- **Estimated Difficulty:** ★★★★
- **Estimated Time:** 5–6 h coding (+~10 GPU-h)
- **Definition of Done:** leakage CSV complete; H3/H4 readouts present; source scoping honest.

### Step S24 — EXP-C1 adversarial robustness (TextAttack)
- **Objective:** Apply label-preserving perturbations (char typos, embedding-constrained synonym
  substitution, benign insertion, back-translation) to the DS1 test set for all encoders +
  classical + (budget permitting) the 2 LLMs, with USE similarity ≥ 0.90 (attacks below discarded);
  negation/antonym prohibited (protocol §12). Evaluate on fixed checkpoints (no retraining).
- **Files to Create:** `src/fnb/adversarial/attacks.py`, `src/fnb/experiments/expC_adversarial.py`;
  dispatch entry.
- **Files to Modify:** `configs/adversarial.yaml`.
- **Expected Output:** `results/adversarial_robustness.csv` (clean acc, acc-under-attack, ASR,
  mean similarity per model/family); released perturbation sets (documented defensive, §16).
- **How to Verify It Works:** every counted attack satisfies USE ≥ 0.90; prohibited transforms
  are absent; ASR computed only on validity-filtered attacks; runs on saved checkpoints.
- **Common Mistakes:** counting meaning-changing attacks; retraining instead of attacking fixed
  checkpoints; unmatched attack budget across families (use identical recipe + query cap).
- **Estimated Difficulty:** ★★★★★
- **Estimated Time:** 7–9 h coding (+~12 GPU-h)
- **Definition of Done:** robustness CSV complete; validity constraint enforced; H5 readout present.

## Milestone 6 — Secondary Experiments RQ-D/E/F (S25–S27)

### Step S25 — EXP-D1/D2 efficiency & compression
- **Objective:** Profile all encoders + LR/LGBM refs + DISTIL + QUANT-BEST on a SINGLE fixed
  NVIDIA T4: params, on-disk MB, peak VRAM, training wall-clock, inference latency (ms/sample @
  b=1,seq=512), throughput (samples/s @ b=32); build F1-vs-params and F1-vs-latency Pareto
  frontiers (H6); optional INT8+KD compression study (protocol §14).
- **Files to Create:** `src/fnb/experiments/expD_efficiency.py`, `src/fnb/analysis/figures.py`
  (Pareto plots); dispatch entry.
- **Files to Modify:** none.
- **Expected Output:** `results/efficiency.csv`, `results/compression_study.csv`, Pareto plots.
- **How to Verify It Works:** all timing rows carry the T4 tag; non-T4 runs excluded; H6 ratio
  (≥95% F1 at <50% params) computed; plots render.
- **Common Mistakes:** mixing accelerators in efficiency tables; measuring latency at wrong
  batch/seq; forgetting peak VRAM.
- **Estimated Difficulty:** ★★★
- **Estimated Time:** 4–5 h coding (+~6 GPU-h)
- **Definition of Done:** efficiency + compression CSVs on fixed T4; Pareto plots produced.

### Step S26 — EXP-E1 input granularity
- **Objective:** Run title / content / title+content conditions on {DS1,DS2,DS3} for all 5
  encoders (+ classical), 3 seeds (title-only uses HP-TITLE max_len 64); test ranking stability
  via Friedman + Nemenyi (exploratory, E2) (protocol §6.1).
- **Files to Create:** `src/fnb/experiments/expE_granularity.py`; dispatch entry.
- **Files to Modify:** none.
- **Expected Output:** `results/granularity.csv` + ranking-stability table.
- **How to Verify It Works:** three conditions × datasets with title-only capped at 64; Friedman
  + Nemenyi labeled exploratory; only datasets with title+body used.
- **Common Mistakes:** running granularity on DS4/DS5 (no both-fields); using confirmatory FDR on
  exploratory ranking.
- **Estimated Difficulty:** ★★★
- **Estimated Time:** 4–5 h coding (+~8 GPU-h)
- **Definition of Done:** granularity CSV + exploratory ranking table produced.

### Step S27 — EXP-F0/F1/F2 LLM case study (QLoRA)
- **Objective:** Run the 2-LLM case study: F0 contamination probe (gate), F1 QLoRA fine-tune on
  DS1 (HP-LLM, 3 seeds, ≤20 GPU-h cap else partial), F2 zero-/5-shot (contamination-suspect,
  separate table, no ECE). Flagged datasets excluded from that LLM's zero-shot claims
  (protocol §4.7).
- **Files to Create:** `src/fnb/models/llm_qlora.py`, `src/fnb/experiments/expF_llm.py`;
  dispatch entry.
- **Files to Modify:** `configs/llm_qlora.yaml`.
- **Expected Output:** `results/llm_contamination.csv`, `results/llm_finetune_ds1.csv`,
  `results/llm_prompted.csv`; committed QLoRA adapters.
- **How to Verify It Works:** contamination flags computed before any zero-shot claim; fine-tune
  honors QLoRA config; prompted results in a separate table without ECE; GPU-h ≤ 20 (else marked
  partial).
- **Common Mistakes:** merging prompted with fine-tuned results; computing ECE for zero-shot LLMs;
  exceeding the 20 GPU-h cap without marking partial.
- **Estimated Difficulty:** ★★★★★
- **Estimated Time:** 8–10 h coding (+≤20 GPU-h)
- **Definition of Done:** all three LLM CSVs + adapters produced; contamination gate enforced;
  budget cap respected.

## Milestone 7 — Analysis & Interpretability (S28–S31)

### Step S28 — EXP-G1 calibration
- **Objective:** Report ECE (15 bins) + reliability diagrams for all probabilistic models on DS1
  test (+ transfer targets); LLM zero-shot excluded (protocol §8).
- **Files to Create:** `src/fnb/analysis/calibration.py`, `src/fnb/experiments` entry.
- **Files to Modify:** `src/fnb/analysis/figures.py` (reliability diagrams).
- **Expected Output:** `results/calibration.csv`; reliability plots.
- **How to Verify It Works:** ECE only for probabilistic models; diagrams render; values match S13.
- **Common Mistakes:** including LLM zero-shot; wrong bin count.
- **Estimated Difficulty:** ★★
- **Estimated Time:** 3–4 h
- **Definition of Done:** calibration CSV + plots produced for eligible models.

### Step S29 — EXP-G2 error analysis & failure taxonomy
- **Objective:** Produce per-dataset confusion matrices, fake→real vs real→fake asymmetry, a
  written coding scheme (`docs/error_coding_scheme.md`), ≥200 double-coded errors where feasible,
  difficulty buckets (DS2 easy / DS5 hard), and performance-vs-length (protocol §13).
- **Files to Create:** `src/fnb/analysis/error_analysis.py`, `docs/error_coding_scheme.md`;
  dispatch entry.
- **Files to Modify:** `src/fnb/analysis/figures.py`.
- **Expected Output:** `results/error_analysis.csv`; `docs/error_coding_scheme.md`.
- **How to Verify It Works:** confusion matrices per dataset; error-type distribution present;
  length buckets computed; coding scheme documented.
- **Common Mistakes:** no written coding scheme; ignoring class asymmetry.
- **Estimated Difficulty:** ★★★
- **Estimated Time:** 5–6 h
- **Definition of Done:** error CSV + coding scheme produced; asymmetry and length analysis included.

### Step S30 — EXP-G3 interpretability with faithfulness
- **Objective:** Integrated Gradients + leave-one-token occlusion on the best encoder with
  comprehensiveness/sufficiency faithfulness checks; attention illustrative ONLY (protocol §13).
- **Files to Create:** `src/fnb/analysis/interpretability.py`; dispatch entry.
- **Files to Modify:** `src/fnb/analysis/figures.py` (attribution figures).
- **Expected Output:** `results/interpretability.csv`; example attribution figures.
- **How to Verify It Works:** comprehensiveness/sufficiency computed; IG/occlusion agreement
  reported; attention never used as the sole explanation.
- **Common Mistakes:** presenting attention as explanation; skipping faithfulness metrics.
- **Estimated Difficulty:** ★★★★
- **Estimated Time:** 6–7 h
- **Definition of Done:** interpretability CSV + faithfulness scores + figures produced.

### Step S31 — EXP-G4 confirmatory statistics aggregation
- **Objective:** Assemble the pre-registered confirmatory family (EXP-1 pairwise encoders, EXP-A1
  ΔF1, EXP-B3 full-vs-source + EXP-B1/B2 deltas) and apply BH-FDR at q=0.05 across the WHOLE
  family, attaching effect sizes + CIs; emit H1–H4 verdicts (protocol §10; matrix §9).
- **Files to Create:** `src/fnb/experiments/expG_confirmatory.py`; dispatch entry.
- **Files to Modify:** none.
- **Expected Output:** `results/confirmatory_stats.csv` with per-comparison (adjusted p, CI of
  difference, Cliff's delta) and H1–H4 significance verdicts.
- **How to Verify It Works:** the family membership matches matrix §9 exactly; BH-FDR applied once
  across all family members; each verdict cites the required triple.
- **Common Mistakes:** applying FDR per-test instead of family-wide; including exploratory
  comparisons in the confirmatory family; issuing a verdict from a raw delta.
- **Estimated Difficulty:** ★★★★
- **Estimated Time:** 4–5 h
- **Definition of Done:** confirmatory CSV + H1–H4 verdicts produced under family-wide BH-FDR.

## Milestone 8 — Paper Outputs & Final Verification (S32)

### Step S32 — Paper tables, figures & reproducibility gate
- **Objective:** Generate all paper-ready tables/figures from the result CSVs and run the full
  reproducibility pre-report gate (checklist §"Pre-report Gate").
- **Files to Create:** `scripts/build_paper_outputs.py`; extend `src/fnb/analysis/figures.py`.
- **Files to Modify:** `ENVIRONMENT.md` (final pinned versions + hardware), `README.md` (repro steps).
- **Expected Output:** `results/paper/` with the main results table, transfer table, leakage
  table, robustness table, efficiency/Pareto, calibration, and confirmatory-verdict table; a
  passing reproducibility gate report.
- **How to Verify It Works:** every table sources only from `results/*.csv` with mean ± std +
  provenance; the 10-point pre-report gate is all-green; `ENVIRONMENT.md` complete.
- **Common Mistakes:** hand-editing numbers into tables; missing provenance/version stamps;
  publishing before the gate is green.
- **Estimated Difficulty:** ★★★
- **Estimated Time:** 5–6 h
- **Definition of Done:** all paper tables/figures generated from CSVs; reproducibility gate
  passes; framework ready for the paper's Results section.

---

# 5. Cursor Prompts (one per step)

Global coding standards referenced by every prompt below (do not repeat verbatim; assume they
apply): Python 3.11, `src`-layout package `fnb`, PEP 8 + type hints on all public functions,
Google-style docstrings, `ruff`+`black` clean, no hard-coded frozen constants (read from
`configs/` via `fnb.config.load_config`), structured logging via `fnb.utils.logging_utils`,
determinism via `fnb.utils.seeding.set_global_seed`, `pytest` unit tests for deterministic logic,
and strict conformance to `docs/final_protocol.md` (v1.0, FROZEN) and `docs/experiment_matrix.md`.
Each prompt generates exactly ONE logical component.

### Prompt S1 — Scaffolding

```
Create the project scaffolding for a Python research package named `fnb` using a src-layout.
Follow the folder tree in MASTER_IMPLEMENTATION_GUIDE.md §2 exactly. Create pyproject.toml
(setuptools src-layout, package = src/fnb, console entry points for scripts/), requirements.txt
with pinned versions for: torch, transformers, tokenizers, datasets, accelerate, peft,
bitsandbytes, scikit-learn, lightgbm, nltk, sentence-transformers, textattack, statsmodels,
scikit-posthocs, pandas, numpy, scipy, pyarrow, pyyaml, matplotlib, captum, pytest, ruff, black.
Create empty __init__.py for fnb and every sub-package (config, utils, data, models, training,
evaluation, adversarial, analysis, experiments). Add .gitkeep to data/{raw,processed,splits},
results, artifacts, logs. The execution target is KAGGLE (single T4), authored in Cursor: write
README.md documenting the Kaggle flow (first notebook cell: git clone the repo, then
`pip install -e .`, enable Internet, attach input datasets), the /kaggle/input (read-only) vs
/kaggle/working (outputs) convention, and the M1→M8 run order. Create notebooks/kaggle_runner.ipynb
as a thin skeleton with cells for: clone+install, attach datasets, dispatch a stage/experiment via
scripts/, and save /kaggle/working outputs as a versioned dataset. Write an ENVIRONMENT.md stub
with sections for pinned libs, CUDA, and the Kaggle T4 hardware. Do not implement logic yet.
Production-quality, documented.
```

### Prompt S2 — Config mirrors

```
Create the ten YAML config files in configs/ listed in MASTER_IMPLEMENTATION_GUIDE.md §2 and
docs/reproducibility_checklist.md §8. Every value MUST be transcribed exactly from
docs/final_protocol.md (v1.0, FROZEN) §4–§14 and docs/experiment_matrix.md §0 — do not invent or
change any number. Include: protocol.yaml (protocol_version: v1.0, primary seeds [13,21,42,87,100],
secondary seeds [13,42,100], compute budget 120 GPU-h, RQ-F cap 20 GPU-h, dataset-version tag
list); datasets.yaml (DS1..DS5 ids/roles/text-type/source paths/label columns/licenses);
preprocessing.yaml (classical vs neural tracks per §6); dedup.yaml (within 0.95, cross 0.90, DS1
sensitivity [0.90,0.95,0.98]); classical_grids.yaml (per-model grids for LR/SVM/RF/LGBM);
encoder.yaml (AdamW wd 0.01, lr 2e-5, batch 16, warmup 0.06, cosine decay, epochs 10, patience 3,
max_len 512, window 512, stride 128, title max_len 64, class_weighted_ce true, smote false);
llm_qlora.yaml (r 64, alpha 16, dropout 0.2, NF4, double_quant true, lr 2e-4, effective batch 16
via grad accumulation, zero-shot and 5-shot prompt templates); adversarial.yaml (attacks: char
typos, embedding synonym, benign insertion, back-translation; prohibited: negation, antonym; USE
similarity >= 0.90); metrics.yaml (metric list + ECE 15 bins); stats.yaml (bootstrap n 10000,
mcnemar true, cliffs_delta true, bh_fdr q 0.05, friedman_nemenyi true). Add a comment on each key
citing its protocol section. Do not write code.
```

### Prompt S3 — Config loader & schema

```
Implement src/fnb/config/loader.py and src/fnb/config/schema.py. Provide load_config(name: str)
that reads configs/{name}.yaml, validates it against a typed schema (use pydantic v2 dataclasses),
logs a stable SHA-256 config hash via fnb.utils.hashing, and returns the validated typed object.
schema.py defines one model per config from Prompt S2 with exact field types and NO silent
defaults for frozen values; reject unknown extra keys; assert protocol_version == "v1.0". Write
tests/test_config.py: all ten configs load and validate; a fixture with a removed/invalid frozen
field raises a clear ValidationError; the config hash is stable across calls. Google-style
docstrings, full type hints, ruff/black clean.
```

### Prompt S4 — Seeding & determinism

```
Implement src/fnb/utils/seeding.py with set_global_seed(seed: int) that seeds random, numpy,
torch, torch.cuda.manual_seed_all, os.environ PYTHONHASHSEED, and transformers.set_seed; enables
torch.use_deterministic_algorithms(True), torch.backends.cudnn.deterministic=True,
cudnn.benchmark=False; sets CUBLAS_WORKSPACE_CONFIG=:4096:8 (documented: must be set before CUDA
init); and returns a torch.Generator plus a worker_init_fn for DataLoader reproducibility. Follow
docs/reproducibility_checklist.md §1 exactly. Write tests/test_seeding.py proving two same-seed
runs yield identical numpy and torch draws on CPU, different seeds differ, and deterministic flags
are set. Full type hints, docstrings, ruff/black clean.
```

### Prompt S5 — Logging, hashing, run registry

```
Implement src/fnb/utils/logging_utils.py, hashing.py, run_registry.py, and io.py per
docs/reproducibility_checklist.md §9 and §10. logging_utils: configure console+file logging to
logs/{run_id}.log at a fixed level with structured fields. hashing: sha256 of files, of a pandas
DataFrame (stable row/col ordering), and of a resolved config dict. run_registry: generate a
unique run_id per (experiment, model, dataset-version, split, seed); enforce a clean git working
tree (capture git commit SHA); append a row to results/runs.csv with git SHA, config hash, seed,
library+CUDA versions, hardware, dataset-version tag, start/end timestamps, wall-clock; write a
metadata.json into a given artifact dir. io: parquet/csv/.idx read-write helpers (append-only for
result CSVs). Write a smoke test creating a run and asserting the log file, runs.csv row, and
metadata.json exist with all required fields. Docstrings, type hints, ruff/black clean.
```

### Prompt S6 — Dataset registration & hashing (EXP-P0), Kaggle inputs

```
Implement src/fnb/data/acquire.py and the P0 stage of scripts/run_data_pipeline.py per
docs/experiment_matrix.md EXP-P0 and docs/final_protocol.md §4.6. The runtime is a KAGGLE
notebook: datasets DS1–DS5 are attached read-only under /kaggle/input/<slug>/ (paths, Kaggle
slug, and pinned version in configs/datasets.yaml). Do NOT download or copy raw bytes into the
repo. Load each dataset from its /kaggle/input path, and compute one SHA-256 per dataset over its
attached files using fnb.utils.hashing, writing data/SNAPSHOT_HASHES.txt plus a resolved manifest
of {dataset -> kaggle_slug, version, license}. For FakeNewsNet, read from its attached hydrated
Kaggle dataset (record slug/version + recovery fraction; do not re-crawl). Respect licenses (§16).
Re-running over the same attached dataset versions must reproduce identical hashes. Add a CLI flag
to run only the P0 stage. Log via fnb.utils.logging_utils. Make paths configurable so the same
code runs on Kaggle (/kaggle/input, /kaggle/working) via configs, not hard-coded. Docstrings,
type hints, ruff/black clean. Do not modify other pipeline stages.
```

### Prompt S7 — Binarization (EXP-P1a)

```
Implement src/fnb/data/binarize.py per docs/final_protocol.md §4.3 and EXP-P1. Apply the frozen
global label space {real=0, fake=1} to all datasets and the LIAR 6-way→binary map
({pants-fire,false,barely-true}->fake; {half-true,mostly-true,true}->real), reading any
dataset-specific label columns from configs/datasets.yaml. Drop rows with no clean binary mapping
and COUNT them. Write results/label_mapping_report.csv (per dataset: applied mapping,
dropped/ambiguous counts). Produce vBIN dataframes. Add the binarize cases to
tests/test_preprocess.py asserting the LIAR mapping, a {0,1} label set, and correct counts. Wire
into scripts/run_data_pipeline.py P1a stage. Docstrings, type hints, ruff/black clean.
```

### Prompt S8 — Preprocessing tracks (EXP-P1b)

```
Implement src/fnb/data/preprocess.py per docs/final_protocol.md §6 and EXP-P1. Produce two frozen
tracks reading steps from configs/preprocessing.yaml. Classical (vCLEAN-C): Unicode NFC ->
lowercase -> strip URLs/HTML/mentions/hashtags/symbols -> NLTK stopword removal -> WordNet
lemmatization. Neural (vCLEAN-N): Unicode NFC -> strip URLs/HTML only (NO stopword removal, NO
lemmatization; subword tokenization is done later at model time). Write
data/processed/{DSx}_vCLEAN-C.parquet and _vCLEAN-N.parquet. Record NLTK stopword + WordNet
versions and download date into ENVIRONMENT.md. Add per-step unit tests in
tests/test_preprocess.py on crafted strings for BOTH tracks (verify the neural track preserves
case and stopwords). Do NOT fit any vectorizer here. Wire into run_data_pipeline.py P1b stage.
Docstrings, type hints, ruff/black clean.
```

### Prompt S9 — Within-dataset dedup (EXP-P2)

```
Implement src/fnb/data/dedup.py per docs/final_protocol.md §4.5 and EXP-P2. Remove exact
duplicates (hash-based) then near-duplicates using TF-IDF cosine similarity >= threshold read
from configs/dedup.yaml (0.95). Use sparse TF-IDF and blocked/batched cosine to avoid O(n^2)
memory. Produce data/processed/{DSx}_vDEDUP.parquet and write results/dedup_counts.csv (removed
count and %-removed per dataset). Preserve survivor indices for later split alignment. Do not peek
at labels. Write tests/test_dedup.py injecting exact + near-duplicate rows and asserting removal
at 0.95 and correct counts. Wire into run_data_pipeline.py P2 stage. Docstrings, type hints,
ruff/black clean.
```

### Prompt S10 — Splits (EXP-P3)

```
Implement src/fnb/data/splits.py per docs/final_protocol.md §4.4 and EXP-P3. Create stratified
70/15/15 train/val/test splits saving INDICES ONLY to
data/splits/{DSx}_{S-RAND|S-SRC|S-TEMP}_{seed}.idx for the mandated seeds
(configs/protocol.yaml). S-RAND for all datasets; S-SRC (source-disjoint) for DS1 ONLY iff source
is recoverable (else record "not applicable" with a note); S-TEMP only where reliable timestamps
exist (else "not applicable" per dataset). Splitting must happen BEFORE any resampling or
vectorizer fitting. Write tests/test_splits.py verifying 70/15/15 ratios, class stratification,
disjoint+complete indices, seed reproducibility, and source-disjointness for S-SRC. Wire into
run_data_pipeline.py P3 stage. Docstrings, type hints, ruff/black clean.
```

### Prompt S11 — Dataset statistics (EXP-P4)

```
Implement src/fnb/data/stats.py per docs/final_protocol.md §4.4 and EXP-P4. Using the committed
split indices, compute per dataset/split: N, class ratio, mean/median article length, for both
vBIN and vDEDUP versions. Write results/dataset_stats.csv with a dataset-version tag column. Wire
into run_data_pipeline.py P4 stage. Docstrings, type hints, ruff/black clean.
```

### Prompt S12 — Cross-dataset overlap removal (EXP-P5)

```
Implement src/fnb/data/xdedup.py per docs/final_protocol.md §4.5 and EXP-P5. For every directed
transfer pair among {DS1,DS2,DS3} and each DS->DS4, remove TEST-side items whose TF-IDF cosine
>= 0.90 (from configs/dedup.yaml) to ANY training-side item; save per-pair cleaned test indices
(vXDEDUP(train->test)). Write results/interdataset_overlap.csv with overlap counts per direction,
and additionally report WELFake<->ISOT residual test size and class balance after removal (they
share Reuters real news). Reuse the TF-IDF utility from dedup.py for consistency. Add cross-dedup
cases to tests/test_dedup.py (inject a train<->test near-duplicate across two toy datasets; assert
test-side removal at 0.90). Wire into run_data_pipeline.py P5 stage. Docstrings, type hints,
ruff/black clean.
```

### Prompt S13 — Metric suite

```
Implement src/fnb/evaluation/metrics.py per docs/final_protocol.md §8, docs/evaluation_protocol.md
Part 2, and docs/experiment_matrix.md §0.7. Provide: core_metrics(y_true, y_pred, y_prob) ->
dict of macro-F1 (primary), accuracy, macro-precision, macro-recall, per-class F1 (fake/real),
ROC-AUC, PR-AUC; ece(y_true, y_prob, n_bins=15) gated to probabilistic inputs; transfer_metrics
(in_domain_f1, cross_domain_f1) -> ΔF1 and std across targets; robustness_metrics(clean, attacked,
similarities) -> clean acc, acc-under-attack, ASR, mean similarity. Read bins/metric list from
configs/metrics.yaml. Use macro averaging everywhere; scores (not hard labels) for AUC. Write
tests/test_metrics.py with known-answer checks against sklearn and a hand-computed ECE example.
Docstrings, type hints, ruff/black clean.
```

### Prompt S14 — Statistical stack

```
Implement src/fnb/evaluation/stats.py per docs/final_protocol.md §10 and docs/evaluation_protocol.md
R6, reading params from configs/stats.yaml. Provide: paired_bootstrap_diff(y_true, pred_a, pred_b,
n=10000, stratified=True, seed) -> (mean_diff_macroF1, ci_low, ci_high, two_sided_p);
mcnemar(y_true, pred_a, pred_b); cliffs_delta(scores_a, scores_b); bh_fdr(pvalues, q=0.05) ->
adjusted p and reject flags; friedman_nemenyi(score_matrix) for exploratory ranking; and
assemble_confirmatory(family) applying BH-FDR ACROSS THE WHOLE family and returning per-comparison
(adjusted_p, effect_size, ci_of_difference). Enforce that a confirmatory claim requires the full
triple. Write tests/test_stats.py validating against statsmodels (BH-FDR), a hand McNemar example,
bootstrap CI coverage on synthetic data, and Cliff's delta signs; make the bootstrap seed-reproducible.
Docstrings, type hints, ruff/black clean.
```

### Prompt S15 — Result-row schema & writer

```
Implement src/fnb/evaluation/results_io.py per docs/evaluation_protocol.md R3/R4 and
docs/reproducibility_checklist.md §10. Define the canonical result-row schema with columns:
protocol_version (must be v1.0), dataset_version_tag, split_type, model_id, seed, run_id,
config_hash, git_sha, plus metric columns. Provide write_result_rows(csv_path, rows) that
validates the schema and APPENDS (never overwrites), and aggregate_seeds(df) producing mean ± std
per (model, dataset_version, split). Reject rows missing provenance columns. Write
tests/test_results_io.py: valid rows append; a row missing protocol_version is rejected;
aggregation matches manual mean ± std. Docstrings, type hints, ruff/black clean.
```

### Prompt S16 — Classical trainer

```
Implement src/fnb/models/classical.py per docs/final_protocol.md §5–§7 and EXP-1's classical
requirements. Build a TF-IDF pipeline (unigrams+bigrams, max_features=20000, min_df=5) FIT ON
TRAIN INDICES ONLY, feeding LogisticRegression, LinearSVM (with Platt scaling for probabilities),
RandomForest, and LightGBM. Read per-model grids from configs/classical_grids.yaml and select by
validation macro-F1. Return per-document (y_pred, y_prob) for val/test and serialize the fitted
vectorizer to artifacts/tfidf_{seed}.pkl. SMOTE is prohibited; never rebalance the test set. Write
tests/test_classical.py on a tiny synthetic dataset proving all four models train/predict with
probabilities and that TF-IDF is fit only on train indices. Docstrings, type hints, ruff/black clean.
```

### Prompt S17 — Encoder dataset & windowing

```
Implement src/fnb/training/datasets.py per docs/final_protocol.md §6.1. Create a torch Dataset over
vCLEAN-N text that tokenizes FULL documents into sliding windows (window=512, stride=128 from
configs/encoder.yaml), keeping a document->window index mapping so window logits can be mean-pooled
per document later. Support a title-only mode capped at max_len=64. Never truncate to a single 512
window (transformers must see the whole document). Add tokenization tests to tests/test_encoder.py:
a long document yields the expected window count at 512/128, a short document yields one window,
title-only caps at 64, and the document mapping is preserved. Docstrings, type hints, ruff/black clean.
```

### Prompt S18 — Custom training loop

```
Implement src/fnb/training/loop.py, scheduler.py, early_stopping.py, and the HF encoder wrapper in
src/fnb/models/encoder.py per docs/final_protocol.md §7. Write a HAND-CODED PyTorch training loop —
DO NOT use transformers.Trainer or accelerate's Trainer. Read all settings from configs/encoder.yaml:
AdamW (wd=0.01), lr 2e-5, batch 16, linear warmup 6% of steps -> cosine decay to 0, max 10 epochs,
early stopping on VALIDATION MACRO-F1 with patience 3, inverse-frequency class-weighted cross-entropy
(computed from train label frequencies). During validation, mean-pool window logits per document
(§6.1). Use fnb.utils.seeding for determinism. Return the checkpoint at best val macro-F1 with a
per-epoch log (loss, val macro-F1, LR trace, early-stop decision). Write tests/test_training_loop.py:
loss decreases on a tiny overfit set, scheduler decays to ~0, early stopping fires after patience 3,
class weights match train frequencies, and a grep-style assertion that no HF Trainer is imported.
Docstrings, type hints, ruff/black clean.
```

### Prompt S19 — Encoder inference & pooling

```
Extend src/fnb/models/encoder.py with predict(model, dataset) that runs in eval mode under no_grad,
computes per-window logits, mean-pools them per document using the mapping from
src/fnb/training/datasets.py, and returns per-document (y_pred, y_prob) via softmax. Pooling must be
IDENTICAL to the validation path in the training loop (pool LOGITS, not probabilities). Extend
tests/test_encoder.py: a document split into k windows yields one pooled logit equal to the mean of
its window logits, probabilities sum to 1, and results are deterministic under a fixed seed.
Docstrings, type hints, ruff/black clean.
```

### Prompt S20 — EXP-1 in-distribution + best_encoder

```
Implement src/fnb/experiments/exp1_indist.py per docs/experiment_matrix.md EXP-1 and §9, plus a
dispatch entry in scripts/run_experiment.py. Train and evaluate all 9 configs (LR, SVM, RF, LGBM,
BERT, ROBERTA, DEBERTA, DISTIL, ALBERT) on DS1 vDEDUP, split S-RAND, over the 5 primary seeds
{13,21,42,87,100}, reusing fnb.models.classical, fnb.models.encoder, and fnb.training.loop.
Compute METRICS-CORE + ECE via fnb.evaluation.metrics and write per-(model,seed) rows to
results/ds1_indist.csv via fnb.evaluation.results_io (full provenance, mean ± std aggregation).
Run pairwise encoder comparisons via fnb.evaluation.stats (STATS-CONF) -> results/ds1_encoder_pairwise_stats.csv
(H1). Select best_encoder with the deterministic tie-break: highest mean macro-F1, then lowest ECE,
then fewest parameters; write it to results/best_encoder.txt. Never report best-run-only.
Docstrings, type hints, ruff/black clean.
```

### Prompt S21 — EXP-P6 dedup sensitivity

```
Implement src/fnb/experiments/exp_p6_dedup_sens.py per docs/experiment_matrix.md EXP-P6 and a
dispatch entry. Re-run the best encoder (from results/best_encoder.txt) and LR on DS1 at dedup
thresholds {0.90,0.95,0.98} (from configs/dedup.yaml), 3 seeds, REUSING fnb.data.dedup (do not
reimplement dedup). Compute METRICS-CORE (mean ± std) and write results/dedup_sensitivity_ds1.csv
(threshold × model × seed) with full provenance. Docstrings, type hints, ruff/black clean.
```

### Prompt S22 — EXP-A1/A2/A3 transfer

```
Implement src/fnb/experiments/expA_transfer.py per docs/experiment_matrix.md EXP-A1/A2/A3,
docs/final_protocol.md §4.2, and a dispatch entry. EXP-A1: directed article-type transfer over
{DS1,DS2,DS3} with all 9 models, 5 seeds; off-diagonal tests MUST use the vXDEDUP(train->test)
cleaned indices from fnb.data.xdedup; reuse the DS1 in-domain diagonal from EXP-1. Compute
METRICS-CORE + TRANSFER (ΔF1, std across targets) and test ΔF1 via STATS-CONF (H2). Write
results/transfer_matrix.csv, results/transfer_deltaF1.csv, and results/transfer_family_exploratory.csv
(Friedman+Nemenyi, labeled exploratory). EXP-A2: domain-shift probe to DS4 (train {DS1,DS2,DS3},
test DS4 with vXDEDUP) -> results/domain_shift_ds4.csv. EXP-A3: LIAR (DS5) analyzed SEPARATELY,
never merged into article transfer -> results/liar_shortstatement.csv. Full provenance, mean ± std.
Docstrings, type hints, ruff/black clean.
```

### Prompt S23 — EXP-B leakage & source-only

```
Implement src/fnb/models/source_only.py and src/fnb/experiments/expB_leakage.py per
docs/experiment_matrix.md EXP-B1/B2/B3 and docs/final_protocol.md §11, plus a dispatch entry.
source_only.py: the SRCONLY shortcut baseline using source/metadata features where recoverable,
else surface features ONLY (length, punctuation rate, capitalization rate) with NO article text.
expB_leakage.py on DS1, best encoder + LR, 5 seeds: EXP-B1 before-vs-after dedup delta (H3 part 1);
EXP-B2 random vs source-disjoint split delta (H3 part 2) — if source is unrecoverable, record
"not applicable" with an honest note and report only the dedup contrast; EXP-B3 source-only vs
full-text macro-F1 ratio (H4, >=0.80 check) via STATS-CONF. Write all rows to
results/leakage_analysis.csv with full provenance and the source-recoverability decision.
Docstrings, type hints, ruff/black clean.
```

### Prompt S24 — EXP-C1 adversarial

```
Implement src/fnb/adversarial/attacks.py and src/fnb/experiments/expC_adversarial.py per
docs/experiment_matrix.md EXP-C1 and docs/final_protocol.md §12, plus a dispatch entry. Using
TextAttack, build LABEL-PRESERVING recipes from configs/adversarial.yaml: character-level typos,
embedding-constrained synonym substitution, benign word insertion, and back-translation paraphrase.
PROHIBIT negation and antonym substitution. Enforce a Universal Sentence Encoder similarity >= 0.90
constraint; discard attacks below it. Apply to the DS1 vDEDUP TEST set for all encoders + classical
+ (budget permitting) MISTRAL and LLAMA, evaluating on their FIXED saved checkpoints (no retraining);
use an identical attack recipe and a fixed per-example query cap for every model for budget-matched
comparison. Compute ROBUST metrics (clean acc, acc-under-attack, ASR, mean similarity of successful
attacks) per model/family and write results/adversarial_robustness.csv. Save the perturbation sets
(documented as defensive, §16). Docstrings, type hints, ruff/black clean.
```

### Prompt S25 — EXP-D1/D2 efficiency

```
Implement src/fnb/experiments/expD_efficiency.py and Pareto plotting in src/fnb/analysis/figures.py
per docs/experiment_matrix.md EXP-D1/D2 and docs/final_protocol.md §14, plus a dispatch entry.
Profile all encoders + LR/LGBM references + DISTIL + QUANT-BEST (dynamic INT8 of the best encoder)
on DS1 vDEDUP, 3 seeds: parameter count, on-disk size (MB), peak VRAM, training wall-clock,
inference latency (ms/sample at batch=1, seq=512), throughput (samples/s at batch=32). ALL timing
MUST be tagged as measured on the single fixed NVIDIA T4; exclude any non-T4 run from efficiency
tables. Compute the H6 ratio (compact model >= 95% F1 at < 50% params) and build F1-vs-params and
F1-vs-latency Pareto frontiers. Write results/efficiency.csv and results/compression_study.csv
(optional INT8 + KD trade-off). Docstrings, type hints, ruff/black clean.
```

### Prompt S26 — EXP-E1 granularity

```
Implement src/fnb/experiments/expE_granularity.py per docs/experiment_matrix.md EXP-E1 and
docs/final_protocol.md §6.1, plus a dispatch entry. Run title-only / content / title+content
conditions on {DS1,DS2,DS3} ONLY (datasets with both fields) for all 5 encoders (+ classical for
completeness), 3 seeds; title-only uses HP-TITLE (max_len 64). Compute METRICS-CORE per condition
and test ranking stability with Friedman + Nemenyi, labeled EXPLORATORY (not FDR-protected). Write
results/granularity.csv and a ranking-stability table. Docstrings, type hints, ruff/black clean.
```

### Prompt S27 — EXP-F0/F1/F2 LLM case study

```
Implement src/fnb/models/llm_qlora.py and src/fnb/experiments/expF_llm.py per
docs/experiment_matrix.md EXP-F0/F1/F2 and docs/final_protocol.md §4.7, §5, §7, plus a dispatch
entry. Models: MISTRAL (Mistral-7B-Instruct-v0.2) and LLAMA (Llama-3.1-8B-Instruct). EXP-F0:
guided prefix-completion contamination probe on DS1–DS4 -> results/llm_contamination.csv; flagged
datasets are excluded from that LLM's zero-shot claims (gate). EXP-F1: 4-bit QLoRA fine-tune on
DS1 vDEDUP using configs/llm_qlora.yaml (r=64, alpha=16, dropout=0.2, NF4, double-quant, lr 2e-4,
effective batch 16 via gradient accumulation), 3 seeds, capped at 20 GPU-h total (mark PARTIAL if
exceeded); METRICS-CORE -> results/llm_finetune_ds1.csv; save QLoRA adapters. EXP-F2: zero-shot and
5-shot inference with fixed templates -> results/llm_prompted.csv in a SEPARATE table, labeled
contamination-suspect, EXCLUDING ECE. Never merge prompted with fine-tuned results. Reuse the
custom loop where applicable (no HF Trainer). Docstrings, type hints, ruff/black clean.
```

### Prompt S28 — EXP-G1 calibration

```
Implement src/fnb/analysis/calibration.py and reliability-diagram plotting in
src/fnb/analysis/figures.py per docs/experiment_matrix.md EXP-G1 and docs/final_protocol.md §8,
plus a dispatch entry. Compute ECE (15 bins, via fnb.evaluation.metrics) for ALL probabilistic
models on DS1 vDEDUP test and transfer targets; EXCLUDE LLM zero-shot. Write results/calibration.csv
and reliability plots. Docstrings, type hints, ruff/black clean.
```

### Prompt S29 — EXP-G2 error analysis

```
Implement src/fnb/analysis/error_analysis.py and create docs/error_coding_scheme.md per
docs/experiment_matrix.md EXP-G2 and docs/final_protocol.md §13, plus a dispatch entry. For the
best encoder + LR: per-dataset confusion matrices, fake->real vs real->fake asymmetry, a WRITTEN
failure-coding scheme (sarcasm/satire, ambiguous/figurative, atypical source, drift) in
docs/error_coding_scheme.md, a sample of >=200 errors flagged for double-coding where feasible,
difficulty buckets (DS2 easy / DS5 hard), and performance vs article length. Write
results/error_analysis.csv and figures. Docstrings, type hints, ruff/black clean.
```

### Prompt S30 — EXP-G3 interpretability

```
Implement src/fnb/analysis/interpretability.py and attribution figures in
src/fnb/analysis/figures.py per docs/experiment_matrix.md EXP-G3 and docs/final_protocol.md §13,
plus a dispatch entry. On the best encoder, compute Integrated Gradients (use captum) and
leave-one-token occlusion attributions, plus comprehensiveness and sufficiency faithfulness metrics.
Attention weights are ILLUSTRATIVE ONLY and must never be presented as the sole explanation. Write
results/interpretability.csv and example attribution figures. Docstrings, type hints, ruff/black clean.
```

### Prompt S31 — EXP-G4 confirmatory stats

```
Implement src/fnb/experiments/expG_confirmatory.py per docs/experiment_matrix.md EXP-G4 and §9, and
docs/final_protocol.md §10, plus a dispatch entry. Assemble the PRE-REGISTERED confirmatory family
EXACTLY as matrix §9: (1) pairwise encoder macro-F1 on DS1 in-distribution (from EXP-1), (2) each
model's in-domain-vs-cross-domain ΔF1 (from EXP-A1), (3) full-text vs source-only plus dedup and
split-regime deltas (from EXP-B1/B2/B3). Apply Benjamini–Hochberg FDR at q=0.05 ACROSS THE WHOLE
family (once), attach Cliff's delta and 95% CI of the difference per comparison, and emit H1–H4
verdicts. Write results/confirmatory_stats.csv. Do not include any exploratory comparison in the
family; never issue a verdict from a raw delta. Docstrings, type hints, ruff/black clean.
```

### Prompt S32 — Paper outputs & reproducibility gate

```
Implement scripts/build_paper_outputs.py and extend src/fnb/analysis/figures.py per
docs/experiment_matrix.md §12 and docs/reproducibility_checklist.md "Pre-report Gate". Generate,
from results/*.csv ONLY (never hand-edited numbers, always mean ± std with provenance +
protocol_version), the paper-ready tables and figures into results/paper/: main in-distribution
table, transfer table + ΔF1, leakage table, adversarial robustness table, efficiency table +
Pareto plots, calibration table, and the confirmatory H1–H4 verdict table. Then run the 10-point
reproducibility pre-report gate and emit a pass/fail report, checking seeds/determinism, complete
ENVIRONMENT.md, dataset hashes + version tags + dedup/overlap counts + dataset_stats.csv, committed
split indices + natural-distribution test sets, tokenizer/vectorizer revisions, checkpoints +
adapters + metadata, configs-only value sourcing, complete logs, tracker rows with git commit +
config hash, and protocol_version=v1.0 stamped on every result row. Update ENVIRONMENT.md and
README.md with final pinned versions and reproduction steps. Docstrings, type hints, ruff/black clean.
```

---

# 6. Validation Checklist (per milestone)

### M1 — Project Foundation
- **What to test:** package install/import; all ten configs parse and validate; seeding
  reproducibility; logging/hashing/run-registry smoke.
- **How to test:** `pip install -e .` + `python -c "import fnb"`; `pytest tests/test_config.py
  tests/test_seeding.py`; run a registry smoke script.
- **Expected outputs:** version prints; tests green; `logs/{run_id}.log`, `results/runs.csv`
  row, `metadata.json` created; stable config + file hashes.
- **Failure conditions:** import errors; any frozen value missing/mismatched in a config; two
  same-seed runs diverge; missing provenance fields; determinism flags not set.

### M2 — Data Pipeline
- **What to test:** snapshot hashes stable; binary label space + LIAR map; both preprocessing
  tracks; within-dataset dedup at 0.95; 70/15/15 stratified splits (indices only); dataset stats;
  cross-dataset overlap removal at 0.90.
- **How to test:** `python scripts/run_data_pipeline.py` (P0→P5); `pytest tests/test_preprocess.py
  tests/test_dedup.py tests/test_splits.py`.
- **Expected outputs:** `data/SNAPSHOT_HASHES.txt`; `data/processed/{DSx}_{vBIN,vCLEAN-C,vCLEAN-N,
  vDEDUP}.parquet`; `data/splits/*.idx`; `results/{label_mapping_report,dedup_counts,dataset_stats,
  interdataset_overlap}.csv`; source-recoverability + temporal-applicability notes.
- **Failure conditions:** non-reproducible hashes; labels outside {0,1}; neural track altered by
  stopword/lemmatization; TF-IDF fit before split; split ratios off; N/A silently skipped;
  test-side (not train-side) items not removed in cross-dedup.

### M3 — Evaluation & Statistics Core
- **What to test:** metric correctness vs sklearn; ECE gating; bootstrap/McNemar/Cliff's/BH-FDR vs
  references; result-row schema + mean ± std aggregation.
- **How to test:** `pytest tests/test_metrics.py tests/test_stats.py tests/test_results_io.py`.
- **Expected outputs:** green tests; a sample result CSV with full provenance.
- **Failure conditions:** micro instead of macro averaging; ECE computed for non-probabilistic
  models; BH-FDR applied per-test instead of family-wide; a row missing `protocol_version`.

### M4 — Model & Training Infrastructure
- **What to test:** classical models train with train-only TF-IDF + probabilities; sliding-window
  tokenization; custom loop honors §7 and uses NO HF Trainer; windowed inference pooling parity.
- **How to test:** `pytest tests/test_classical.py tests/test_encoder.py tests/test_training_loop.py`;
  a 1-epoch smoke train of one encoder on a small DS1 sample.
- **Expected outputs:** green tests; smoke checkpoint + per-epoch log with val macro-F1 + LR trace.
- **Failure conditions:** TF-IDF fit on all data; SMOTE used; truncation instead of windowing; HF
  Trainer imported; early stopping keyed off accuracy; validation/test pooling mismatch.

### M5 — Primary Experiments (RQ-A/B/C)
- **What to test:** EXP-1 (9 models × 5 seeds) + best_encoder; EXP-P6 sensitivity; EXP-A transfer +
  DS4 probe + LIAR track; EXP-B leakage/source-only; EXP-C1 adversarial validity.
- **How to test:** `python scripts/run_experiment.py --exp EXP-1` (then P6, A*, B*, C1).
- **Expected outputs:** `results/ds1_indist.csv`, `ds1_encoder_pairwise_stats.csv`,
  `best_encoder.txt`, `dedup_sensitivity_ds1.csv`, `transfer_matrix.csv`, `transfer_deltaF1.csv`,
  `transfer_family_exploratory.csv`, `domain_shift_ds4.csv`, `liar_shortstatement.csv`,
  `leakage_analysis.csv`, `adversarial_robustness.csv`.
- **Failure conditions:** best-run-only reporting; off-diagonal transfer without cross-dedup; LIAR
  merged into article transfer; source-disjoint claimed when unrecoverable; counted attacks below
  USE 0.90; fewer than 5 seeds on a primary endpoint.

### M6 — Secondary Experiments (RQ-D/E/F)
- **What to test:** efficiency on fixed T4; granularity on {DS1,DS2,DS3}; LLM contamination gate +
  QLoRA fine-tune + separate prompted table; 20 GPU-h cap.
- **How to test:** `python scripts/run_experiment.py --exp EXP-D1` (then E1, F0→F1→F2).
- **Expected outputs:** `efficiency.csv`, `compression_study.csv`, `granularity.csv`,
  `llm_contamination.csv`, `llm_finetune_ds1.csv`, `llm_prompted.csv`, QLoRA adapters, Pareto plots.
- **Failure conditions:** mixed accelerators in efficiency tables; granularity on DS4/DS5; prompted
  merged with fine-tuned; ECE on zero-shot LLM; RQ-F over 20 GPU-h without a PARTIAL mark.

### M7 — Analysis & Interpretability
- **What to test:** calibration eligibility; error analysis + coding scheme; interpretability
  faithfulness; confirmatory family assembly + family-wide BH-FDR.
- **How to test:** `python scripts/run_experiment.py --exp EXP-G1` (then G2, G3, G4).
- **Expected outputs:** `calibration.csv`, `error_analysis.csv`, `docs/error_coding_scheme.md`,
  `interpretability.csv`, `confirmatory_stats.csv` with H1–H4 verdicts.
- **Failure conditions:** LLM zero-shot in ECE; attention used as sole explanation; exploratory
  comparison inside the confirmatory family; verdict from a raw delta.

### M8 — Paper Outputs & Final Verification
- **What to test:** every paper table/figure regenerates from CSVs; reproducibility gate all-green.
- **How to test:** `python scripts/build_paper_outputs.py`.
- **Expected outputs:** `results/paper/*`; a passing 10-point gate report; complete `ENVIRONMENT.md`.
- **Failure conditions:** any hand-edited number; missing provenance/version stamp; any gate item red.

---

# 7. Git Workflow (per milestone)

Conventions: author and commit in **Cursor**, push to the Git remote; **Kaggle** notebooks
`git clone`/pull that remote to execute. One feature branch per milestone
(`milestone/M1-foundation`, …); commit per step (`feat(Sx): …`) with tests passing (tests are run
on Kaggle and their output pasted back before you consider the step done); open a PR into `main` at
milestone end; tag a release when a milestone leaves the project in a verified runnable state.
Never commit large data — raw bytes stay in `/kaggle/input/`, and heavy outputs
(`processed/*.parquet`, checkpoints, QLoRA adapters) are persisted as **Kaggle output datasets**,
not git. The repo commits only code, configs, `SNAPSHOT_HASHES.txt`, split indices, and result
CSVs (small). Enforce a clean working tree before any experiment run; the Kaggle notebook records
the cloned commit SHA into each run's metadata so the executed code version is provenance-tracked.

**Cursor ⇄ Kaggle loop per step:** (1) generate/edit code in Cursor → commit + push; (2) on Kaggle,
pull the commit, run the step's cell(s); (3) paste the resulting output/CSV/log back into Cursor;
(4) commit the produced small artifacts (CSVs, hashes) and, if needed, publish a new Kaggle output
dataset version for heavy artifacts the next step will consume.

| Milestone | Files expected in the milestone PR | Suggested squash/merge commit message | Tag when |
|:--:|------|------|------|
| M1 | `pyproject.toml`, `requirements*.txt`, `README.md`, `ENVIRONMENT.md`, `src/fnb/{config,utils}/*`, `configs/*.yaml`, `tests/test_{config,seeding}.py` | `feat(M1): project foundation — package, frozen configs, seeding, logging, run registry` | `v0.1.0-foundation` |
| M2 | `src/fnb/data/*`, `scripts/run_data_pipeline.py`, `data/SNAPSHOT_HASHES.txt`, `data/splits/*.idx`, `results/{label_mapping_report,dedup_counts,dataset_stats,interdataset_overlap}.csv`, `tests/test_{preprocess,dedup,splits}.py` | `feat(M2): data pipeline EXP-P0..P5 — snapshots, binarize, preprocess, dedup, splits, stats, cross-dedup` | `v0.2.0-data` |
| M3 | `src/fnb/evaluation/{metrics,stats,results_io}.py`, `configs/{metrics,stats}.yaml`, `tests/test_{metrics,stats,results_io}.py` | `feat(M3): evaluation core — metric suite, statistical stack, provenance result schema` | `v0.3.0-eval` |
| M4 | `src/fnb/models/{classical,encoder}.py`, `src/fnb/training/*`, `tests/test_{classical,encoder,training_loop}.py` | `feat(M4): model+training infra — classical, sliding-window encoders, custom PyTorch loop` | `v0.4.0-infra` |
| M5 | `src/fnb/experiments/{exp1_indist,exp_p6_dedup_sens,expA_transfer,expB_leakage,expC_adversarial}.py`, `src/fnb/models/source_only.py`, `src/fnb/adversarial/attacks.py`, `configs/adversarial.yaml`, primary result CSVs, `results/best_encoder.txt` | `feat(M5): primary experiments RQ-A/B/C — in-dist, transfer, leakage, adversarial` | `v0.5.0-primary` |
| M6 | `src/fnb/experiments/{expD_efficiency,expE_granularity,expF_llm}.py`, `src/fnb/models/llm_qlora.py`, `configs/llm_qlora.yaml`, secondary result CSVs, QLoRA adapters | `feat(M6): secondary experiments RQ-D/E/F — efficiency, granularity, LLM case study` | `v0.6.0-secondary` |
| M7 | `src/fnb/analysis/{calibration,error_analysis,interpretability,figures}.py`, `src/fnb/experiments/expG_confirmatory.py`, `docs/error_coding_scheme.md`, analysis result CSVs | `feat(M7): analysis — calibration, error analysis, interpretability, confirmatory stats` | `v0.7.0-analysis` |
| M8 | `scripts/build_paper_outputs.py`, `results/paper/*`, final `ENVIRONMENT.md`, `README.md` | `feat(M8): paper outputs + reproducibility gate — tables, figures, verification` | `v1.0.0-framework` (framework complete, experiment-ready) |

Tag `v1.0.0-framework` marks the frozen, verified codebase used to run the full experiment sweep.
Any post-run bug fix bumps a patch tag and requires re-running affected experiments.

---

# 8. Final Verification

The framework is experiment-ready ONLY when ALL of the following are true (this is the
`docs/reproducibility_checklist.md` pre-report gate, made concrete):

1. **Install & tests (on Kaggle):** in a Kaggle notebook, `git clone` + `pip install -e .` clean;
   `pytest -q` fully green; `ruff`/`black` clean — paste the notebook output back into Cursor as
   evidence.
2. **Configs:** all ten `configs/*.yaml` validate; every frozen value matches `final_protocol.md`
   v1.0; no module hard-codes a frozen constant (grep audit); `protocol_version=v1.0` everywhere.
3. **Determinism:** `set_global_seed` seeds all RNGs; deterministic algorithms + `CUBLAS_WORKSPACE_CONFIG`
   set; same-seed runs reproduce; irreducible nondeterminism documented in `ENVIRONMENT.md`.
4. **Data integrity:** `data/SNAPSHOT_HASHES.txt` present and stable; FakeNewsNet hydrated IDs +
   date + recovery fraction recorded; `label_mapping_report.csv`, `dedup_counts.csv`,
   `interdataset_overlap.csv`, `dataset_stats.csv` all present.
5. **Splits:** committed index files for every dataset × regime × seed; 70/15/15 stratified;
   natural-distribution test sets confirmed; S-SRC/S-TEMP N/A cases recorded, never silently skipped.
6. **Models/training:** custom loop verified (no HF Trainer); sliding-window full-document exposure;
   TF-IDF train-only; SMOTE absent; class-weighted CE; early stopping on val macro-F1 (patience 3);
   checkpoints + base-model revisions + QLoRA adapters saved with `metadata.json`.
7. **Evaluation:** METRICS-CORE + ECE gating verified; STATS-CONF matches reference libraries;
   BH-FDR applied family-wide; every confirmatory row carries (adjusted p, Cliff's δ, CI of difference).
8. **Reporting discipline:** every result CSV row carries full provenance and mean ± std across the
   mandated seeds; no best-run-only anywhere; no borrowed literature numbers as our results.
9. **Budget:** compute-budget ledger reconciled against ≤120 T4 GPU-h (RQ-F ≤20 GPU-h); efficiency
   numbers exclusively from the single fixed T4.
10. **Deliverables:** every CSV in `docs/experiment_matrix.md` §12 is regenerable end-to-end from
    `scripts/`; `results/paper/*` builds from CSVs only; the 10-point gate report is all-green.

Only when items 1–10 are green may experiment numbers be reported or the paper's Results written.

---

# 9. Experiment Phase

Run ONLY after Final Verification (§8) is fully green. **All execution happens in Kaggle
notebooks on the single T4**; Cursor is used only to author code and to receive pasted-back
outputs. Execute in the frozen dependency order from `docs/experiment_matrix.md` §10. Primary
endpoints first; secondary only if within budget. Because Kaggle sessions are ephemeral and
time-boxed (≈9–12 h) with a weekly GPU quota, each experiment is run in its own session and MUST
persist `/kaggle/working/` (results CSVs, checkpoints, adapters) as a Kaggle output dataset; the
next experiment attaches that output as an input to resume. The ≤120 GPU-h budget (protocol §1) is
therefore spread across multiple sessions.

**Execution order (one experiment at a time; record GPU-h per run into the budget ledger):**

1. **Data pipeline:** `python scripts/run_data_pipeline.py` → P0 → P1 → P2 → P3 → P4 → P5.
   Confirm all integrity CSVs before any model runs (they gate transfer/leakage validity).
2. **EXP-1** (in-distribution, DS1, 9 models × 5 seeds) → `ds1_indist.csv`,
   `ds1_encoder_pairwise_stats.csv`, `best_encoder.txt`. **This freezes `best_encoder`** for all
   downstream RQ-C/D/G steps.
3. **EXP-P6** (dedup sensitivity, DS1) → `dedup_sensitivity_ds1.csv`.
4. **Phase 2 — RQ-A:** EXP-A1 (transfer matrix), EXP-A2 (DS4 probe), EXP-A3 (LIAR track).
5. **Phase 3 — RQ-B:** EXP-B1 (dedup contrast), EXP-B2 (split-regime), EXP-B3 (source-only).
6. **Phase 4 — RQ-C:** EXP-C1 (adversarial; uses EXP-1 and — if run — EXP-F1 checkpoints).
7. **Primary analysis:** EXP-G1 (calibration), EXP-G2 (error analysis), EXP-G4 (confirmatory stats
   → H1–H4 verdicts). These complete the primary story.
8. **Secondary (budget-permitting, cut order F→E→D):** EXP-D1/D2 (efficiency), EXP-E1 (granularity),
   EXP-F0→F1→F2 (LLM case study, ≤20 GPU-h else PARTIAL), EXP-G3 (interpretability).

**Run rules (enforced by the framework, restated here):**
- Every run executes in a Kaggle notebook cell as `!python scripts/run_experiment.py --exp EXP-XX
  --seed S` after cloning a clean commit; the run registry stamps the cloned commit SHA + config
  hash + the Kaggle dataset versions attached. Paste the resulting CSV/log back into Cursor and
  persist `/kaggle/working/` as a Kaggle output dataset.
- Primary endpoints use 5 seeds `{13,21,42,87,100}`; secondary use 3 seeds `{13,42,100}`.
- All efficiency/timing runs execute on the single fixed Kaggle NVIDIA T4; other GPUs may produce
  accuracy numbers but are excluded from efficiency tables.
- Report mean ± std; never best-run-only. Contamination-flagged datasets are excluded from that
  LLM's zero-shot claims. Prompted LLM results stay in a separate, clearly-labeled table.
- If projected total exceeds 120 GPU-h, cut secondary phases in order F → E → D; primary endpoints
  are never cut. Any cut is recorded in the budget ledger and reported as such.

Output of this phase: every CSV in `docs/experiment_matrix.md` §12, each row carrying
`protocol_version=v1.0` + dataset-version tag + split + model + seed.

---

# 10. Paper Phase

Write the paper's empirical sections ONLY from the generated artifacts. Each Results subsection maps
1:1 to result CSVs and figures produced above (build them with
`python scripts/build_paper_outputs.py` → `results/paper/`).

| Paper section | Source artifact(s) | Reported content | Hypothesis / RQ |
|---------------|--------------------|------------------|-----------------|
| Datasets & Setup | `dataset_stats.csv`, `SNAPSHOT_HASHES.txt`, `label_mapping_report.csv`, `dedup_counts.csv`, `interdataset_overlap.csv` | dataset sizes/balances, snapshot hashes, label mapping, dedup/overlap counts | Reproducibility (protocol §4, §15) |
| In-Distribution Results | `ds1_indist.csv`, `ds1_encoder_pairwise_stats.csv`, `confirmatory_stats.csv` | per-model macro-F1 (mean ± std) + full suite; encoder equivalence verdict | **H1** (RQ4) |
| Cross-Dataset Generalization | `transfer_matrix.csv`, `transfer_deltaF1.csv`, `domain_shift_ds4.csv`, `transfer_family_exploratory.csv` | ΔF1 per model; DS4 probe; exploratory ranking | **H2** (RQ-A) |
| Short-Statement Track | `liar_shortstatement.csv` | LIAR results reported separately (never merged) | RQ-A control (§4.2) |
| Leakage & Shortcut Audit | `leakage_analysis.csv`, `dedup_sensitivity_ds1.csv` | before/after dedup, split-regime, source-only/full ratio | **H3, H4** (RQ-B) |
| Adversarial Robustness | `adversarial_robustness.csv` | clean vs attacked, ASR, similarity per family | **H5** (RQ-C) |
| Efficiency / Pareto | `efficiency.csv`, `compression_study.csv`, Pareto plots | params/size/VRAM/latency/throughput; F1-vs-compute frontier | **H6** (RQ-D) |
| Input Granularity | `granularity.csv` + ranking table | title/content/title+content ranking (exploratory) | E2 (RQ-E) |
| LLM Case Study | `llm_finetune_ds1.csv`, `llm_prompted.csv`, `llm_contamination.csv` | QLoRA fine-tuned vs encoders; prompted (separate, contamination-suspect) | RQ-F, E3 |
| Calibration | `calibration.csv` + reliability diagrams | ECE per probabilistic model | support RQ-D/E |
| Error Analysis | `error_analysis.csv`, `docs/error_coding_scheme.md` | confusion matrices, failure taxonomy, length effects | support (§13) |
| Interpretability | `interpretability.csv` + attribution figures | IG/occlusion faithfulness (comprehensiveness/sufficiency) | support (§13) |
| Statistical Verdicts | `confirmatory_stats.csv` | family-wide BH-FDR verdicts with (adjusted p, δ, CI) for H1–H4 | RQ-A/B, H1–H4 |
| Threats / Limitations / Ethics | protocol §16, §17; scoping notes | English-only, text-only scope; leakage caveats; dual-use statement | Broader Impact |

**Writing rules:**
- Every table/figure is generated by `build_paper_outputs.py`; no number is hand-entered.
- Every reported value is mean ± std over the mandated seeds with the dataset-version tag and
  `protocol_version=v1.0` in the caption.
- Every confirmatory claim states the triple (BH-adjusted p, Cliff's delta, 95% CI of the
  difference); exploratory results are labeled exploratory and never given confirmatory language.
- Clean and attacked results appear together; efficiency numbers cite the fixed T4; LLM zero-/5-shot
  appear only in a separate, clearly-labeled table.
- Primary endpoints (RQ-A/B/C) are reported in full even if secondary endpoints were cut for budget;
  any cut is stated explicitly.

This completes the master implementation manual. Follow it in order, milestone by milestone, with
exactly one next step at any time, until the paper's Results section is fully populated from the
generated artifacts.
