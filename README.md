# fnb — Fake News Benchmark

A reproducible, cross-dataset **fake-news detection benchmark** spanning three model
families — classical TF-IDF baselines, transformer **encoders** (sliding-window), and
4-bit **QLoRA**-fine-tuned LLMs — evaluated under a single frozen protocol with rigorous
statistics (bootstrap CIs, McNemar, Cliff's δ, BH-FDR).

- **What to build:** `MASTER_IMPLEMENTATION_GUIDE.md` (the implementation constitution).
- **How to run it (operator runbook):** `KAGGLE_WORKFLOW.md`.
- **Environment / reproducibility:** `ENVIRONMENT.md`.

This README is the quickstart: how to install and run each milestone on **Kaggle (single
NVIDIA T4)**, which is the sole execution target. Code is *authored in Cursor*, pushed to
GitHub, and *executed on Kaggle*.

---

## 1. Package layout (src-layout)

```
FakeNewsBenchmark/
├── src/fnb/            # the installable package (import as `fnb`)
│   ├── config/         # config loading + schema validation + typed constants
│   ├── utils/          # seeding, logging, hashing, run registry, IO
│   ├── data/           # acquire, binarize, preprocess, dedup, splits, stats, xdedup
│   ├── models/         # classical, encoder, source_only, llm_qlora
│   ├── training/       # torch datasets, custom loop, scheduler, early stopping
│   ├── evaluation/     # metrics + statistics
│   ├── adversarial/    # TextAttack recipes (label-preserving)
│   ├── analysis/       # calibration, error analysis, interpretability, figures
│   └── experiments/    # EXP-* drivers
├── scripts/            # thin argparse CLIs (orchestration only, no research logic)
├── notebooks/          # Kaggle T4 runner notebook (thin wrapper over scripts/)
├── configs/            # FROZEN protocol mirrors (YAML) — code reads values ONLY from here
├── data/               # raw (kept on Kaggle) / processed / splits
├── results/, artifacts/, logs/   # reproducible outputs
├── docs/, literature/  # FROZEN planning + literature
├── pyproject.toml      # packaging (src-layout) + tool config (ruff/black/pytest)
├── requirements.txt    # pinned '==' dependencies (single source of truth)
└── ENVIRONMENT.md      # pinned libs, CUDA, Kaggle T4 hardware
```

The package is installed **editable** with `pip install -e .`. Dependency versions are
pinned in `requirements.txt` and consumed by `pyproject.toml` via dynamic metadata, so the
single install command reproduces the pinned environment.

Console entry points (installed by `pip install -e .`) mirror the scripts:

| Command | Script |
|---|---|
| `fnb-data-pipeline` | `scripts/run_data_pipeline.py` |
| `fnb-run-experiment` | `scripts/run_experiment.py` |
| `fnb-build-paper` | `scripts/build_paper_outputs.py` |

---

## 2. Kaggle flow (the only execution target)

Code lives on GitHub; Kaggle runs it. The loop is: **Cursor → GitHub → Kaggle**, then paste
outputs back. Full operator detail is in `KAGGLE_WORKFLOW.md`; the essentials:

### 2.1 One-time notebook setup
1. Kaggle **Settings → Phone verification** (unlocks GPU + Internet).
2. **Create → New Notebook**. In the right panel:
   - **Accelerator → GPU T4 x1** for training experiments; **None/CPU** for tests and the
     data pipeline (to save GPU quota).
   - **Internet → On** (required for `pip install` and model downloads).
3. **Add Data** → attach each input dataset (DS1–DS5). They mount read-only under
   `/kaggle/input/<slug>/`.

### 2.2 First notebook cell — clone + install
The **first cell of every session** pulls the latest code and installs the package:

```python
# Cell 1 — clone the repo and install the package (editable)
!git clone https://github.com/<your-username>/fake-news-benchmark.git
%cd fake-news-benchmark
!pip install -e . -q
# sanity check
!python -c "import fnb; print('fnb', fnb.__version__)"
```

> Private repo? Use a fine-grained read-only token:
> `!git clone https://<TOKEN>@github.com/<your-username>/fake-news-benchmark.git`

### 2.3 `/kaggle/input` (read-only) vs `/kaggle/working` (outputs)

| Path | Access | Purpose |
|---|---|---|
| `/kaggle/input/<slug>/` | **read-only** | Attached input datasets (raw DS1–DS5, plus any output datasets you re-attach). Raw bytes live *only* here — never committed to git. |
| `/kaggle/working/` | read-write | All run outputs (processed parquet, checkpoints, adapters, result CSVs, logs). **Wiped when the session ends.** |

Because `/kaggle/working/` is ephemeral, heavy outputs must be **saved as a Kaggle output
dataset** (right panel → *Save Version* / *Create Dataset*) so the next session can re-attach
them under `/kaggle/input/`. This is how, e.g., adversarial `EXP-C1` reuses `EXP-1`
checkpoints without retraining, staying within the ≤120 GPU-h budget.

Small text artifacts (result CSVs, `SNAPSHOT_HASHES.txt`, split `*.idx`, logs) are brought
back into the repo and committed; large binaries stay on Kaggle. See `KAGGLE_WORKFLOW.md` §4–§5
for the exact mapping.

### 2.4 Running a stage / experiment
After Cell 1, dispatch via the scripts (or their console entry points):

```python
# Unit tests (CPU accelerator — cheap)
!pytest -q

# A data-pipeline stage (CPU)
!python scripts/run_data_pipeline.py --stage all      # equivalently: !fnb-data-pipeline --stage all

# A training experiment (GPU T4)
!python scripts/run_experiment.py --exp EXP-1 --seed 13   # equivalently: !fnb-run-experiment --exp EXP-1 --seed 13

# Paper tables + figures (CPU)
!python scripts/build_paper_outputs.py                # equivalently: !fnb-build-paper
```

The preferred, self-documenting way to run all of the above is the **runner notebook**,
`notebooks/kaggle_runner.ipynb`, which contains ready-made cells for clone+install, attaching
data, dispatching a stage/experiment, and saving `/kaggle/working` as a versioned dataset.

---

## 3. Milestone run order (M1 → M8)

The roadmap is strictly linear; each milestone ends with a runnable project and depends only
on previous milestones (see `MASTER_IMPLEMENTATION_GUIDE.md` §3).

| # | Milestone | Steps | Depends on | Produces | Typical accelerator |
|:--:|---|:--:|:--:|---|:--:|
| **M1** | Project Foundation | S1–S5 | — | Installable package, configs, determinism, logging, run registry | CPU |
| **M2** | Data Pipeline (EXP-P0–P5) | S6–S12 | M1 | Snapshots+hashes, binarized/preprocessed/deduped datasets, splits, stats, cross-dedup | CPU |
| **M3** | Evaluation & Statistics Core | S13–S15 | M1 | Metric suite, statistical stack, result-row schema/writer | CPU |
| **M4** | Model & Training Infrastructure | S16–S19 | M2, M3 | Classical trainer, encoder + sliding-window, custom training loop, inference | GPU T4 |
| **M5** | Primary Experiments (RQ-A/B/C) | S20–S24 | M4 | EXP-1, EXP-P6, EXP-A*, EXP-B*, EXP-C1 result CSVs | GPU T4 |
| **M6** | Secondary Experiments (RQ-D/E/F) | S25–S27 | M5 | EXP-D*, EXP-E1, EXP-F* result CSVs | GPU T4 |
| **M7** | Analysis & Interpretability | S28–S31 | M5 (M6 where applicable) | Calibration, error analysis, interpretability, confirmatory stats | GPU/CPU |
| **M8** | Paper Outputs & Final Verification | S32 | M5–M7 | Paper tables, figures, full reproducibility gate pass | CPU |

**Priority (protocol §1):** M5 and the primary parts of M7 (G1/G2/G4) are the primary
endpoints and are never cut. M6 (secondary D/E/F) runs only within the ≤120 GPU-h budget
(RQ-F capped at 20 GPU-h); if budget is threatened, cut secondary phases in order **F → E → D**.

---

## 4. Local development (Cursor)

Authoring happens in Cursor; heavy GPU work is Kaggle-only. For a lightweight local checkout:

```bash
python -m venv .venv && . .venv/Scripts/activate   # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -e '.[dev]'      # or: pip install -e .
pytest -q                    # 0 tests until they land in Milestone 1+
ruff check . && black --check .
```

> The pinned `requirements.txt` includes GPU libraries (torch, bitsandbytes, …) that are
> heavy or CUDA-specific. For pure authoring you generally do not need a full local install;
> rely on Kaggle for execution and reproducibility. See `ENVIRONMENT.md`.

---

## 5. Reproducibility contract (summary)

- **Frozen constants** live only in `configs/*.yaml`; no value is hard-coded in code.
- **Determinism:** a single global seed seeds every RNG and enables deterministic algorithms.
- **Provenance:** every run records git SHA, resolved config hash, seed, library + CUDA
  versions, hardware, dataset-version tag, and wall-clock into `results/runs.csv` + a per-run
  `logs/{run_id}.log` and `metadata.json`.
- **Data hygiene:** raw bytes never enter git; only hashes, split indices, and small result
  CSVs are committed. The pinned Kaggle dataset *version* is the frozen snapshot.

See `MASTER_IMPLEMENTATION_GUIDE.md` and `docs/reproducibility_checklist.md` for the full
protocol.
