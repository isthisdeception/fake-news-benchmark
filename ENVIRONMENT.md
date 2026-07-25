# Environment & Reproducibility

**Status:** STUB — to be finalized on the actual Kaggle T4 runtime (Milestone 1 / EXP-P0).
Values marked _TBD_ must be captured from the real runtime with `pip freeze`,
`nvidia-smi`, and `python -c "import torch; ..."` and pasted back here. Do not invent
numbers: this file is a reproducibility record.

Execution target: **Kaggle Notebooks, single NVIDIA T4 GPU.** Code is authored in Cursor,
pushed to GitHub, and executed on Kaggle (see `README.md` §2 and `KAGGLE_WORKFLOW.md`).

---

## 1. Python & pinned libraries

- **Python:** requires `>=3.10` (Kaggle image Python: _TBD_ — record `python --version`).
- **Runtime dependencies:** declared in [`requirements.txt`](./requirements.txt) using
  minimum-version **floors** (`>=`), not exact pins. `pyproject.toml` consumes them via dynamic
  metadata. On Kaggle this deliberately keeps the image's (newer) torch/numpy/etc. and only adds
  missing packages — it never downgrades the Kaggle CUDA stack. Heavier, later-milestone libs
  (peft, bitsandbytes, textattack) are optional extras installed per-stage (`.[llm]`,
  `.[adversarial]`).
- **Full lock (the real reproducibility record):** [`requirements.lock.txt`](./requirements.lock.txt)
  (_TBD placeholder_) must hold a complete `pip freeze` captured **on the Kaggle T4 runtime**
  after install. On a fixed Kaggle image, that freeze — not the floors above — is the exact,
  reproducible version set (transitive deps + the Kaggle CUDA build of torch included).

Key pinned libraries (see `requirements.txt` for the complete, versioned list):

| Category | Libraries |
|---|---|
| Deep learning core | `torch`, `transformers`, `tokenizers`, `datasets`, `accelerate` |
| PEFT / quantization | `peft`, `bitsandbytes` |
| Classical ML | `scikit-learn`, `lightgbm` |
| NLP | `nltk`, `sentence-transformers` |
| Adversarial | `textattack` |
| Statistics | `statsmodels`, `scikit-posthocs`, `scipy` |
| Data / numerics | `pandas`, `numpy`, `pyarrow`, `pyyaml` |
| Plotting / interpretability | `matplotlib`, `captum` |
| Dev tooling | `pytest`, `ruff`, `black` |

> Note on Kaggle: several of these ship pre-installed in the Kaggle GPU image. Where a pin
> conflicts with the Kaggle CUDA build of `torch`, prefer the Kaggle-provided `torch` and
> record the divergence here and in `requirements.lock.txt`.

---

## 2. CUDA / GPU stack

Capture the following on the Kaggle T4 runtime and record them here:

| Item | Value | How to capture |
|---|---|---|
| NVIDIA driver version | _TBD_ | `nvidia-smi` |
| CUDA (driver) version | _TBD_ | `nvidia-smi` (top-right) |
| CUDA (torch build) | _TBD_ | `python -c "import torch; print(torch.version.cuda)"` |
| cuDNN version | _TBD_ | `python -c "import torch; print(torch.backends.cudnn.version())"` |
| torch version | _TBD_ | `python -c "import torch; print(torch.__version__)"` |
| `torch.cuda.is_available()` | _TBD_ | should be `True` on GPU T4 sessions |
| bitsandbytes CUDA setup | _TBD_ | `python -m bitsandbytes` (verifies 4-bit/NF4 support) |

**Determinism knobs** (set by `fnb.utils.seeding.set_global_seed`, Milestone 1 / S4):
`PYTHONHASHSEED`, `CUBLAS_WORKSPACE_CONFIG=:4096:8` (must be set **before** CUDA init),
`torch.use_deterministic_algorithms(True)`, `cudnn.deterministic=True`, `cudnn.benchmark=False`.

---

## 3. Hardware — Kaggle T4

| Item | Value |
|---|---|
| Platform | Kaggle Notebooks |
| Accelerator | 1× NVIDIA Tesla **T4** (`GPU T4 x1`) |
| GPU memory | ~15 GB (16 GB nominal) — _confirm via `nvidia-smi`_ |
| GPU architecture | Turing (compute capability 7.5) |
| System RAM | _TBD_ (record from runtime) |
| vCPUs | _TBD_ |
| Disk (`/kaggle/working`) | ephemeral, wiped at session end |
| Session limits | up to ~9–12 h/session; ~30 GPU-h/week quota (see `KAGGLE_WORKFLOW.md` §6) |
| Compute budget | ≤120 GPU-h total; RQ-F capped at 20 GPU-h (protocol §1) |

**Single-T4 rule (protocol §14):** all efficiency/latency numbers must come from **one fixed
T4 session type** so runtime comparisons are valid. Record the exact T4 session used for the
efficiency experiments (EXP-D) here once measured.

---

## 4. Capture checklist (run once on Kaggle, then fill in above)

```python
import sys, torch, platform
print("python", sys.version)
print("platform", platform.platform())
print("torch", torch.__version__, "cuda", torch.version.cuda,
      "cudnn", torch.backends.cudnn.version())
print("cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0))
```

```bash
nvidia-smi
pip freeze > requirements.lock.txt   # commit this back into the repo
```

---

## 5. NLTK resources (EXP-P1b / protocol §6 classical track)

Recorded automatically by `fnb.data.preprocess.write_nltk_versions` when running
`--stage P1b`. Authoritative JSON copy: `results/nltk_resource_versions.json`.

| Item | Value |
|---|---|
| NLTK package version | <!-- NLTK_VERSION --> |
| Stopwords / WordNet download date | <!-- NLTK_DOWNLOAD_DATE --> |
| English stopwords count | <!-- NLTK_STOPWORDS_N --> |
| WordNet synset count (fingerprint) | <!-- NLTK_WORDNET_SYNSETS --> |

Classical track uses NLTK English stopwords + WordNet lemmatizer. Neural track does
**not** use stopword removal or lemmatization.
