# FINAL PROTOCOL — Fake News Detection Benchmark (v1.0, FROZEN)

> STATUS: IMMUTABLE. Frozen 2026-07-25.
> This document supersedes `docs/experimental_design.md`. All code in this repository
> MUST conform to the settings below. No experimental setting may be added, removed,
> or altered after this point unless the user explicitly instructs a change, in which
> case the version is bumped (v1.1, ...) and the change is recorded in §16 Changelog.
> Where a value is marked [FROZEN], it is a hard constant; code must read it from
> `configs/` mirrors of this document, not hard-code divergent values.

This protocol integrates the reviewed corpus (P001–P012) and resolves the critique in
the design review. Each resolved issue is tagged (e.g., "fixes critique #2").

---

## 0. Change Control

- Version: **1.0** [FROZEN]
- Any modification requires: (a) explicit user instruction, (b) version bump, (c) a Changelog entry in §16, (d) migration note for affected code/results.
- Results produced under a given version MUST record that version string.

---

## 1. Scope, Endpoints, and Compute Budget (fixes critique #1)

- **Primary endpoint (headline thesis):** *A reproducible, leakage-audited measurement of cross-dataset generalization in fake news detection.* [FROZEN]
- **Primary RQs:** RQ-A (cross-dataset generalization), RQ-B (leakage/shortcut attribution), RQ-C (adversarial robustness).
- **Secondary RQs (report only if primary complete within budget):** RQ-D (efficiency/Pareto), RQ-E (input granularity), RQ-F (encoder-vs-decoder LLM case study).
- **Compute budget:** target ≤ 120 GPU-hours total on Kaggle free tier (NVIDIA T4). The experiment matrix in §5–§7 is sized to fit this. Decoder-LLM runs (RQ-F) are capped at 20 GPU-hours; if exceeded, RQ-F is reported as partial. [FROZEN]

---

## 2. Research Questions [FROZEN]

- **RQ-A.** How much does macro-F1 degrade under leave-one-dataset-out transfer across article-type datasets?
- **RQ-B.** What fraction of in-distribution accuracy is attributable to near-duplicate and source/metadata shortcuts (vs. veracity signal)?
- **RQ-C.** How does label-preserving adversarial perturbation affect each model family?
- **RQ-D (secondary).** Accuracy-per-compute across families.
- **RQ-E (secondary).** Effect of input granularity (title / content / title+content) on model ranking.
- **RQ-F (secondary).** Encoder vs. open decoder-LLM comparison (case study; n=2 LLMs).

---

## 3. Hypotheses (directional, pre-registered) [FROZEN]

- **H1.** Among top encoders, pairwise macro-F1 differences are NOT significant after FDR correction.
- **H2.** Every model loses ≥ 10 macro-F1 points under cross-dataset transfer (RQ-A).
- **H3.** Removing near-duplicates + enforcing source-disjoint splits lowers WELFake macro-F1 by ≥ 3 points (RQ-B).
- **H4.** A source/metadata-only classifier reaches ≥ 0.80 of the full-text model's macro-F1 on at least one dataset (shortcut evidence, RQ-B).
- **H5.** Decoder LLMs (case study) degrade less than encoders under adversarial perturbation (RQ-C), replicating P001.
- **H6.** DistilBERT/ALBERT reach ≥ 95% of the best encoder's macro-F1 at < 50% parameters (RQ-D).

Pre-registration note (fixes critique #16): hypotheses and the primary comparison family (§10) are fixed here BEFORE running experiments; confirmatory tests are limited to this set. Anything else is exploratory and labeled as such.

---

## 4. Datasets [FROZEN]

### 4.1 Frozen dataset set and roles
| ID | Dataset | Role | Text type |
|----|---------|------|-----------|
| DS1 | WELFake | Primary in-distribution + leakage case study (RQ-B) | Article (title+text) |
| DS2 | ISOT | Saturation/easy anchor | Article (title+text) |
| DS3 | FakeNewsNet (PolitiFact+GossipCop) | Transfer target | Article (title+text) |
| DS4 | COVID-19 Fake News | Domain-shift probe | Short article/claim |
| DS5 | LIAR | Hard, SHORT-STATEMENT control (analyzed separately) | Short statement |

### 4.2 Cross-dataset construct control (fixes critique #3)
- **Main transfer (RQ-A) is computed ONLY among article-type datasets {DS1, DS2, DS3}.** DS4 is a domain-shift probe; **DS5 (LIAR) is EXCLUDED from article-to-article transfer** and analyzed only in a separate short-statement track, because mixing statement-length and article-length text confounds transfer with text-type shift.

### 4.3 Label binarization (frozen mapping)
- Global label space: `{real = 0, fake = 1}`.
- LIAR 6-way → binary: `{pants-fire, false, barely-true} → fake`; `{half-true, mostly-true, true} → real`. [FROZEN]
- Any dataset lacking a clean binary mapping is documented and its ambiguous rows dropped (recorded with counts).

### 4.4 Splits
- Stratified **70/15/15** train/val/test, `random_state` per §9 seeds. [FROZEN]
- Split BEFORE any resampling or vectorizer fitting.
- Absolute per-dataset sizes and class balances MUST be recorded in `results/dataset_stats.csv` (fixes critique #13; do not repeat P011's omission).
- **Temporal split** is applied ONLY to datasets with reliable per-item timestamps; if a dataset lacks them, the temporal analysis is reported as "not applicable" for that dataset (honest reporting; fixes over-promise).

### 4.5 Deduplication & inter-dataset leakage (fixes critique #4)
- **Within-dataset:** remove exact duplicates, then near-duplicates via TF-IDF cosine ≥ **0.95**. [FROZEN]
- **Across-dataset (before every transfer):** remove test-side items whose TF-IDF cosine ≥ **0.90** to any training-side item. Report overlap counts in `results/interdataset_overlap.csv`. [FROZEN]
- A dedup-threshold sensitivity check at {0.90, 0.95, 0.98} is run ONCE on DS1 and reported (fixes critique #20).

### 4.6 Snapshotting (fixes critique #14)
- Each dataset is frozen as a committed snapshot with a recorded SHA-256 content hash in `data/SNAPSHOT_HASHES.txt`. FakeNewsNet content (subject to link rot) is cached at snapshot time; experiments use ONLY the cached snapshot. [FROZEN]

### 4.7 LLM contamination handling (fixes critique #2)
- Decoder-LLM (RQ-F) zero-shot/few-shot results are treated as **contamination-suspect** and reported separately from fine-tuned results.
- A contamination probe is run per LLM per dataset (guided prefix-completion memorization test); results recorded in `results/llm_contamination.csv`. Any dataset flagged contaminated for a given LLM is excluded from that LLM's zero-shot claims. [FROZEN]

---

## 5. Models [FROZEN]

- **Classical (TF-IDF, fit on TRAIN only):** Logistic Regression, Linear SVM, Random Forest, LightGBM.
- **Encoders:** `bert-base-uncased`, `roberta-base`, `microsoft/deberta-v3-base`, `distilbert-base-uncased`, `albert-base-v2`.
- **Decoder LLMs (RQ-F case study, n=2 to avoid single-model overreach — fixes critique #8):** `mistralai/Mistral-7B-Instruct-v0.2` and `meta-llama/Llama-3.1-8B-Instruct`, each via 4-bit QLoRA fine-tune + zero-shot + 5-shot.
- **Efficiency references (RQ-D):** DistilBERT and a dynamically-quantized version of the best encoder.
- **Shortcut control baseline (RQ-B, novel — fixes review "high-value addition"):** a classifier trained on **source/metadata ONLY (no article text)**. Where no source field exists, use non-content surface features (length, punctuation, capitalization rate). [FROZEN]

Decoder-LLM conclusions are explicitly scoped as a 2-model case study, not a general claim about "decoder LLMs."

---

## 6. Preprocessing [FROZEN]

Two frozen tracks; preprocessing is entangled with model family and all family-level comparisons are bounded accordingly (fixes critique #7 framing):

- **Classical track:** Unicode NFC → lowercase → remove URLs/HTML/mentions/hashtags/symbols → NLTK stopword removal → WordNet lemmatization → TF-IDF (unigrams+bigrams, `max_features=20000`, `min_df=5`), vectorizer fit on TRAIN only (P003 anti-leakage).
- **Neural track:** Unicode NFC → remove URLs/HTML → model-native subword tokenization; NO stopword removal / lemmatization.

### 6.1 Full-document exposure (fixes critique #7)
- Transformers process the FULL document via sliding windows: window 512 tokens, stride 128; window logits mean-pooled. This gives transformers whole-document access comparable to TF-IDF. [FROZEN]
- Title-only condition (RQ-E): max 64 tokens.
- Granularity conditions (RQ-E) run ONLY on datasets that contain both title and body {DS1, DS2, DS3}.

---

## 7. Training Configuration [FROZEN]

Single frozen configuration per family (no post-hoc tuning — keeps the protocol immutable and fair):

- **Optimizer:** AdamW, weight_decay = 0.01.
- **Scheduler:** linear warm-up 6% of steps → cosine decay to 0. [FROZEN]
- **Encoder LR:** 2e-5 (fixed). **QLoRA LR:** 2e-4. **Classical:** frozen grid per model in `configs/classical_grids.yaml`, selected by validation macro-F1.
- **Batch size:** 16 for encoders; effective 16 via gradient accumulation for LLMs. [FROZEN]
- **Epochs:** max 10, early stopping on validation macro-F1, patience 3.
- **Max length:** 512 (windowed as §6.1); 64 for title-only.
- **QLoRA:** r=64, alpha=16, dropout=0.2, NF4, double-quant (P001 recipe).
- **Class imbalance:** inverse-frequency class-weighted cross-entropy. **SMOTE is PROHIBITED** for all models (fixes critique #10). Test sets always use the NATURAL class distribution (never a rebalanced test set).
- **Reporting:** all seeds reported as mean ± std; NEVER report best-run-only (corrects P011).

---

## 8. Evaluation Metrics [FROZEN]

- **Primary:** Macro-F1. **Reported alongside:** Accuracy, Macro-Precision, Macro-Recall, per-class F1 (fake/real), ROC-AUC, PR-AUC.
- **Calibration:** Expected Calibration Error (15 bins) computed ONLY for models with valid probability outputs (transformers via softmax; SVM via Platt scaling; tree/linear via `predict_proba`). LLM zero-shot is EXCLUDED from ECE (fixes critique #12).
- **Transfer (RQ-A):** ΔF1 = in-domain macro-F1 − cross-domain macro-F1; and std of macro-F1 across transfer targets.
- **Robustness (RQ-C):** clean accuracy, accuracy-under-attack, Attack Success Rate, mean semantic similarity of successful attacks.
- Every model reports the SAME metric suite on every applicable dataset. Borrowed/literature numbers are NEVER used as our results (corrects P010/P012).

---

## 9. Seeds & Repetition [FROZEN]

- Primary endpoint (RQ-A, RQ-B, RQ-C): **5 seeds = {13, 21, 42, 87, 100}**.
- Secondary endpoints: 3 seeds = {13, 42, 100}.
- Classical deterministic models still run all seeds where stochastic (RF/LightGBM).

---

## 10. Statistical Analysis [FROZEN] (fixes critiques #5, #6)

- **Confirmatory comparison family (pre-registered):** the pairwise comparisons among the 5 encoders on DS1 in-distribution, plus each model's in-domain-vs-cross-domain ΔF1 for RQ-A, plus full-text vs source-only for RQ-B. This is the ONLY family eligible for confirmatory claims.
- **Primary test:** paired **bootstrap test on the macro-F1 difference** (10,000 stratified resamples), reporting the 95% CI **of the difference** and a two-sided p-value. (The design's "non-overlapping marginal CIs" rule is REMOVED.)
- **Accuracy-level test:** McNemar on paired predictions (same test set).
- **Multiplicity:** **Benjamini–Hochberg FDR at q = 0.05** applied across the entire confirmatory family. A difference is "significant" iff its BH-adjusted p < 0.05. [FROZEN]
- **Effect size:** Cliff's delta on per-seed macro-F1 (report with every significant result).
- **Overall ranking (secondary/exploratory):** Friedman + Nemenyi post-hoc.
- No confirmatory claim may be made from a raw delta without (adjusted p, effect size, CI of the difference).

---

## 11. Leakage & Shortcut Analysis (RQ-B) [FROZEN]

- Report DS1 metrics **before vs after** dedup (§4.5) and **random-split vs source-disjoint split**.
- If per-article source is recoverable for DS1, run the source-disjoint split; if NOT recoverable, record this and report only the dedup contrast (honest scoping).
- Train the **source/metadata-only shortcut baseline** (§5) and report its macro-F1 relative to the full-text model. H4 is evaluated here.
- Deliverable: `results/leakage_analysis.csv` quantifying accuracy attributable to shortcuts.

---

## 12. Adversarial Robustness (RQ-C) [FROZEN] (fixes critique #9)

- **Perturbations are LABEL-PRESERVING ONLY:** character-level typos, embedding-constrained synonym substitution, benign word insertion, and back-translation paraphrase.
- **PROHIBITED:** negation and antonym substitution (can flip the gold label).
- **Semantic constraint:** Universal Sentence Encoder similarity ≥ **0.90** between original and perturbed text (via TextAttack constraints); attacks violating this are discarded. [FROZEN]
- Applied to the DS1 test set for all encoders + classical + (budget-permitting) the 2 LLMs.
- Report clean vs attacked accuracy, Attack Success Rate, and mean similarity.

---

## 13. Error Analysis [FROZEN]

- Per-dataset confusion matrices; fake→real vs real→fake asymmetry.
- Failure taxonomy (sarcasm/satire, ambiguous/figurative, atypical source, drift) using a WRITTEN CODING SCHEME in `docs/error_coding_scheme.md`; a random sample of ≥ 200 errors double-coded where feasible.
- Difficulty stratification: easy (DS2/ISOT) vs hard (DS5/LIAR) buckets; performance vs article length.
- Interpretability via **Integrated Gradients and leave-one-token occlusion** with a faithfulness (comprehensiveness/sufficiency) check. Attention weights are illustrative ONLY and never used as the sole explanation (fixes critique #15).

---

## 14. Efficiency Analysis (RQ-D) [FROZEN] (fixes critique #11)

- ALL timing measured on a SINGLE fixed accelerator: **NVIDIA T4 (16 GB), Kaggle**. Runs on any other GPU are excluded from efficiency tables. [FROZEN]
- Report: parameter count, on-disk size (MB), peak VRAM, training wall-clock, inference latency (ms/sample at batch=1, seq=512), throughput (samples/s at batch=32).
- Pareto frontier: macro-F1 vs parameters, and macro-F1 vs latency.
- Optional compression study: dynamic INT8 quantization + KD on the best encoder (P010), reporting the accuracy/size trade-off explicitly.

---

## 15. Reproducibility & Artifacts [FROZEN]

- Pin and record: Python, PyTorch, Transformers, scikit-learn, TextAttack, CUDA versions in `ENVIRONMENT.md`.
- Enable deterministic algorithms; set all seeds (§9).
- Commit dataset snapshot hashes (§4.6), all `configs/*.yaml`, and per-model result CSVs.
- Release: code, configs, splits (indices), perturbation sets, and result tables. No proprietary/API-gated models are used for our reported results (corrects P002).

---

## 16. Ethics & Responsible Release [FROZEN]

- Respect each dataset's license; no redistribution beyond permitted terms; no PII beyond what datasets already contain.
- Dual-use: do NOT release tooling whose primary purpose is evading detectors; adversarial perturbation configs are released for reproducibility but documented as defensive evaluation.
- Include a Broader Impact / limitations statement in the final paper (English-only, text-only scope; risk of over-trust in automated verdicts).

---

## 17. Explicit Out-of-Scope (frozen; not to be silently added) 

- Multilingual / cross-lingual detection (gap G3) — OUT.
- Image/video/audio multimodal detection (gap G4) — OUT.
- API/closed-weight models (e.g., GPT-4o) as our reported systems — OUT.
- Any dataset not listed in §4.1 — OUT.

Adding any of the above requires a version bump per §0.

---

## Changelog

- **v1.0 (2026-07-25):** Initial frozen protocol. Resolves design-review critiques #1–#23:
  scoped primary endpoint + compute budget (#1); LLM contamination probe (#2);
  article-only transfer + binarization mapping (#3); inter-dataset dedup (#4);
  bootstrap-difference test replacing CI-overlap rule (#5); global BH-FDR (#6);
  sliding-window full-document transformer input (#7); two-LLM case study (#8);
  label-preserving-only adversarial suite (#9); SMOTE prohibited (#10);
  fixed T4 for efficiency (#11); ECE only for probabilistic models (#12);
  mandatory dataset-stats recording (#13); dataset snapshot hashing (#14);
  IG/occlusion with faithfulness, attention demoted (#15); pre-registration (#16);
  softened "first"/contribution claims via scoped RQs (#17); directional hypotheses (#18);
  5 seeds for primary endpoint (#19); dedup-threshold sensitivity (#20);
  ethics/dual-use section (#21); licensing/versioning (#22); single sharp primary thesis (#23);
  plus the source/metadata-only shortcut control baseline.
