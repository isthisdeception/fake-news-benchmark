# Experiment Matrix — Fake News Detection Benchmark (Master Implementation Plan)

Author role: Senior NLP researcher
Derived **entirely** from `docs/final_protocol.md` (v1.0, FROZEN). Cross-refs:
`docs/research_questions.md` (RQ1–RQ5), `docs/hypotheses.md` (H1–H6, E1–E3).

> This document is the single source of truth for implementation. Every experiment reads its
> frozen constants from `configs/` mirrors of the protocol, not from hard-coded values
> (protocol §0). No experiment here adds, removes, or alters a frozen setting. Any change
> requires a protocol version bump (§0) and a corresponding edit here.

---

## 0. Conventions (read first)

### 0.1 Dataset IDs and roles (protocol §4.1)
| ID | Dataset | Role | Text type | In article-transfer set? |
|----|---------|------|-----------|:---:|
| DS1 | WELFake | Primary in-distribution + leakage case study | Article (title+text) | Yes |
| DS2 | ISOT | Saturation/easy anchor | Article (title+text) | Yes |
| DS3 | FakeNewsNet (PolitiFact+GossipCop) | Transfer target | Article (title+text) | Yes |
| DS4 | COVID-19 Fake News | Domain-shift probe | Short article/claim | Probe only (test target) |
| DS5 | LIAR | Short-statement control | Short statement | **No — separate track** |

### 0.2 Dataset version tags (frozen data lineage)
Every result must record the dataset version string it used.
- `vRAW` — committed raw snapshot; SHA-256 recorded in `data/SNAPSHOT_HASHES.txt` (§4.6).
- `vBIN` — after frozen binary label mapping `{real=0, fake=1}` (§4.3); ambiguous rows dropped + counted.
- `vCLEAN-C` — classical-track preprocessing (§6): NFC→lowercase→strip URLs/HTML/mentions/hashtags/symbols→NLTK stopwords→WordNet lemmatize.
- `vCLEAN-N` — neural-track preprocessing (§6): NFC→strip URLs/HTML→model-native subword tokenization (no stopword/lemmatization).
- `vDEDUP` — after within-dataset exact + near-dup removal (TF-IDF cosine ≥ 0.95) (§4.5).
- `vXDEDUP(train→test)` — after cross-dataset test-side removal (cosine ≥ 0.90 to any train item) (§4.5); computed per transfer pair.
- Dedup-sensitivity variants (DS1 only): `vDEDUP@0.90`, `vDEDUP@0.95`, `vDEDUP@0.98` (§4.5).

### 0.3 Split types (§4.4)
- `S-RAND` — stratified 70/15/15 random, seed-controlled (default).
- `S-SRC` — source-disjoint split (train/val/test share no source); DS1 only, iff source recoverable.
- `S-TEMP` — temporal (train-past/test-future); only for datasets with reliable timestamps, else "N/A".
- Split is performed BEFORE any resampling or vectorizer fitting.

### 0.4 Model IDs (protocol §5)
- Classical (`vCLEAN-C`, TF-IDF fit on TRAIN only): `LR` (LogReg), `SVM` (LinearSVM), `RF` (RandomForest), `LGBM` (LightGBM).
- Encoders (`vCLEAN-N`): `BERT` (bert-base-uncased), `ROBERTA` (roberta-base), `DEBERTA` (deberta-v3-base), `DISTIL` (distilbert-base-uncased), `ALBERT` (albert-base-v2).
- Decoder LLMs (RQ-F case study, n=2): `MISTRAL` (Mistral-7B-Instruct-v0.2), `LLAMA` (Llama-3.1-8B-Instruct); each as QLoRA fine-tune + zero-shot + 5-shot.
- Efficiency refs (RQ-D): `DISTIL` and `QUANT-BEST` (dynamic INT8 of the best encoder).
- Shortcut control (RQ-B): `SRCONLY` (source/metadata-only; where absent, surface features: length, punctuation, capitalization rate).

### 0.5 Frozen hyperparameter blocks (§6.1, §7) — referenced by name below
- **HP-CLASSICAL:** TF-IDF unigrams+bigrams, `max_features=20000`, `min_df=5`, fit on TRAIN only; per-model grid in `configs/classical_grids.yaml`, selected by validation macro-F1.
- **HP-ENCODER:** AdamW `wd=0.01`; LR `2e-5`; batch 16; scheduler linear warm-up 6% → cosine decay to 0; max 10 epochs, early stop on val macro-F1, patience 3; max_len 512 via sliding window (window=512, stride=128, window logits mean-pooled); class-weighted cross-entropy (inverse frequency); **SMOTE prohibited**.
- **HP-LLM:** QLoRA `r=64, alpha=16, dropout=0.2`, NF4, double-quant; LR `2e-4`; effective batch 16 via gradient accumulation; other schedule as HP-ENCODER; plus zero-shot and 5-shot inference variants.
- **HP-TITLE:** as HP-ENCODER but max_len 64 (title-only condition, RQ-E).

### 0.6 Seeds (§9)
- Primary endpoints (RQ-A/B/C): **5 seeds {13,21,42,87,100}**.
- Secondary endpoints (RQ-D/E/F): **3 seeds {13,42,100}**.
- Deterministic classical models still run all seeds where stochastic (RF, LGBM).

### 0.7 Metric suite (§8) — referenced as METRICS-CORE below
- **METRICS-CORE:** macro-F1 (primary), accuracy, macro-precision, macro-recall, per-class F1 (fake/real), ROC-AUC, PR-AUC. Reported as mean ± std across seeds; never best-run-only.
- **ECE:** Expected Calibration Error (15 bins) — probabilistic models only (transformers softmax; SVM Platt; tree/linear `predict_proba`); LLM zero-shot excluded.
- **TRANSFER:** ΔF1 = in-domain macro-F1 − cross-domain macro-F1; std of macro-F1 across transfer targets.
- **ROBUST:** clean accuracy, accuracy-under-attack, Attack Success Rate (ASR), mean semantic similarity of successful attacks.
- **EFFICIENCY:** parameter count, on-disk size (MB), peak VRAM, training wall-clock, inference latency (ms/sample @ batch=1, seq=512), throughput (samples/s @ batch=32) — all on a single fixed NVIDIA T4 (§14).

### 0.8 Statistics (§10) — referenced as STATS-CONF
- **STATS-CONF:** paired bootstrap on macro-F1 difference (10,000 stratified resamples, 95% CI of difference, two-sided p) + McNemar on paired predictions + Cliff's delta; **BH-FDR at q=0.05** across the confirmatory family. Confirmatory family = pairwise encoder comparisons on DS1 in-distribution + each model's in-vs-cross ΔF1 (RQ-A) + full-text vs source-only (RQ-B). Ranking (secondary): Friedman + Nemenyi.

---

## Phase 0 — Data preparation (prerequisite deliverables)

### EXP-P0 — Snapshot & hash all datasets
- **Objective:** Freeze reproducible dataset snapshots; cache FakeNewsNet content against link rot (§4.6).
- **Dataset version:** DS1–DS5 `vRAW`.
- **Model:** none.
- **Hyperparameters:** none.
- **Metrics:** none (integrity check: SHA-256 per dataset).
- **Expected outputs:** `data/SNAPSHOT_HASHES.txt`; cached FakeNewsNet snapshot committed.

### EXP-P1 — Binarization + preprocessing tracks
- **Objective:** Produce frozen label space and both preprocessing tracks (§4.3, §6).
- **Dataset version:** DS1–DS5 `vRAW → vBIN → {vCLEAN-C, vCLEAN-N}`. LIAR 6-way→binary per frozen map.
- **Model:** none.
- **Hyperparameters:** preprocessing per §6 (both tracks).
- **Metrics:** counts of dropped/ambiguous rows per dataset.
- **Expected outputs:** `data/processed/{DSx}_{vCLEAN-C,vCLEAN-N}.parquet`; `results/label_mapping_report.csv`.

### EXP-P2 — Within-dataset deduplication
- **Objective:** Remove exact + near-duplicates (cosine ≥ 0.95) within each dataset (§4.5).
- **Dataset version:** `vCLEAN-* → vDEDUP` for DS1–DS5.
- **Model:** TF-IDF cosine similarity (dedup utility, not a classifier).
- **Hyperparameters:** exact-match hash; TF-IDF near-dup threshold 0.95.
- **Metrics:** removed-count and %-removed per dataset.
- **Expected outputs:** `data/processed/{DSx}_vDEDUP.parquet`; `results/dedup_counts.csv`.

### EXP-P3 — Splits (random / source-disjoint / temporal)
- **Objective:** Create frozen 70/15/15 splits and record split indices (§4.4).
- **Dataset version:** `vDEDUP` for DS1–DS5.
- **Model:** none.
- **Hyperparameters:** stratified 70/15/15; `S-RAND` per seed; `S-SRC` (DS1 iff source recoverable); `S-TEMP` where timestamps exist else "N/A".
- **Metrics:** none.
- **Expected outputs:** `data/splits/{DSx}_{split}_{seed}.idx`; temporal-applicability note per dataset.

### EXP-P4 — Dataset statistics recording (fixes critique #13)
- **Objective:** Record absolute sizes and class balances per dataset/split (§4.4).
- **Dataset version:** `vBIN` and `vDEDUP`.
- **Model:** none.
- **Hyperparameters:** none.
- **Metrics:** N per split, class ratio, mean/median article length.
- **Expected outputs:** `results/dataset_stats.csv`.

### EXP-P5 — Cross-dataset overlap removal (per transfer pair)
- **Objective:** Remove test-side items with cosine ≥ 0.90 to any training-side item before each transfer (§4.5).
- **Dataset version:** produces `vXDEDUP(train→test)` for every directed pair in {DS1,DS2,DS3} and DS→DS4.
- **Model:** TF-IDF cosine.
- **Hyperparameters:** cross-dataset threshold 0.90.
- **Metrics:** overlap counts per pair.
- **Expected outputs:** `results/interdataset_overlap.csv`; per-pair cleaned test indices.

### EXP-P6 — Dedup-threshold sensitivity (DS1 only, run once) (fixes critique #20)
- **Objective:** Quantify sensitivity of DS1 to dedup threshold.
- **Dataset version:** DS1 `vDEDUP@{0.90,0.95,0.98}`.
- **Model:** best encoder (from EXP-1) + `LR` reference.
- **Hyperparameters:** HP-ENCODER / HP-CLASSICAL; 3 seeds.
- **Metrics:** METRICS-CORE at each threshold.
- **Expected outputs:** `results/dedup_sensitivity_ds1.csv`.

---

## Phase 1 — In-distribution WELFake benchmark (feeds H1)

### EXP-1 — Full model sweep on WELFake in-distribution
- **Objective:** Establish DS1 in-distribution performance for all classical + encoder models; identify the **best encoder** (used downstream by RQ-C/D). Tests **H1** (pairwise encoders indistinguishable after FDR).
- **Dataset version:** DS1 `vDEDUP`, split `S-RAND`.
- **Model:** `LR, SVM, RF, LGBM` (classical); `BERT, ROBERTA, DEBERTA, DISTIL, ALBERT` (encoders). 9 configs.
- **Hyperparameters:** HP-CLASSICAL / HP-ENCODER; 5 seeds.
- **Metrics:** METRICS-CORE + ECE; pairwise encoder comparisons via STATS-CONF.
- **Expected outputs:** `results/ds1_indist.csv` (per model × seed); `results/ds1_encoder_pairwise_stats.csv`; frozen `best_encoder` identifier written to `results/best_encoder.txt`.

---

## Phase 2 — RQ-A: Cross-dataset generalization (primary; H2, E1)

### EXP-A1 — Directed article-type transfer matrix {DS1,DS2,DS3}
- **Objective:** Measure ΔF1 under train-on-one / test-on-others across article-type datasets (RQ1, H2). In-domain diagonal reuses EXP-1 for DS1.
- **Dataset version:** train on `{DS1,DS2,DS3} vDEDUP`; test on each of the three with `vXDEDUP(train→test)` applied off-diagonal.
- **Model:** `LR, SVM, RF, LGBM, BERT, ROBERTA, DEBERTA, DISTIL, ALBERT` (9). Train set × model = 27 trained models; each evaluated on 3 test sets.
- **Hyperparameters:** HP-CLASSICAL / HP-ENCODER; 5 seeds.
- **Metrics:** METRICS-CORE per (train,test); TRANSFER (ΔF1, std across targets); ΔF1 tested via STATS-CONF (in confirmatory family).
- **Expected outputs:** `results/transfer_matrix.csv` (train × test × model × seed); `results/transfer_deltaF1.csv`; family-level ΔF1 comparison (E1, Friedman+Nemenyi, exploratory) in `results/transfer_family_exploratory.csv`.

### EXP-A2 — Domain-shift probe to DS4 (COVID-19)
- **Objective:** Report degradation onto a topical/short-form domain-shift target (probe, not in confirmatory family).
- **Dataset version:** train `{DS1,DS2,DS3} vDEDUP`; test DS4 with `vXDEDUP(train→DS4)`.
- **Model:** all 9 (as EXP-A1).
- **Hyperparameters:** HP-CLASSICAL / HP-ENCODER; 5 seeds.
- **Metrics:** METRICS-CORE; TRANSFER ΔF1 vs each source's in-domain.
- **Expected outputs:** `results/domain_shift_ds4.csv`.

### EXP-A3 — Short-statement track (DS5 LIAR, analyzed separately)
- **Objective:** Report LIAR performance in isolation; **never** mixed into article-to-article transfer (§4.2).
- **Dataset version:** DS5 `vDEDUP`, split `S-RAND` (+ `S-TEMP` if timestamps exist).
- **Model:** all 9.
- **Hyperparameters:** HP-CLASSICAL / HP-ENCODER; 5 seeds.
- **Metrics:** METRICS-CORE + ECE.
- **Expected outputs:** `results/liar_shortstatement.csv`.

---

## Phase 3 — RQ-B: Leakage & shortcut attribution (primary; H3, H4)

### EXP-B1 — Dedup contrast on DS1 (before vs after)
- **Objective:** Quantify accuracy attributable to near-duplicate leakage (H3, part 1).
- **Dataset version:** DS1 pre-dedup (`vCLEAN-*`, no §4.5) **vs** `vDEDUP`, split `S-RAND`.
- **Model:** best encoder + `LR`.
- **Hyperparameters:** HP-ENCODER / HP-CLASSICAL; 5 seeds.
- **Metrics:** METRICS-CORE; before/after macro-F1 delta via STATS-CONF.
- **Expected outputs:** rows in `results/leakage_analysis.csv`.

### EXP-B2 — Split-regime contrast on DS1 (random vs source-disjoint)
- **Objective:** Quantify accuracy attributable to source fingerprints (H3, part 2). If source not recoverable, record and report dedup contrast only (§11 honest scoping).
- **Dataset version:** DS1 `vDEDUP`, split `S-RAND` **vs** `S-SRC`.
- **Model:** best encoder + `LR`.
- **Hyperparameters:** HP-ENCODER / HP-CLASSICAL; 5 seeds.
- **Metrics:** METRICS-CORE; random-vs-source-disjoint delta via STATS-CONF.
- **Expected outputs:** rows in `results/leakage_analysis.csv`; source-recoverability note.

### EXP-B3 — Source/metadata-only shortcut baseline
- **Objective:** Test whether veracity is predictable without article text (H4). Full-text vs source-only is in the confirmatory family.
- **Dataset version:** DS1 (and any dataset with a source field) `vDEDUP`, split `S-RAND`; `SRCONLY` feature set (source/metadata or surface features).
- **Model:** `SRCONLY` (compared against best full-text model from EXP-1).
- **Hyperparameters:** HP-CLASSICAL grid on surface/metadata features; 5 seeds.
- **Metrics:** METRICS-CORE; ratio (source-only macro-F1 / full-text macro-F1); STATS-CONF.
- **Expected outputs:** rows in `results/leakage_analysis.csv` including the ≥0.80 ratio check for H4.

---

## Phase 4 — RQ-C: Adversarial robustness (primary; H5)

### EXP-C1 — Label-preserving adversarial suite on DS1 test set
- **Objective:** Measure robustness of each family to label-preserving perturbations (RQ3). Feeds H5 (LLM vs encoder ASR).
- **Dataset version:** DS1 `vDEDUP` **test set** only; attacked variant `vDEDUP+ADV`.
- **Model:** all encoders + classical + (budget-permitting) `MISTRAL`, `LLAMA` (using their EXP-1/EXP-F1 fine-tuned checkpoints).
- **Hyperparameters:** attacks = char typos, embedding-constrained synonym substitution, benign insertion, back-translation paraphrase; **negation/antonym prohibited**; USE similarity ≥ 0.90 (TextAttack constraint); attacks below threshold discarded. Evaluated on the fixed checkpoints (no retraining); seeds inherited from source checkpoints.
- **Metrics:** ROBUST (clean acc, accuracy-under-attack, ASR, mean similarity of successful attacks); per-family ASR mean ± std.
- **Expected outputs:** `results/adversarial_robustness.csv`; released perturbation sets (documented as defensive, §16).

---

## Phase 5 — RQ-D: Efficiency / accuracy-per-compute (secondary; H6)

### EXP-D1 — Efficiency profiling on fixed T4
- **Objective:** Build the accuracy-vs-compute Pareto frontier (RQ5, H6). All timing on ONE fixed NVIDIA T4 (§14).
- **Dataset version:** DS1 `vDEDUP`, split `S-RAND`.
- **Model:** all encoders + `LR`/`LGBM` reference + `DISTIL` + `QUANT-BEST` (dynamic INT8 of best encoder).
- **Hyperparameters:** HP-ENCODER / HP-CLASSICAL; 3 seeds; inference @ batch=1 seq=512 (latency) and batch=32 (throughput).
- **Metrics:** EFFICIENCY + macro-F1; compact-vs-best ratio for H6 (≥95% F1 at <50% params).
- **Expected outputs:** `results/efficiency.csv`; Pareto plots (F1 vs params, F1 vs latency).

### EXP-D2 — Optional compression study (best encoder)
- **Objective:** Report accuracy/size trade-off of INT8 quantization + knowledge distillation on the best encoder (§14, optional).
- **Dataset version:** DS1 `vDEDUP`.
- **Model:** `QUANT-BEST` + distilled student.
- **Hyperparameters:** dynamic INT8; KD from best encoder; 3 seeds.
- **Metrics:** macro-F1 vs on-disk size vs latency delta.
- **Expected outputs:** `results/compression_study.csv`.

---

## Phase 6 — RQ-E: Input granularity (secondary; E2)

### EXP-E1 — Title vs content vs title+content
- **Objective:** Test whether input granularity changes model ranking (RQ-E, E2). Runs ONLY on datasets with both title and body {DS1,DS2,DS3} (§6.1).
- **Dataset version:** DS1/DS2/DS3 `vDEDUP`, three input conditions each.
- **Model:** all 5 encoders (+ classical for completeness).
- **Hyperparameters:** HP-ENCODER (content, title+content) / HP-TITLE (title-only, max_len 64); 3 seeds.
- **Metrics:** METRICS-CORE per condition; ranking stability via Friedman + Nemenyi (exploratory).
- **Expected outputs:** `results/granularity.csv`; ranking-stability table.

---

## Phase 7 — RQ-F: Encoder vs decoder-LLM case study (secondary; H5 support, E3)

### EXP-F0 — LLM contamination probe (gate before any LLM claim)
- **Objective:** Flag memorized datasets per LLM via guided prefix-completion; exclude flagged datasets from that LLM's zero-shot claims (§4.7).
- **Dataset version:** DS1–DS4 `vDEDUP` (probe inputs).
- **Model:** `MISTRAL`, `LLAMA`.
- **Hyperparameters:** guided prefix-completion memorization test.
- **Metrics:** per-LLM per-dataset contamination flag + memorization score.
- **Expected outputs:** `results/llm_contamination.csv`.

### EXP-F1 — QLoRA fine-tune LLMs on WELFake
- **Objective:** Fine-tuned decoder-LLM performance for the encoder-vs-decoder comparison (RQ4).
- **Dataset version:** DS1 `vDEDUP`, split `S-RAND`.
- **Model:** `MISTRAL`, `LLAMA` (QLoRA).
- **Hyperparameters:** HP-LLM; 3 seeds; capped at 20 GPU-h total (§1) — if exceeded, reported partial.
- **Metrics:** METRICS-CORE (reported alongside fine-tuned encoders from EXP-1); STATS-CONF vs encoders is exploratory except the H1 encoder-pairwise family.
- **Expected outputs:** `results/llm_finetune_ds1.csv`; QLoRA adapters.

### EXP-F2 — Zero-shot & 5-shot LLM inference (contamination-suspect)
- **Objective:** Report prompt-based LLM performance, kept separate from fine-tuned results (§4.7, E3).
- **Dataset version:** DS1 `vDEDUP` (and DS2/DS3/DS4 if not contamination-flagged in EXP-F0).
- **Model:** `MISTRAL`, `LLAMA` (zero-shot, 5-shot).
- **Hyperparameters:** fixed prompt template + 5-shot exemplars; no training.
- **Metrics:** METRICS-CORE **excluding ECE** (LLM zero-shot excluded from ECE, §8); explicitly labeled contamination-suspect.
- **Expected outputs:** `results/llm_prompted.csv` (separate table).

---

## Phase 8 — Analysis & interpretability (support; applies across RQs)

### EXP-G1 — Calibration (ECE)
- **Objective:** Report ECE for all probabilistic models (§8).
- **Dataset version:** DS1 `vDEDUP` test (+ transfer targets).
- **Model:** all probabilistic (transformers softmax; SVM Platt; tree/linear predict_proba). LLM zero-shot excluded.
- **Hyperparameters:** 15-bin ECE.
- **Metrics:** ECE; reliability diagrams.
- **Expected outputs:** `results/calibration.csv`; reliability plots.

### EXP-G2 — Error analysis & failure taxonomy (§13)
- **Objective:** Characterize failure modes and fake→real vs real→fake asymmetry.
- **Dataset version:** DS1 (+ DS2 easy, DS5 hard buckets).
- **Model:** best encoder + `LR`.
- **Hyperparameters:** written coding scheme in `docs/error_coding_scheme.md`; ≥200 errors double-coded where feasible.
- **Metrics:** confusion matrices; error-type distribution; performance vs article length.
- **Expected outputs:** `results/error_analysis.csv`; `docs/error_coding_scheme.md`.

### EXP-G3 — Interpretability with faithfulness (§13)
- **Objective:** Token-attribution explanations with a comprehensiveness/sufficiency check; attention illustrative only.
- **Dataset version:** DS1 `vDEDUP`.
- **Model:** best encoder.
- **Hyperparameters:** Integrated Gradients + leave-one-token occlusion.
- **Metrics:** comprehensiveness, sufficiency (faithfulness).
- **Expected outputs:** `results/interpretability.csv`; example attribution figures.

### EXP-G4 — Confirmatory statistics aggregation (§10)
- **Objective:** Assemble the confirmatory family and apply BH-FDR globally; attach effect sizes + CIs.
- **Dataset version:** consumes EXP-1, EXP-A1, EXP-B3 outputs.
- **Model:** none (analysis).
- **Hyperparameters:** STATS-CONF; BH-FDR q=0.05 across whole family.
- **Metrics:** adjusted p, 95% CI of difference, Cliff's delta per comparison.
- **Expected outputs:** `results/confirmatory_stats.csv`; significance verdicts for H1–H4.

---

## 9. Confirmatory family membership (pre-registered, §10)

Only these comparisons are FDR-protected confirmatory; everything else is exploratory:
1. Pairwise macro-F1 among the 5 encoders on DS1 in-distribution (EXP-1) → **H1**.
2. Each model's in-domain-vs-cross-domain ΔF1 for article transfer (EXP-A1) → **H2**.
3. Full-text vs source-only on DS1 (EXP-B3), plus dedup and split-regime deltas (EXP-B1/B2) → **H3, H4**.

## 10. Execution order & dependencies

```
P0 → P1 → P2 → P3 → P4 ; P5 (needs P2/P3) ; P6 (needs EXP-1)
EXP-1 (needs P1–P4) → best_encoder.txt
  ├─ Phase 2 EXP-A1/A2/A3 (needs P5 for off-diagonal)
  ├─ Phase 3 EXP-B1/B2/B3
  ├─ Phase 4 EXP-C1 (needs EXP-1 + EXP-F1 checkpoints)
  ├─ Phase 5 EXP-D1/D2
  ├─ Phase 6 EXP-E1
  └─ Phase 7 EXP-F0 (gate) → EXP-F1 → EXP-F2
Phase 8 EXP-G1..G4 consume the above; G4 last.
```
Priority (protocol §1): complete primary endpoints (Phase 1–4 + G1/G2/G4) first; run secondary
(D/E/F) only if within the ≤ 120 GPU-h budget; RQ-F (LLMs) capped at 20 GPU-h or reported partial.

## 11. Compute-budget ledger (target ≤ 120 T4 GPU-h, §1)

| Phase | Experiments | Rough GPU-h (est.) |
|-------|-------------|:---:|
| 0 Data prep | P0–P6 | ~4 (mostly CPU) |
| 1 In-dist | EXP-1 (5 enc × 5 seeds + classical) | ~30 |
| 2 Transfer | EXP-A1/A2/A3 (encoders reuse P1 checkpoints where identical train set) | ~28 |
| 3 Leakage | EXP-B1/B2/B3 | ~10 |
| 4 Adversarial | EXP-C1 (inference-time attacks) | ~12 |
| 5 Efficiency | EXP-D1/D2 | ~6 |
| 6 Granularity | EXP-E1 | ~8 |
| 7 LLM case study | EXP-F0/F1/F2 | ≤ 20 (hard cap) |
| 8 Analysis | EXP-G1–G4 | ~2 |
| **Total** | | **≈ 120** |

Estimates are planning figures; actuals are recorded per run and reconciled against the cap.
If the total is projected to exceed 120 GPU-h, cut secondary phases in order F → E → D
(primary endpoints are never cut).

## 12. Deliverables index (result artifacts, §15)

`data/SNAPSHOT_HASHES.txt`, `results/dataset_stats.csv`, `results/dedup_counts.csv`,
`results/interdataset_overlap.csv`, `results/dedup_sensitivity_ds1.csv`,
`results/ds1_indist.csv`, `results/ds1_encoder_pairwise_stats.csv`, `results/best_encoder.txt`,
`results/transfer_matrix.csv`, `results/transfer_deltaF1.csv`, `results/domain_shift_ds4.csv`,
`results/liar_shortstatement.csv`, `results/leakage_analysis.csv`,
`results/adversarial_robustness.csv`, `results/efficiency.csv`,
`results/compression_study.csv`, `results/granularity.csv`,
`results/llm_contamination.csv`, `results/llm_finetune_ds1.csv`, `results/llm_prompted.csv`,
`results/calibration.csv`, `results/error_analysis.csv`, `results/interpretability.csv`,
`results/confirmatory_stats.csv`, plus `configs/*.yaml`, split indices, perturbation sets,
`ENVIRONMENT.md`. Every result row records the protocol version (`v1.0`) and dataset version tag.
