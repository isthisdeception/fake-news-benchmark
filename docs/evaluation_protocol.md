# Evaluation Protocol — Fake News Detection Benchmark (FIXED)

Author role: Senior NLP researcher
Consistent with and subordinate to `docs/final_protocol.md` (v1.0, FROZEN). Cross-refs:
`docs/experiment_matrix.md` (EXP-*), `docs/research_questions.md` (RQ1–RQ5),
`docs/hypotheses.md` (H1–H6, E1–E3).

> This document fixes **how results are measured, aggregated, tested, and interpreted** so that
> no evaluation choice is made after seeing results. It adds no new experimental setting; it
> operationalizes the metrics and statistics already frozen in the protocol (§8–§10). Any change
> requires a protocol version bump (§0 of the protocol).

---

## Part 1 — The Fixed Evaluation Protocol (governing rules)

**R1. Primary metric.** **Macro-F1** is the single primary metric for every classification
experiment and the only metric used to declare a "winner", trigger early stopping, and select
classical grids. All other metrics are diagnostic/supporting.

**R2. Natural test distribution.** Test sets always use the **natural class distribution**;
resampling/class-weighting affects training only. No SMOTE anywhere (protocol §7). Rebalanced
test sets are prohibited.

**R3. Aggregation.** Every reported number is **mean ± std across seeds** — 5 seeds
{13,21,42,87,100} for primary endpoints (RQ-A/B/C), 3 seeds {13,42,100} for secondary (RQ-D/E/F).
**Best-run-only reporting is prohibited** (corrects P011/P012).

**R4. Provenance.** Every result row records `protocol_version=v1.0`, the **dataset-version tag**
(`vDEDUP`, `vXDEDUP(train→test)`, etc.), split type, model ID, and seed. Borrowed/literature
numbers are never reported as our results (corrects P010/P012).

**R5. Confirmatory vs exploratory.** Only the pre-registered confirmatory family (protocol §10;
matrix §9) is eligible for confirmatory claims; everything else is labeled exploratory. A
confirmatory claim requires the triple **(BH-adjusted p, effect size, CI of the difference)** —
never a raw delta.

**R6. Statistical stack (fixed).**
- Difference test: **paired bootstrap on the macro-F1 difference**, 10,000 stratified resamples,
  report 95% CI of the difference and two-sided p.
- Accuracy-level: **McNemar** on paired predictions (same test set).
- Effect size: **Cliff's delta** on per-seed macro-F1.
- Multiplicity: **Benjamini–Hochberg FDR at q = 0.05** across the entire confirmatory family; a
  result is "significant" iff BH-adjusted p < 0.05.
- Ranking (secondary/exploratory only): **Friedman + Nemenyi**.

**R7. Calibration eligibility.** ECE is computed only for models with valid probabilities
(transformer softmax; SVM via Platt; tree/linear via `predict_proba`); **LLM zero-shot is
excluded from ECE** (protocol §8/§12).

**R8. Efficiency isolation.** All efficiency metrics are measured on a **single fixed NVIDIA T4
(16 GB, Kaggle)**; numbers from any other accelerator are excluded from efficiency tables
(protocol §14).

**R9. Adversarial validity.** Only **label-preserving** perturbations count; each attacked
example must satisfy **USE similarity ≥ 0.90** or it is discarded (protocol §12). Robustness is
read only from validity-filtered attacks.

**R10. Contamination gating.** LLM zero-/few-shot results on any dataset flagged by the
contamination probe (EXP-F0) are excluded from that LLM's zero-shot claims and reported
separately as contamination-suspect (protocol §4.7).

---

## Part 2 — Metric Catalog (why / which RQ / how interpreted)

### Classification quality

**Macro-F1** *(primary)*
- **Why included:** Balances fake and real classes equally under class-imbalanced/real-world
  streams; the harmonic mean of macro-precision/recall resists the accuracy inflation that a
  majority class produces. It is the field-comparable headline metric and the axis of every
  hypothesis threshold.
- **Which RQ:** All (RQ1–RQ5); it is the quantity in ΔF1 (RQ1), the leakage/shortcut ratios
  (RQ2), the family comparisons (RQ4), and the Pareto axis (RQ5).
- **How interpreted:** Higher is better. Differences are believed only after STATS-CONF (R6);
  a raw gap without adjusted-p + effect size + CI is treated as noise (R5). Cross-experiment
  comparisons use identical test sets.

**Accuracy**
- **Why included:** Universally reported in the corpus (all primary studies), so it anchors
  comparability with prior WELFake numbers (P002, P003, P012); also the substrate for McNemar.
- **Which RQ:** RQ1–RQ4 (reported alongside macro-F1); primary readout under attack (RQ3).
- **How interpreted:** Diagnostic only. On balanced WELFake it tracks macro-F1; large
  accuracy–macro-F1 gaps flag class-skewed errors and are investigated in error analysis.

**Macro-Precision / Macro-Recall**
- **Why included:** Decompose macro-F1 so a change can be attributed to false-positive vs
  false-negative behavior; near-universally reported in the corpus.
- **Which RQ:** RQ1–RQ4 (supporting).
- **How interpreted:** Read as a pair. Falling macro-recall under transfer/attack indicates
  missed fakes (costly); falling macro-precision indicates over-flagging of real news.

**Per-class F1 (fake / real)**
- **Why included:** Exposes which class collapses under distribution shift, leakage removal, or
  attack — a single macro number can hide a one-sided failure.
- **Which RQ:** RQ1 (which class fails to transfer), RQ2 (which class the shortcut exploits),
  RQ3 (which class attacks target).
- **How interpreted:** A large fake-vs-real F1 asymmetry is a substantive finding, cross-checked
  against the confusion-matrix asymmetry in error analysis (EXP-G2).

### Threshold-independent ranking

**ROC-AUC**
- **Why included:** Threshold-independent separability; comparable to P003/P009 and endorsed by
  surveys P007/P008; robust to the operating-point choice.
- **Which RQ:** RQ1, RQ4 (ranking quality independent of the 0.5 cutoff).
- **How interpreted:** Higher is better; used to check whether macro-F1 differences reflect
  genuine ranking gaps or merely threshold placement. On saturated data (near 1.0) it is treated
  as ceiling-limited and de-emphasized in favor of PR-AUC.

**PR-AUC**
- **Why included:** More informative than ROC-AUC under class imbalance / rare-fake regimes;
  directly targets the "balanced-benchmark accuracy overstates imbalanced reality" concern.
- **Which RQ:** RQ1 (transfer to imbalanced targets), RQ2 (imbalance interacts with shortcuts).
- **How interpreted:** Higher is better; prioritized over ROC-AUC whenever the test set is
  imbalanced (e.g., transfer targets, DS4 probe).

### Transfer / generalization

**ΔF1 = in-domain macro-F1 − cross-domain macro-F1** *(RQ1 primary readout)*
- **Why included:** The direct measurement of generalization loss that the corpus never
  quantifies with a train-on-A/test-on-B protocol (G1; P006 reports 8–40% drops).
- **Which RQ:** RQ1 (and it is the confirmatory quantity for H2).
- **How interpreted:** Larger ΔF1 = worse generalization. **H2 decision rule:** ΔF1 ≥ 10 points
  for every model → confirm; tested per model via paired bootstrap under BH-FDR. A near-zero ΔF1
  after cross-dedup would instead indicate genuine transfer.

**Std of macro-F1 across transfer targets**
- **Why included:** Captures *stability* of generalization, not just its mean; a model can have
  moderate mean ΔF1 but wild per-target variance.
- **Which RQ:** RQ1 (supporting).
- **How interpreted:** Lower = more consistent transfer; high variance flags target-specific
  brittleness for the discussion, not a confirmatory claim.

### Leakage / shortcut attribution (RQ2)

**Before-vs-after-dedup macro-F1 delta**
- **Why included:** Isolates accuracy attributable to near-duplicate leakage (H3 part 1).
- **Which RQ:** RQ2.
- **How interpreted:** **H3 decision rule:** dedup + source-disjoint split lowers DS1 macro-F1 by
  ≥ 3 points → confirm leakage inflation; tested via paired bootstrap + McNemar under BH-FDR.

**Random-vs-source-disjoint macro-F1 delta**
- **Why included:** Isolates source-fingerprint shortcut (H3 part 2). Reported as N/A with an
  honest note if source is not recoverable (protocol §11).
- **Which RQ:** RQ2.
- **How interpreted:** A drop under source-disjoint splitting means part of in-distribution
  accuracy was source memorization, not veracity signal.

**Source-only / full-text macro-F1 ratio** *(RQ2 confirmatory readout)*
- **Why included:** Quantifies how much veracity is predictable without reading article text —
  the sharpest shortcut test (H4); a novel control (protocol §5).
- **Which RQ:** RQ2.
- **How interpreted:** **H4 decision rule:** ratio ≥ 0.80 on ≥ 1 dataset → benchmark rewards
  shortcuts; full-text vs source-only tested via paired bootstrap + Cliff's delta under BH-FDR.

### Adversarial robustness (RQ3)

**Clean accuracy vs accuracy-under-attack**
- **Why included:** The paired reference needed to express robustness; reporting clean-only
  "overstates real-world reliability" (G6; P002/P003/P010/P012 report clean-only).
- **Which RQ:** RQ3.
- **How interpreted:** The gap (clean − attacked) is the robustness cost; compared per family.

**Attack Success Rate (ASR)** *(RQ3 primary readout)*
- **Why included:** Standardized robustness measure independent of clean accuracy; enables the
  encoder-vs-LLM robustness comparison (H5, replicating P001).
- **Which RQ:** RQ3 (and RQ4, encoder vs decoder).
- **How interpreted:** Lower ASR = more robust. **H5 decision rule:** decoder-LLM ASR < encoder
  ASR (per-family mean ± std, 2-LLM case study) → confirm the P001 asymmetry.

**Mean semantic similarity of successful attacks**
- **Why included:** Validity guard — confirms successful attacks stayed label-preserving
  (≥ 0.90 USE), so ASR is not inflated by meaning-changing edits (R9).
- **Which RQ:** RQ3 (validity control).
- **How interpreted:** Must sit at/above the 0.90 constraint; attacks below it are discarded
  before ASR is computed. Reported for transparency, not ranked.

### Efficiency (RQ5)

**Parameter count / on-disk size (MB)**
- **Why included:** Deployment-relevant size axis; the corpus reports these incomparably (G12).
- **Which RQ:** RQ5.
- **How interpreted:** Pareto axis vs macro-F1. **H6 decision rule:** a compact encoder reaching
  ≥ 95% of best-encoder macro-F1 at < 50% params sits on/above the frontier → confirm.

**Peak VRAM / training wall-clock / inference latency (ms/sample @ b=1,seq=512) / throughput (samples/s @ b=32)**
- **Why included:** Complete, single-accelerator cost picture (train + inference, memory +
  speed), remedying G12's incomparable mix of USD/time/params.
- **Which RQ:** RQ5.
- **How interpreted:** Lower cost at equal macro-F1 is better; all compared only on the fixed T4
  (R8). Latency drives the F1-vs-latency Pareto plot; VRAM gates feasibility on the free tier.

### Calibration & interpretability (support)

**Expected Calibration Error (ECE, 15 bins)**
- **Why included:** Deployment for fact-checking needs trustworthy probabilities, not just
  labels; calibration is unreported across the corpus.
- **Which RQ:** Supports RQ4/RQ5 (a model may win on F1 yet be poorly calibrated).
- **How interpreted:** Lower = better calibrated; probabilistic models only (R7). High ECE
  qualifies any "confidence" claim; reported alongside F1, never traded against it.

**Comprehensiveness / Sufficiency (faithfulness of explanations)**
- **Why included:** Attribution methods must be *faithful*, not just plausible; attention alone
  is demoted to illustrative (protocol §13/#15).
- **Which RQ:** Supports the interpretability deliverable (G5-adjacent), not a hypothesis.
- **How interpreted:** Higher comprehensiveness (removing salient tokens drops confidence) and
  sufficiency (salient tokens alone preserve it) = more faithful; IG/occlusion are trusted only
  when both hold.

### Data-integrity & gating metrics

**Contamination / memorization score (per LLM × dataset)**
- **Why included:** Static-benchmark contamination makes raw zero-shot LLM numbers untrustworthy
  (P006 warning; P002's implausible 12.3% zero-shot GPT-4o).
- **Which RQ:** Gates RQ4 (E3).
- **How interpreted:** Above the flag threshold → that dataset is excluded from the LLM's
  zero-shot claims and reported as contamination-suspect (R10).

**Dedup / cross-dataset overlap counts**
- **Why included:** Make leakage auditable and reproducible (protocol §4.5–§4.6).
- **Which RQ:** Underpins RQ1 (clean transfer) and RQ2 (leakage attribution).
- **How interpreted:** Reported as counts/%; transfer results are only trusted after
  cross-dataset overlap (cosine ≥ 0.90) has been removed.

### Statistical output metrics

- **Paired-bootstrap 95% CI of the difference** — the plausible range of a macro-F1 gap; a CI
  excluding 0 is necessary (with adjusted p) for significance. Answers all confirmatory RQs.
- **McNemar p-value** — accuracy-level paired significance on the same test set; cross-checks the
  bootstrap at the prediction level.
- **Cliff's delta** — non-parametric effect size on per-seed macro-F1; reported with every
  significant result so "significant" is never confused with "large".
- **BH-adjusted p (q=0.05)** — the only p that licenses a confirmatory claim (R5/R6).
- **Friedman + Nemenyi** — overall multi-model ranking, exploratory only (E1/E2); never used for
  a confirmatory verdict.

---

## Part 3 — Experiment-by-Experiment Evaluation Review

For each planned experiment: the evaluation metrics assigned, the **primary readout**, and the
**decision/interpretation**. This confirms every experiment in `experiment_matrix.md` has a
defined, pre-committed evaluation.

| EXP | Metrics evaluated | Primary readout | Decision / interpretation |
|-----|-------------------|-----------------|---------------------------|
| P0–P5 | integrity counts (hashes, dropped rows, dedup/overlap counts) | overlap & dedup counts | Gate: transfer/leakage results valid only after these pass; recorded in stats CSVs. |
| P6 dedup-sensitivity | METRICS-CORE @ {0.90,0.95,0.98} | macro-F1 vs threshold | Report stability; document how threshold choice moves DS1 (robustness of §4.5 choice). |
| **EXP-1** in-dist | METRICS-CORE + ECE; pairwise STATS-CONF | pairwise encoder macro-F1 | **H1**: pairwise BH-adjusted p ≥ 0.05 → encoders indistinguishable; fixes `best_encoder`. |
| **EXP-A1** transfer | METRICS-CORE + TRANSFER; ΔF1 via STATS-CONF | ΔF1 per model | **H2**: ΔF1 ≥ 10 pts (adjusted p, δ, CI) → generalization gap confirmed. |
| EXP-A2 DS4 probe | METRICS-CORE + PR-AUC + ΔF1 | ΔF1 to DS4 | Exploratory domain-shift magnitude; PR-AUC prioritized (imbalanced). |
| EXP-A3 LIAR track | METRICS-CORE + ECE | macro-F1 | Reported separately; never merged into article transfer (§4.2). |
| **EXP-B1** dedup contrast | METRICS-CORE; before/after delta | macro-F1 delta | **H3 (part 1)**: ≥ 3-pt drop → leakage inflation. |
| **EXP-B2** split contrast | METRICS-CORE; rand vs source-disjoint delta | macro-F1 delta | **H3 (part 2)**: drop → source-fingerprint shortcut; N/A if source unrecoverable. |
| **EXP-B3** source-only | METRICS-CORE; source/full ratio via STATS-CONF | ratio | **H4**: ratio ≥ 0.80 → benchmark rewards shortcuts. |
| **EXP-C1** adversarial | ROBUST (clean/attacked acc, ASR, sim) | ASR per family | **H5**: LLM ASR < encoder ASR → robustness asymmetry; validity via ≥0.90 sim (R9). |
| EXP-D1 efficiency | EFFICIENCY + macro-F1 | F1-vs-params/latency Pareto | **H6**: compact ≥95% F1 at <50% params → on frontier. |
| EXP-D2 compression | macro-F1 vs size vs latency | accuracy/size trade-off | Report INT8+KD trade-off explicitly (exploratory). |
| EXP-E1 granularity | METRICS-CORE; Friedman+Nemenyi | ranking per condition | **E2**: ranking change across title/content/title+content (exploratory). |
| EXP-F0 contamination | memorization score/flag | per-LLM×dataset flag | Gate: flagged datasets excluded from that LLM's zero-shot claims (R10). |
| EXP-F1 LLM QLoRA | METRICS-CORE | macro-F1 vs encoders | Encoder-vs-decoder comparison (RQ4); 20 GPU-h cap else partial. |
| EXP-F2 LLM prompted | METRICS-CORE (no ECE) | macro-F1 (separate table) | Contamination-suspect; never merged with fine-tuned results (**E3**). |
| EXP-G1 calibration | ECE + reliability diagrams | ECE | Qualifies confidence claims; probabilistic models only (R7). |
| EXP-G2 error analysis | confusion matrices, error taxonomy, F1 vs length | fake→real vs real→fake asymmetry | Explains *where* models fail (double-coded ≥200 errors). |
| EXP-G3 interpretability | comprehensiveness/sufficiency | faithfulness scores | IG/occlusion trusted only if faithful; attention illustrative only. |
| EXP-G4 stats aggregation | BH-FDR over confirmatory family | adjusted p, δ, CI | Final confirmatory verdicts for H1–H4. |

**Coverage check:** every experiment above has (a) an assigned metric set drawn only from the
frozen suite, (b) a single designated primary readout, and (c) a pre-committed decision rule or
"exploratory" label. No experiment lacks an evaluation, and no metric is introduced outside
protocol §8.

---

## Part 4 — Hypothesis Decision Table (fixed thresholds)

| Hypothesis | RQ | Metric | Fixed rule | Test |
|-----------|----|--------|-----------|------|
| H1 | RQ4 | macro-F1 (pairwise, DS1) | all pairwise BH-adj p ≥ 0.05 | paired bootstrap + McNemar + δ, BH-FDR |
| H2 | RQ1 | ΔF1 | ΔF1 ≥ 10 pts for every model | paired bootstrap on ΔF1, BH-FDR |
| H3 | RQ2 | macro-F1 delta | dedup+source-disjoint drop ≥ 3 pts | paired bootstrap + McNemar, BH-FDR |
| H4 | RQ2 | source/full-text ratio | ratio ≥ 0.80 on ≥ 1 dataset | paired bootstrap + δ, BH-FDR |
| H5 | RQ3 | ASR | LLM ASR < encoder ASR (case study) | per-family mean ± std (n=2 LLMs) |
| H6 | RQ5 | macro-F1 & params | ≥ 95% F1 at < 50% params | ratio vs thresholds, 3 seeds |

E1–E3 use the same metrics but are reported exploratory (Friedman+Nemenyi / separate tables),
never FDR-protected.

---

## Part 5 — Reporting Standards

- Every table reports **mean ± std** over the mandated seeds, with the dataset-version tag and
  protocol version in the caption.
- Every confirmatory claim reports **(BH-adjusted p, Cliff's delta, 95% CI of the difference)**.
- Clean and attacked results always appear **together**; efficiency numbers always cite the
  fixed T4; LLM zero-/few-shot always appears in a **separate, clearly-labeled** table.
- Any "not applicable" (e.g., temporal split, unrecoverable source) is reported explicitly, never
  silently dropped (protocol §4.4, §11).
- Primary endpoints (RQ-A/B/C) are reported in full even if secondary endpoints (RQ-D/E/F) are
  cut for budget (protocol §1).
