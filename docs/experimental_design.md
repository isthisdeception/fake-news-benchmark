# Experimental Design — Fake News Detection Benchmark

Design synthesized from the reviewed corpus (P001–P012), the derived
`literature/pattern_analysis.md`, `literature/research_gap.md`, and
`literature/reviewer_observations.md`. Design decisions are justified by paper IDs:
we *adopt* documented best practices and *remediate* documented weaknesses.

Guiding principle: build the reproducible, leakage-audited, statistically-tested,
cross-dataset + adversarial benchmark that the corpus repeatedly calls for but never
delivers (research gaps G1 + G6 + G7). Scoped for a single researcher on Kaggle.

---

## 1. Research Questions

- **RQ1 (In-distribution).** How do model families — classical ML, encoder transformers (BERT/RoBERTa/DeBERTa/DistilBERT/ALBERT), and instruction/decoder LLMs — compare on standard in-distribution fake news classification? *(Extends the incomplete/contradictory comparisons in P001, P002, P003 with controlled, tested results.)*
- **RQ2 (Cross-dataset generalization).** How much does performance degrade under leave-one-dataset-out transfer (train on A, test on B)? *(Directly targets gap G1; P002/P003/P011/P012 never test transfer, and P006 reports 8–40% degradation.)*
- **RQ3 (Adversarial robustness).** How robust is each model to text perturbations (typo, synonym, negation, insertion, paraphrase)? *(Only P001 tests this; P011 defers it. Gap G6.)*
- **RQ4 (Leakage sensitivity).** How much of reported accuracy is attributable to near-duplicate/source-style leakage on merged corpora such as WELFake? *(Reviewer-flagged risk for P002, P012; unaddressed field-wide.)*
- **RQ5 (Efficiency).** What is the accuracy-per-compute trade-off (params, latency, training time) across families? *(Unifies the incomparable efficiency axes of P002, P003, P010, P011.)*
- **RQ6 (Input granularity).** Does classification on title-only vs. content-only vs. title+content change the ranking of models? *(Adopts P003's granularity axis; tests whether P010's title-only design is a confound.)*

---

## 2. Hypotheses

- **H1.** Encoder transformers (esp. RoBERTa/DeBERTa) will lead in-distribution, but differences among top transformers will be **statistically insignificant** once confidence intervals are computed. *(Motivated by P010's 96.66% vs 96.45% "improvement" reported with no test; P002's tied 98.6% models.)*
- **H2.** All models will show substantial cross-dataset degradation (hypothesized ≥10–20% macro-F1 drop), with classical ML degrading most and large pretrained transformers degrading least. *(P006 degradation ranges; P011 ISOT≈100% vs LIAR≈60%.)*
- **H3.** Decoder LLMs will be **more robust to adversarial perturbations** than encoders despite lower clean accuracy. *(Replicates and stress-tests P001's central finding.)*
- **H4.** Removing near-duplicates and enforcing source-disjoint splits will **measurably lower** WELFake accuracy, revealing leakage-driven inflation. *(Tests the P002/P012 leakage risk.)*
- **H5.** Parameter-efficient models (DistilBERT, ALBERT, compact transformers) will achieve ≥95% of the best model's F1 at a fraction of the compute. *(Motivated by P003 ALBERT, P011 compact variants, P010 distillation.)*
- **H6.** Content-bearing input (content / title+content) will outperform title-only across all models. *(P003 shows content adds substantial signal.)*

---

## 3. Dataset Justification

Selected for public availability (Kaggle-accessible), recurrence in the corpus, and complementary difficulty:

| Dataset | Role | Why (paper support) |
|---------|------|---------------------|
| **WELFake** (~72k) | Primary in-distribution + leakage case study | Most common primary dataset (P002, P003, P012); balanced; enables direct comparison to three papers. Its merged-source construction makes it the ideal leakage testbed (Reviewer §4). |
| **LIAR** (~12.8k) | "Hard" short-statement set | Recurs in P006, P010, P011; consistently *hard* (P011 ≈ 60%), counters benchmark saturation. |
| **ISOT** | Saturation reference | P011/P006 show ≈100%/>99% — included precisely to demonstrate saturation and as an easy anchor. |
| **FakeNewsNet** (PolitiFack + GossipCop) | Cross-domain transfer target | Used by P010, P011; cataloged by P006, P007; has title+content for RQ6. |
| **COVID-19 Fake News** | Domain-shift probe | Topical/temporal domain distinct from politics/celebrity (P010). |

- **Language:** English (matches corpus; multilingual deferred — gap G3 out of solo Kaggle scope, per research_gap.md).
- **Cross-dataset protocol** uses the datasets above in leave-one-out fashion (RQ2).
- **Explicitly excluded:** video/audio multimodal corpora (DFDC/FakeAVCeleb from P007) — exceed free-tier compute (gap G4 feasibility note).

---

## 4. Model Selection

Chosen to span every family present in the corpus, enabling a fair, controlled comparison (remedying P010/P012's borrowed cross-dataset baselines):

- **Classical ML baselines:** Logistic Regression, Linear SVM, Random Forest, LightGBM on TF-IDF (1–2 grams). *(P003, P010; ensures strong non-neural baselines that P002/P011 lacked.)*
- **Encoder transformers:** BERT-base, RoBERTa-base, DeBERTa-v3-base, DistilBERT, ALBERT. *(Union of P001, P002, P003, P004, P010, P011.)*
- **Decoder / instruction LLM (robustness + zero/few-shot):** one open 7B model (e.g., Mistral-7B-Instruct) via 4-bit QLoRA, plus zero-shot and 5-shot prompting. *(P001 QLoRA recipe; P002 zero-vs-fine-tuned; keeps within ~16 GB Kaggle VRAM.)*
- **Efficiency reference:** DistilBERT and a compact transformer variant. *(P010 distillation, P011 compact 0.3–3.1M models.)*

All neural models use identical preprocessing, splits, and evaluation to guarantee comparability (controlled experiment, not literature-borrowed numbers).

---

## 5. Experimental Protocol

**Preprocessing (fixed, documented — remediating P005's missing pipeline and P011's omissions):**
1. Unicode NFC normalization; lowercasing.
2. Remove URLs, HTML, mentions, hashtags, non-linguistic symbols (P003, P010, P012).
3. For classical ML: stopword removal (NLTK) + WordNet lemmatization + TF-IDF, **vectorizer fit on train only** (P003's anti-leakage practice).
4. For transformers: model-native subword tokenization, max length 512 (title+content), no stopword removal.
5. **Leakage audit (novel, RQ4):** exact + near-duplicate removal via TF-IDF cosine ≥ 0.95 (P010's recipe), plus a **source-disjoint** WELFake variant (no source overlap across splits). Report metrics *before and after* dedup.

**Splitting:**
- Stratified 70/15/15 train/val/test with **fixed seeds** (P003, P011).
- **Temporal split** where timestamps exist (train past / test future) to probe drift (P003, P006).
- **Cross-dataset (RQ2):** train on one dataset, evaluate on each other dataset unchanged.

**Input granularity (RQ6):** run title-only, content-only, title+content (P003).

**Repetition:** ≥3 seeds per configuration; report mean ± std and **report all runs, not the best** (correcting P011's best-run reporting).

---

## 6. Training Strategy

Standardized from the corpus modal configuration (pattern_analysis §4–8), fully disclosed:

- **Optimizer:** AdamW (P001, P010, P012), weight decay 0.01.
- **Scheduler:** linear warm-up (6% steps) + cosine annealing (P001, P011, P012) — explicitly reported (fixing corpus-wide omission).
- **Learning rate:** 2e-5 for encoders (P001, P002); tuned in [1e-5, 5e-5] via validation.
- **Batch size:** 16 (modal value; P001, P010).
- **Epochs:** up to 10 with early stopping on val macro-F1 (P001, P010).
- **Max length:** 512 (P002, P009); 128 for title-only.
- **LLM:** 4-bit QLoRA (r=64, α=16, dropout 0.2) per P001; plus zero/5-shot with a fixed model-agnostic JSON prompt (P002) to avoid P002's zero-shot parsing artifact.
- **Class imbalance:** class-weighted loss; SMOTE (if used) applied **inside training folds only** (explicitly correcting P001's ambiguous SMOTE ordering) — RQ on imbalance uses natural (imbalanced) test distributions, not just balanced subsets.
- **Compute:** single Kaggle GPU; log wall-clock and VRAM.

---

## 7. Evaluation Metrics

- **Primary:** Macro-Precision, Macro-Recall, **Macro-F1**, Accuracy (universal in corpus).
- **Discrimination:** ROC-AUC and **PR-AUC** (PR-AUC added for imbalanced/real-world distributions — absent in most corpus papers).
- **Per-class F1** for fake and real separately (P005's good practice).
- **Robustness (RQ3):** accuracy/F1 under each perturbation + **Attack Success Rate** and average confidence drop (extends P001/TextAttack).
- **Generalization (RQ2):** transfer degradation ΔF1 = in-domain F1 − cross-domain F1; std of F1 across target datasets (P009-style stability).
- **Calibration:** Expected Calibration Error (novel vs corpus; supports trustworthy deployment).
- **Efficiency (RQ5):** parameters, model size (MB), training time, inference latency/throughput (P002, P010).

Report a fixed metric suite for **every** model on **every** dataset (no borrowed numbers).

---

## 8. Statistical Analysis

Directly remedies the corpus's biggest weakness (Reviewer §5; only P003/P005/P011 tested significance):

- **Pairwise model comparison:** McNemar's test with **Holm–Bonferroni** correction (adopt P003) for same-test-set comparisons.
- **Confidence intervals:** bootstrap 95% CIs (≥1,000 resamples) on Macro-F1 for every model/dataset cell.
- **Effect size:** Cohen's d / odds ratios (adopt P011), so we never report a raw Δ without magnitude + uncertainty.
- **Cross-config:** Friedman test across datasets + Nemenyi post-hoc for overall model ranking.
- **Decision rule:** a difference is claimed only if the corrected p < 0.05 **and** the 95% CIs are non-overlapping — explicitly preventing the P002/P010/P012 "0.1–1.2% unqualified improvement" error.

---

## 9. Error Analysis

- **Confusion structure:** per-dataset confusion matrices; fake→real vs real→fake asymmetry (P010, P012).
- **Qualitative failure taxonomy:** sarcasm/satire, ambiguous/figurative language, atypical sources, topical drift — categories flagged by P003; sample and label failure cases.
- **Leakage attribution (RQ4):** compare misclassifications before vs after dedup/source-disjoint splitting; quantify accuracy attributable to source-style shortcuts (P002, P012 risk).
- **Difficulty stratification:** performance vs article length, and easy (ISOT) vs hard (LIAR) buckets (P006, P011).
- **Interpretability (gap G5):** token-level attributions (LIME / Integrated Gradients / attention rollout, per P009) on a sample of errors to check whether models rely on content vs stylistic artifacts.
- **Adversarial failure modes:** which perturbation types most degrade which family (P001).

---

## 10. Efficiency Analysis

Standardized accuracy-per-compute reporting (unifying P002 cost, P003 time, P010 compression, P011 params):

- Report params, size (MB), peak VRAM, training wall-clock, and inference latency/throughput for every model on identical hardware.
- **Pareto frontier:** Macro-F1 vs. (a) parameters and (b) inference latency, to identify best accuracy-per-compute (H5).
- **Compression study (optional):** dynamic quantization + knowledge distillation on the best encoder (P010), reporting the accuracy/size trade-off explicitly (correcting P010's under-discussed 96.65%→92.78% drop).

---

## 11. Threats to Validity

- **Internal — Data leakage:** merged-source duplicates and source-style shortcuts (P002, P012). *Mitigation:* dedup (cosine ≥0.95) + source-disjoint splits + before/after reporting (RQ4).
- **Internal — Label quality:** WELFake/LIAR label noise and annotation bias (P006). *Mitigation:* report results on multiple datasets; do not over-claim from any single label source.
- **Internal — SMOTE/oversampling leakage:** (P001 risk). *Mitigation:* resampling strictly inside training folds.
- **Construct — Title-only confound:** headline-only tasks can be artificially easy (P010). *Mitigation:* granularity ablation (RQ6).
- **Construct — Benchmark saturation:** ISOT ≈100% is uninformative (P006, P011). *Mitigation:* include hard (LIAR) and cross-dataset/temporal settings.
- **External — English-only / single-language:** limits generalization (P003, P006–P008). *Acknowledged scope limit;* multilingual deferred (gap G3).
- **External — Modality:** text-only; multimodal cues excluded (P005, P007). *Acknowledged.*
- **Conclusion — Statistical validity:** small deltas mistaken for improvements (P002, P010, P012). *Mitigation:* mandatory significance tests + CIs + effect sizes (§8).
- **Reproducibility — API/proprietary models:** (P002). *Mitigation:* use only open-weight models; publish code, seeds, splits, and full hyperparameters.

---

## 12. Expected Contributions

1. **A reproducible, leakage-audited fake-news benchmark** across five public datasets with fully disclosed splits, seeds, and hyperparameters — the reproducible protocol the corpus lacks (gap G7; Reviewer §8).
2. **The first controlled cross-dataset generalization study** (leave-one-dataset-out + temporal) quantifying degradation across classical/encoder/decoder families (gap G1; RQ2/H2).
3. **A systematic adversarial-robustness evaluation** across families, testing and extending P001's encoder-vs-LLM robustness finding (gap G6; RQ3/H3).
4. **Quantified leakage effect** on WELFake — evidence of how much reported accuracy is inflated by duplicates/source-style shortcuts (RQ4/H4), a caution the field has ignored.
5. **Statistically rigorous rankings** with significance tests, bootstrap CIs, and effect sizes — correcting the corpus-wide habit of unqualified improvement claims (Reviewer §5).
6. **A standardized accuracy-per-compute (Pareto) analysis** unifying the fragmented efficiency reporting of P002/P003/P010/P011 (RQ5/H5).
7. **Open-source release** (code, configs, splits, perturbation sets, and per-model result tables) enabling exact replication on Kaggle.
