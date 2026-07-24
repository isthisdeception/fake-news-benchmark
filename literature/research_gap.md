# Research Gap Analysis — Fake News Detection Literature

Reviewer: Senior NLP faculty assessment
Sources: `literature/master_literature_database.csv` and `literature/pattern_analysis.md`
(which in turn derive only from notes `P001`–`P012`).

Ground rules applied:
- Every gap is supported by **two or more papers**.
- Each gap is scored on **Novelty (1–10)**, **Difficulty**, **Publication Potential**,
  and **Feasibility for one researcher using Kaggle** (free-tier GPU, public datasets).
- A final **ranking** orders all gaps by overall priority for a solo researcher.

Legend — Difficulty / Publication Potential / Feasibility use Low / Medium / High.
"Feasibility (Kaggle)" reflects free GPU quotas (~30 hrs/week, ~16 GB VRAM), dataset
availability, and no multi-GPU/large-video pipelines.

---

## G1. Cross-Dataset / Cross-Domain Generalization Is Not Systematically Evaluated

- **Supporting Papers:** P002, P003, P011, P012, P006, P010
- **Evidence:**
  - P002 and P012 train and test only on WELFake; P003 explicitly lists "experiments rely exclusively on WELFake" as a limitation.
  - P011 reports within-dataset results only (ISOT ≈ 100%, LIAR ≈ 60%, FakeNewsNet noisy) with no train-on-A/test-on-B protocol.
  - P006 (survey) quantifies cross-dataset degradation of ~8–15% (moderate) to >40% (severe) even with identical architectures.
  - P010 builds a combined dataset but does not report leave-one-dataset-out transfer.
- **Why it matters:** Real deployments face distribution shift (new sources, topics, time periods). In-distribution accuracy near 95–100% (P002, P012, P011-ISOT) is almost certainly optimistic; the field lacks a standardized cross-dataset transfer benchmark.
- **Novelty Score:** 6/10 (problem is known, but a rigorous multi-dataset transfer protocol is still missing).
- **Difficulty:** Medium
- **Publication Potential:** High (benchmark + empirical study; strong workshop/journal fit).
- **Feasibility (Kaggle):** High — WELFake, LIAR, ISOT, FakeNewsNet, COVID-19 are all public and text-only; leave-one-out training fits within free GPU limits.

---

## G2. Detection of LLM / AI-Generated News Is Underdeveloped

- **Supporting Papers:** P010, P006, P001
- **Evidence:**
  - P010 is described in its own note as the *first* study to jointly model fake / real / AI-generated news, and its AI class is "near-100% accuracy" / "near-trivially separable" (single-source TuringBench + self-generated titles).
  - P006 (survey) warns that LLM-generated misinformation is "fluent and stylistically indistinguishable from human text" and that static benchmarks suffer contamination.
  - P001 uses GPT-4 to *generate* labels, implicitly acknowledging LLMs both aid and threaten the pipeline.
- **Why it matters:** Generative models now mass-produce plausible misinformation. Existing detectors either ignore the AI-generated class (P001–P005, P011, P012) or make it trivially separable (P010). A realistic, hard AI-vs-human-fake benchmark is missing.
- **Novelty Score:** 8/10
- **Difficulty:** Medium-High
- **Publication Potential:** High (timely; aligns with LLM-safety venues).
- **Feasibility (Kaggle):** Medium-High — open generators (GPT-2/GPT-Neo, and small instruct models via quantization) run on Kaggle; TuringBench and news corpora are public. Challenge is constructing a *non-trivial* AI-generated set (full articles, matched topics), not compute.

---

## G3. Multilingual / Cross-Lingual Fake News Detection Is Scarce

- **Supporting Papers:** P001, P002, P003, P010, P011, P012 (all English-only); P006, P007, P008 (surveys calling for it)
- **Evidence:**
  - CSV `Language(s)` column is English for every primary study (P002, P003, P009, P010, P011, P012) and "Not Reported/English-oriented" for P001, P004, P005.
  - P006, P007, P008 all list multilingual / cross-lingual detection as a top future direction; P008 shows code-mixed settings degrade traditional models.
- **Why it matters:** Misinformation is a global phenomenon; English-only models do not transfer to low-resource or code-mixed contexts, a gap the surveys explicitly flag.
- **Novelty Score:** 6/10
- **Difficulty:** Medium
- **Publication Potential:** High (multilingual NLP venues actively seek this).
- **Feasibility (Kaggle):** Medium — multilingual datasets exist (e.g., M4, code-mixed sets referenced in P006/P008) and XLM-R / mBERT fine-tuning fits Kaggle GPUs; main cost is data curation and label alignment.

---

## G4. Multimodal Detection Is Limited and Modality-Imbalanced

- **Supporting Papers:** P005, P007, P006, P008 (+ text-only majority P001, P002, P003, P011, P012)
- **Evidence:**
  - P005 addresses text+image only and explicitly states video/audio are not handled; it also documents modality imbalance (text dominating image).
  - P007 (survey) catalogs many multimodal datasets but notes datasets are scarce/English-centric; P006 and P008 recommend text+image+audio+video fusion as future work.
  - The majority of primary detectors are text-only (P001, P002, P003, P011, P012).
- **Why it matters:** Real fake news combines text, images, and video; unimodal detectors miss cross-modal inconsistency, and modality imbalance is barely studied (only P005).
- **Novelty Score:** 7/10
- **Difficulty:** High
- **Publication Potential:** High
- **Feasibility (Kaggle):** Low-Medium — text+image is achievable (Fakeddit, Weibo, Twitter-MediaEval); video/audio pipelines (DFDC, FakeAVCeleb from P007) exceed free-tier storage/compute. A solo researcher should scope to text+image only.

---

## G5. Explainability / Interpretability of Transformer Detectors Is Missing

- **Supporting Papers:** P003, P009, P011, P012 (+ P007, P008 future work)
- **Evidence:**
  - P003 states SHAP for deep models was computationally prohibitive (>100 h) and could not be completed, leaving neural models unexplained.
  - P011 and P012 both list explainable predictions (SHAP/LIME/attention) as future work.
  - P009 is the one paper with built-in token-level saliency (Grad/IG/LIME), showing it is feasible but rare in the news domain.
- **Why it matters:** Fact-checking deployment requires trust and auditability; near-black-box classifiers (P002, P010, P011, P012) are hard to operationalize for journalists.
- **Novelty Score:** 6/10
- **Difficulty:** Medium
- **Publication Potential:** Medium-High
- **Feasibility (Kaggle):** High — LIME/SHAP/Integrated Gradients/attention rollout on fine-tuned text models are lightweight and run comfortably within free GPU limits.

---

## G6. Adversarial / Perturbation Robustness Is Rarely Evaluated

- **Supporting Papers:** P001, P011, P009
- **Evidence:**
  - P001 is the only primary study that runs adversarial perturbation testing (TextAttack: drop, insertion, negation, antonym, typo, shuffle) and finds LLMs more robust than BERT-like models.
  - P011 lists adversarial robustness testing explicitly as deferred future work.
  - P009 notes robustness only to synthetic perturbations and omits strong adversarial baselines.
- **Why it matters:** Adversaries actively evade detectors; reporting only clean-test accuracy (P002, P003, P010, P012) overstates real-world reliability. A systematic robustness benchmark across models is absent.
- **Novelty Score:** 7/10
- **Difficulty:** Medium
- **Publication Potential:** High
- **Feasibility (Kaggle):** High — TextAttack / textual perturbation suites run on CPU/single GPU; reuses existing fine-tuned models and public datasets.

---

## G7. Reproducibility Gap: Incomplete Hyperparameter, Scheduler, and Batch-Size Reporting

- **Supporting Papers:** P002, P003, P004, P005, P009, P010, P011, P012
- **Evidence:**
  - Scheduler is Not Reported in P002, P003, P004, P005, P009, P010.
  - Batch size is Not Reported in P003, P004, P005, P011, P012.
  - P011 omits dataset sizes and core hyperparameters (LR, batch, epochs) *despite* an explicit reproducibility emphasis; P012 omits LR/batch/max-length.
- **Why it matters:** The field cannot fairly compare methods when core settings are missing; this undermines every leaderboard-style claim (pattern_analysis §4–6 documents the systemic gap).
- **Novelty Score:** 5/10 (meta-contribution rather than a new method).
- **Difficulty:** Low-Medium
- **Publication Potential:** Medium (reproducibility/benchmark track; resource-paper venues).
- **Feasibility (Kaggle):** High — re-implementing and fully documenting a standardized, reproducible pipeline is ideal solo work; Kaggle notebooks are inherently shareable/reproducible.

---

## G8. Benchmark Saturation and Absence of Temporally Robust / Harder Benchmarks

- **Supporting Papers:** P006, P011, P010, P003
- **Evidence:**
  - P006 reports legacy benchmarks (e.g., ISOT) show >99% accuracy yet degrade sharply on harder sets (LIAR, FakeNewsNet).
  - P011 mirrors this: ISOT ≈ 100% vs LIAR ≈ 60% — evidence of dataset-difficulty imbalance and saturation.
  - P003 flags temporal/topical drift; P010 uses only precollected data (no temporal split).
- **Why it matters:** Saturated benchmarks (ISOT) inflate progress; there is no standardized *temporal* (train-past/test-future) or difficulty-calibrated benchmark, so reported gains may be illusory.
- **Novelty Score:** 7/10
- **Difficulty:** Medium-High
- **Publication Potential:** High (resource/benchmark papers are well-cited).
- **Feasibility (Kaggle):** Medium — requires careful dataset re-splitting (temporal ordering, difficulty stratification) using public data; compute is modest but curation effort is substantial.

---

## G9. No Rigorous, Fair Encoder-vs-Decoder (LLM) Comparison; Zero-Shot LLMs Underperform

- **Supporting Papers:** P001, P002, P010
- **Evidence:**
  - P001 finds BERT-like encoders outperform 7B LLMs (Llama2/Mistral) on classification, but LLMs are more robust — and the note flags an internal contradiction in its own RQ1 conclusion.
  - P002 reports implausibly low zero-shot GPT-4o (12.3%) likely from prompt/label-parsing issues, then near-perfect after fine-tuning (98.6%).
  - P010 uses GPT-family models for generation, not as classifiers, leaving the classifier comparison incomplete.
- **Why it matters:** The encoder-vs-decoder question is central but currently muddied by prompt-sensitivity artifacts and inconsistent protocols; a clean, prompt-controlled comparison is missing.
- **Novelty Score:** 6/10
- **Difficulty:** Medium
- **Publication Potential:** Medium-High
- **Feasibility (Kaggle):** Medium — encoder fine-tuning is easy; LLMs require QLoRA/4-bit (as in P001) to fit ~16 GB, and API models (GPT-4o, P002) cost money. Scope to open 7B models with quantization.

---

## G10. Label Quality, Weak Supervision, and LLM-as-Annotator Reliability

- **Supporting Papers:** P001, P006, P002
- **Evidence:**
  - P001 introduces GPT-4-assisted + human-verified labels (IRR ≈ 72%) and shows AI+human labels beat distant/source-level labels — but the label pipeline is only partially validated.
  - P006 documents annotation biases (annotator political alignment shifts labels; stricter guidelines raise inter-annotator agreement by 15%).
  - P002 uses a nonstandard label convention (0 = fake, 1 = not fake), a documented source of confusion.
- **Why it matters:** Model quality is bounded by label quality; the reliability, bias, and cost trade-offs of LLM-based annotation vs distant supervision vs human labels are not systematically benchmarked.
- **Novelty Score:** 7/10
- **Difficulty:** Medium
- **Publication Potential:** Medium-High
- **Feasibility (Kaggle):** Medium-High — comparing weak/distant/LLM/human-subset labels on an existing corpus is a well-scoped solo study; LLM labeling via quantized open models is affordable.

---

## G11. Class Imbalance and Real-World Distribution Mismatch

- **Supporting Papers:** P001, P003, P006
- **Evidence:**
  - P001 applies SMOTE to counter class imbalance, implying the raw data is skewed.
  - P003 explicitly notes WELFake is balanced whereas real-world distributions are imbalanced and shift over time.
  - P006 lists class imbalance and data noise among core dataset challenges.
- **Why it matters:** Balanced-benchmark accuracy (P002, P003, P012) does not reflect the heavy class imbalance of real streams; imbalance-aware evaluation (PR-AUC, cost-sensitive metrics) is largely absent.
- **Novelty Score:** 5/10
- **Difficulty:** Low-Medium
- **Publication Potential:** Medium
- **Feasibility (Kaggle):** High — re-sampling / cost-sensitive training and imbalanced evaluation on public data is lightweight and solo-friendly.

---

## G12. Efficiency / Compute-Cost Benchmarking Is Not Standardized

- **Supporting Papers:** P002, P003, P010, P011
- **Evidence:**
  - P002 reports fine-tuning cost in USD (GPT-4o $54.30 vs GPT-4o-mini $6.52); P003 reports training time (ALBERT far slower than classical ML).
  - P010 studies pruning/quantization/distillation trade-offs (265M→66M params); P011 reports parameter-efficient variants (0.3–4.3M, "85% smaller than BERT").
  - No shared protocol exists — each paper measures a different efficiency axis (cost, time, params, size).
- **Why it matters:** Deployment needs accuracy-per-compute, but the field reports incomparable efficiency numbers; a standardized accuracy-vs-cost benchmark is missing.
- **Novelty Score:** 5/10
- **Difficulty:** Low
- **Publication Potential:** Medium
- **Feasibility (Kaggle):** High — measuring latency, params, and accuracy across models is straightforward on Kaggle and reproducible.

---

## Ranked Priority List (Solo Researcher, Kaggle)

Ranking weighs **publication potential × feasibility × novelty**, discounting gaps whose
compute/data needs exceed the free tier. Higher rank = better fit for a solo Kaggle project.

| Rank | Gap | Novelty | Difficulty | Pub. Potential | Feasibility (Kaggle) | Supporting Papers |
|------|-----|:-------:|:----------:|:--------------:|:--------------------:|-------------------|
| 1 | G1 Cross-dataset generalization | 6 | Medium | High | High | P002, P003, P011, P012, P006, P010 |
| 2 | G6 Adversarial robustness | 7 | Medium | High | High | P001, P011, P009 |
| 3 | G2 AI-generated news detection | 8 | Med-High | High | Med-High | P010, P006, P001 |
| 4 | G5 Explainability / XAI | 6 | Medium | Med-High | High | P003, P009, P011, P012 |
| 5 | G8 Benchmark saturation / temporal split | 7 | Med-High | High | Medium | P006, P011, P010, P003 |
| 6 | G10 Label quality / LLM-as-annotator | 7 | Medium | Med-High | Med-High | P001, P006, P002 |
| 7 | G3 Multilingual / cross-lingual | 6 | Medium | High | Medium | P001–P003, P010–P012, P006–P008 |
| 8 | G9 Encoder-vs-decoder comparison | 6 | Medium | Med-High | Medium | P001, P002, P010 |
| 9 | G7 Reproducibility standardization | 5 | Low-Med | Medium | High | P002–P005, P009–P012 |
| 10 | G11 Class imbalance / real-world dist. | 5 | Low-Med | Medium | High | P001, P003, P006 |
| 11 | G12 Efficiency / cost benchmarking | 5 | Low | Medium | High | P002, P003, P010, P011 |
| 12 | G4 Multimodal detection | 7 | High | High | Low-Med | P005, P007, P006, P008 |

### Professor's recommendation
For a single researcher on Kaggle, **G1 (cross-dataset generalization)** and **G6 (adversarial robustness)** are the highest-yield starting points: both are supported by multiple papers, address demonstrable over-optimistic reporting, reuse public text datasets and existing fine-tuned models, and fit free-tier compute. **G2 (AI-generated news)** is the highest-novelty option and is publishable, but its scientific value hinges on constructing a *hard* (topic-matched, full-article) AI-generated set rather than the trivially separable data seen in P010. **G4 (multimodal)** is scientifically important but is ranked last for solo feasibility because video/audio pipelines exceed the free tier; if attempted, it should be scoped to text+image only.

A strong, cohesive single paper could combine **G1 + G6 + G7**: a reproducible cross-dataset benchmark with adversarial stress-testing and full hyperparameter disclosure — directly remedying the three systemic weaknesses documented across the corpus.
