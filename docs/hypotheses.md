# Testable Hypotheses — Fake News Detection Benchmark (WELFake-anchored)

Author role: Senior NLP researcher
Grounding documents: `docs/research_questions.md`, `docs/final_protocol.md` (v1.0, FROZEN),
`literature/master_literature_database.csv` (P001–P012).

Status and scope. Hypotheses **H1–H6 are the pre-registered, confirmatory set** frozen in
`final_protocol.md` §3; they are fixed BEFORE any experiment is run and are the only hypotheses
eligible for confirmatory claims (protocol §10). This document restates each with a full
scientific rationale, its supporting literature, and the exact experiment(s) that test it, and
maps each to the research questions in `docs/research_questions.md`. Hypotheses labeled
**E-** are exploratory (hypothesis-generating), reported as such and never used for confirmatory
claims. No hypothesis here introduces any dataset, model, or setting beyond the frozen protocol.

Global test machinery (applies to every confirmatory hypothesis unless stated otherwise):
paired **bootstrap test on the macro-F1 difference** (10,000 stratified resamples, 95% CI of the
difference, two-sided p), **McNemar** at the accuracy level, **Cliff's delta** effect size,
**Benjamini–Hochberg FDR at q = 0.05** across the confirmatory family; **5 seeds
{13,21,42,87,100}** for primary endpoints, **3 seeds {13,42,100}** for secondary
(protocol §9–§10). Test sets always use the natural class distribution (protocol §7).

---

## H1 — Top encoders are statistically indistinguishable in-distribution
*(RQ4; frozen H1)*

**Hypothesis statement.** Among the five encoders (`bert-base-uncased`, `roberta-base`,
`deberta-v3-base`, `distilbert-base-uncased`, `albert-base-v2`), pairwise macro-F1 differences
on the WELFake in-distribution test set are **not** statistically significant after BH-FDR
correction. (Null of "no difference" is expected to survive.)

**Scientific rationale.** WELFake is a balanced, comparatively easy article corpus on which
transformer scores cluster near the ceiling (P002 fine-tuned ≈ 98.6%; P003 ALBERT macro-F1 up
to 0.99; P012 ≈ 95.3%). When multiple strong models saturate the same benchmark, apparent
"winners" are usually within noise; leaderboard-style rankings without significance testing are
therefore likely to be artifacts. If H1 holds, it demonstrates that single-dataset accuracy
cannot separate modern encoders — motivating the stress tests (H2–H5).

**Related literature.** P002 (WELFake, BERT/GPT-4o, near-perfect fine-tuned), P003 (WELFake,
ALBERT best with McNemar significance testing), P012 (WELFake, BERT, 95.3%); P011 supplies the
methodological precedent for paired significance testing (paired t-test, Cohen's d, 95% CI).

**Experiment(s).** Fine-tune all five encoders on WELFake under the single frozen config
(protocol §6–§7), 5 seeds; compute all pairwise macro-F1 differences on the WELFake test set;
apply paired bootstrap + McNemar + Cliff's delta with BH-FDR. This pairwise-encoder set is the
core of the confirmatory family (protocol §10).

---

## H2 — Every model loses ≥ 10 macro-F1 points under cross-dataset transfer
*(RQ1; frozen H2)*

**Hypothesis statement.** For every model family, `ΔF1 = in-domain macro-F1 (WELFake) −
cross-domain macro-F1` is **≥ 10 points** under leave-one-dataset-out transfer across the
article-type set {WELFake, ISOT, FakeNewsNet}.

**Scientific rationale.** Detectors partly learn source-, topic-, and style-specific regularities
that do not transfer; the survey evidence quantifies cross-dataset drops of ~8–15% up to >40%
even with identical architectures (P006). Because prior WELFake work reports only in-distribution
scores (P002, P003, P012), the true generalization gap is unmeasured. A ≥ 10-point drop would
confirm that near-ceiling in-distribution numbers overstate deployable performance.

**Related literature.** P006 (survey quantifying cross-dataset degradation), P011 (within-dataset
only: ISOT ≈ 100% vs LIAR ≈ 60%, no train-on-A/test-on-B protocol), P010 (combined dataset but no
leave-one-dataset-out), P002/P003/P012 (WELFake-only baselines to transfer *from*).

**Experiment(s).** Train each classical and encoder model on WELFake; evaluate in-distribution
and on {ISOT, FakeNewsNet} after inter-dataset near-duplicate removal (cosine ≥ 0.90,
protocol §4.5); report ΔF1 and std across targets (protocol §8, RQ-A); 5 seeds; ΔF1 is in the
confirmatory family, tested via paired bootstrap with BH-FDR.

---

## H3 — Dedup + source-disjoint splitting lowers WELFake macro-F1 by ≥ 3 points
*(RQ2; frozen H3)*

**Hypothesis statement.** Removing near-duplicates (cosine ≥ 0.95) and enforcing a
source-disjoint split reduces WELFake in-distribution macro-F1 by **≥ 3 points** relative to the
random-split, non-deduplicated baseline.

**Scientific rationale.** Random splits let near-identical articles (and same-source items) fall
on both sides of the split, so part of measured accuracy reflects memorized duplicates and
source fingerprints rather than veracity signal. Removing that leakage should measurably lower
the score; the size of the drop quantifies the inflation. This isolates *leakage* as one
mechanism behind saturated WELFake numbers.

**Related literature.** P006 (LLM-era static-benchmark contamination and cross-dataset caveats),
P011 (its noisy small-split behavior on FakeNewsNet/LIAR illustrates split-sensitivity), P002
(down-samples WELFake to an easy 5k subset — an example of how sampling choices inflate scores).

**Experiment(s).** Report WELFake macro-F1 **before vs after** exact+near-duplicate removal and
**random vs source-disjoint** split (protocol §4.5, §11); dedup-threshold sensitivity
{0.90,0.95,0.98} run once on WELFake; deliverable `results/leakage_analysis.csv`; 5 seeds;
before/after contrast tested with paired bootstrap + McNemar under BH-FDR.

---

## H4 — A source/metadata-only classifier captures most of the signal (shortcut evidence)
*(RQ2; frozen H4)*

**Hypothesis statement.** A classifier trained on **source/metadata or non-content surface
features only (no article text)** reaches **≥ 0.80 of the full-text model's macro-F1** on at
least one dataset.

**Scientific rationale.** If veracity can be predicted at 80% of full-text performance without
reading the article, the benchmark rewards spurious shortcuts (source identity, length,
punctuation, capitalization) rather than genuine content understanding. This is the sharpest
falsifiable test of the "inflated accuracy = artifact" concern and, being a novel control, is a
distinct contribution (protocol §5, design-review "high-value addition").

**Related literature.** P006 (documents dataset-quality/shortcut and contamination risks), P010
(shows AI-generated class near-trivially separable from single-source artifacts — a shortcut
analogue), P002 (nonstandard labels and easy subset — how construct problems inflate scores).

**Experiment(s).** Train the source/metadata-only baseline (protocol §5) and compare its
macro-F1 to the full-text model on WELFake (and other datasets where a source field exists);
full-text vs source-only is in the confirmatory family (protocol §11); paired bootstrap +
Cliff's delta under BH-FDR; recorded in `results/leakage_analysis.csv`.

---

## H5 — Decoder LLMs degrade less than encoders under adversarial perturbation
*(RQ3, RQ4; frozen H5)*

**Hypothesis statement.** Under label-preserving adversarial perturbation of the WELFake test
set, the decoder-only LLMs (Mistral-7B-Instruct, Llama-3.1-8B-Instruct, QLoRA fine-tuned) show a
**smaller drop in accuracy (lower Attack Success Rate)** than the fine-tuned encoders,
replicating P001.

**Scientific rationale.** Larger pretrained decoders encode broader lexical/semantic redundancy,
so surface perturbations (typos, synonym swaps, paraphrase) are less likely to move them across
the decision boundary than the narrower features encoders rely on. P001 reports exactly this
encoder-vs-LLM robustness asymmetry; testing it under our controlled, contamination-audited
protocol turns an isolated finding into a replicable claim. Scoped explicitly as an n=2 LLM case
study (protocol §5).

**Related literature.** P001 (only primary study running adversarial perturbation; finds LLMs
more robust than BERT-like models, though its RQ1 conclusion is internally contradictory — a
reason to re-test cleanly), P009 (robustness to *synthetic* perturbations only), P011
(adversarial robustness deferred to future work).

**Experiment(s).** Apply label-preserving attacks (character typos, embedding-constrained synonym
substitution, benign insertion, back-translation) with USE similarity ≥ 0.90; negation/antonym
prohibited (protocol §12); report clean vs attacked accuracy, Attack Success Rate, and mean
similarity of successful attacks for encoders + classical + the 2 LLMs; compare per-family ASR
(5-seed mean ± std).

---

## H6 — Compact encoders retain ≥ 95% of the best encoder's macro-F1 at < 50% params
*(RQ5; frozen H6)*

**Hypothesis statement.** DistilBERT and/or ALBERT achieve **≥ 95% of the best encoder's WELFake
macro-F1 while using < 50% of its parameters**, placing them on the accuracy-vs-compute Pareto
frontier.

**Scientific rationale.** On a saturated, balanced corpus the marginal accuracy of extra
parameters is small, so distilled/factorized encoders should recover nearly all of the accuracy
at a fraction of the cost. Confirming this reframes the "which model" question as an
efficiency-per-accuracy question and yields a directly deployable recommendation.

**Related literature.** P003 (ALBERT best on WELFake while far cheaper than heavier deep models),
P010 (pruning/quantization/distillation trade-offs, 265M→66M), P011 (compact 0.3–3.1M-param
transformers "~85% smaller than BERT" with minimal loss), P012 (progressive BERT training halves
training time) — collectively motivating compact-model parity.

**Experiment(s).** Under the frozen config, measure macro-F1 and compute cost (params, on-disk
size, peak VRAM, training wall-clock, inference latency ms/sample at batch=1 seq=512,
throughput) on a **single fixed NVIDIA T4** (protocol §14); plot macro-F1 vs params and vs
latency; 3 seeds (secondary endpoint). Compact-vs-best-encoder ratio evaluated against the 95% /
50% thresholds.

---

## Exploratory hypotheses (not confirmatory)

These are hypothesis-generating, reported with exploratory labeling only (protocol §3, §10);
they use the same metrics but make no FDR-protected confirmatory claim.

- **E1 — Classical TF-IDF models degrade *more* than encoders under transfer (RQ1).** Rationale:
  bag-of-words features are more source/topic-specific than contextual embeddings, so their ΔF1
  should exceed encoders'. Tested from the same RQ1 transfer matrix by comparing family-level
  ΔF1 distributions (Friedman + Nemenyi, exploratory).
- **E2 — Input granularity changes model ranking (RQ-E, protocol §6.1/§2).** Rationale: title-only
  input strips body evidence, which may reorder families relative to title+content. Tested on
  {WELFake, ISOT, FakeNewsNet} across title / content / title+content conditions; ranking
  stability assessed with Friedman + Nemenyi. (P003 precedent: input-granularity benchmark.)
- **E3 — Zero-/few-shot LLM scores are contamination-inflated relative to QLoRA fine-tunes
  (RQ4).** Rationale: pretraining exposure can memorize benchmark items, inflating zero-shot
  numbers. Tested via the per-LLM guided prefix-completion contamination probe (protocol §4.7);
  any dataset flagged contaminated is excluded from that LLM's zero-shot claims. (P002's
  implausible 12.3% zero-shot GPT-4o motivates treating zero-shot numbers cautiously.)

---

## Hypothesis ↔ RQ ↔ gap map

| Hypothesis | RQ | Frozen RQ | Gap | Confirmatory? |
|-----------|----|-----------|-----|:-------------:|
| H1 | RQ4 | RQ-F / H1 family | G9 | Yes |
| H2 | RQ1 | RQ-A | G1 | Yes |
| H3 | RQ2 | RQ-B | G1, G8 | Yes |
| H4 | RQ2 | RQ-B | G1, G8 | Yes |
| H5 | RQ3, RQ4 | RQ-C | G6, G9 | Yes |
| H6 | RQ5 | RQ-D | G12, G7 | No (secondary) |
| E1 | RQ1 | RQ-A | G1 | No (exploratory) |
| E2 | — | RQ-E | G1/G7 | No (exploratory) |
| E3 | RQ4 | RQ-F | G9, G2 | No (exploratory) |

H1–H5 are the primary confirmatory program; H6 is a secondary (budget-permitting) endpoint;
E1–E3 are exploratory. All directional thresholds (≥ 10 points, ≥ 3 points, ≥ 0.80, ≥ 95% /
< 50%) are fixed here in advance and are the values against which the experiments will be judged.
