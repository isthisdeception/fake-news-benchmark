# Research Questions — Fake News Detection Benchmark (WELFake-anchored)

Author role: Senior NLP researcher
Grounding documents: `docs/final_protocol.md` (v1.0, FROZEN), `literature/research_gap.md`,
`literature/pattern_analysis.md`.

Selected dataset: **WELFake (DS1)** — the primary in-distribution and leakage case-study
corpus, chosen because it is the single most common primary-study dataset in the reviewed
corpus (P002, P003, P012; `pattern_analysis.md` §1) and therefore the strongest cross-paper
comparison anchor.

Scope note: These questions are the publication-facing articulation of the confirmatory
program pre-registered in `final_protocol.md` §2–§3. They do **not** introduce any setting,
dataset, or model beyond the frozen protocol; each maps to a frozen RQ (RQ-A…RQ-F), a
directional hypothesis (H1…H6), and the statistical machinery in §10. Every question is
stated as a *measurable contrast with a decision rule*, not as a description.

---

## RQ1 — Cross-dataset generalization (in-distribution → transfer)

**Question.** When a fixed model, trained on WELFake, is evaluated under leave-one-dataset-out
transfer to the other article-type corpora (ISOT, FakeNewsNet), by how many macro-F1 points
does performance degrade relative to its in-distribution WELFake score, and is the degradation
of comparable magnitude across the classical (TF-IDF) and encoder families?

**Operationalization (how it is answered experimentally).**
- Train each classical and encoder model on WELFake; evaluate in-distribution (WELFake test)
  and cross-dataset on {ISOT, FakeNewsNet}, restricted to the article-type transfer set
  {DS1, DS2, DS3} so text-type is held constant (protocol §4.2).
- Primary quantity: `ΔF1 = in-domain macro-F1 − cross-domain macro-F1` and the std of macro-F1
  across transfer targets (protocol §8, §2 RQ-A).
- Inter-dataset near-duplicate removal (TF-IDF cosine ≥ 0.90) is applied before every transfer
  so measured degradation is not masked by overlap (protocol §4.5).
- Decision rule: 5 seeds {13,21,42,87,100}; paired bootstrap on the ΔF1 with 95% CI and
  BH-FDR at q=0.05 (protocol §9–§10). Confirmatory link: **H2** (every model loses ≥ 10
  macro-F1 points under transfer).

**Justification.** Directly targets **G1** (cross-dataset generalization is not systematically
evaluated; P002/P012 test only on WELFake, P003 lists WELFake-only as a limitation, P011
reports within-dataset only, P006 quantifies 8–40% cross-dataset drops). Reported
in-distribution accuracy near 95–100% is "almost certainly optimistic" (`research_gap.md`
G1); a leave-one-dataset-out contrast is the minimal experiment that converts that suspicion
into a measured quantity. G1 is the corpus's top-ranked, highest-yield gap.

---

## RQ2 — Shortcut/leakage attribution on WELFake

**Question.** What fraction of WELFake's in-distribution macro-F1 is attributable to
near-duplicate leakage and non-veracity surface/source shortcuts, rather than to genuine
veracity signal?

**Operationalization.**
- Contrast 1 (leakage): WELFake test macro-F1 **before vs after** exact + near-duplicate
  removal (cosine ≥ 0.95), and **random split vs source-disjoint split** where per-article
  source is recoverable (protocol §4.5, §11).
- Contrast 2 (shortcut): train the **source/metadata-only baseline** — no article text, only
  source/metadata or surface features (length, punctuation, capitalization) — and compare its
  macro-F1 to the full-text model (protocol §5, §11).
- Deliverable `results/leakage_analysis.csv`; dedup-threshold sensitivity {0.90,0.95,0.98} run
  once on WELFake (protocol §4.5).
- Decision rule: full-text vs source-only is in the confirmatory family; paired bootstrap +
  McNemar + BH-FDR (protocol §10). Confirmatory links: **H3** (dedup + source-disjoint lowers
  WELFake macro-F1 by ≥ 3 points) and **H4** (source-only reaches ≥ 0.80 of full-text macro-F1
  on ≥ 1 dataset).

**Justification.** Sharpens the mechanism behind **G1/G8**: saturated, single-dataset scores
(ISOT ≈ 100%, WELFake ≈ 95–100%; `research_gap.md` G8, P006/P011) may reflect leakage and
dataset artifacts rather than detection ability. No reviewed paper decomposes accuracy into
"veracity signal vs shortcut," and the source/metadata-only control is flagged in the design
review as a novel, high-value addition (protocol §5). This question makes the inflation
hypothesis falsifiable.

---

## RQ3 — Adversarial robustness across model families

**Question.** Under label-preserving adversarial perturbation of the WELFake test set, how far
does each model family's accuracy fall from its clean accuracy, and does robustness (Attack
Success Rate) differ systematically between classical, encoder, and decoder-LLM families?

**Operationalization.**
- Apply label-preserving attacks only — character typos, embedding-constrained synonym
  substitution, benign insertion, back-translation paraphrase — with a Universal Sentence
  Encoder similarity constraint ≥ 0.90; negation/antonym substitution are prohibited because
  they can flip the gold label (protocol §12).
- Metrics: clean accuracy, accuracy-under-attack, Attack Success Rate, mean semantic similarity
  of successful attacks (protocol §8, §12). Applied to all encoders + classical + (budget
  permitting) the 2 LLMs.
- Decision rule: report per-family ASR with 5-seed mean ± std; confirmatory link **H5**
  (decoder LLMs degrade less than encoders, replicating P001).

**Justification.** Targets **G6** (adversarial robustness rarely evaluated). Only P001 runs
adversarial testing; P011 defers it to future work; P009 covers synthetic perturbations only.
Clean-test-only reporting (P002, P003, P010, P012) "overstates real-world reliability"
(`research_gap.md` G6). G6 is the corpus's second-ranked gap and complements RQ1: RQ1 measures
robustness to *distribution* shift, RQ3 to *adversarial* shift.

---

## RQ4 — A prompt- and protocol-controlled encoder-vs-decoder comparison

**Question.** Under one fixed, fully-disclosed training and evaluation protocol, do open
decoder-only LLMs (Mistral-7B-Instruct, Llama-3.1-8B-Instruct via 4-bit QLoRA) match or beat
fine-tuned encoders on WELFake macro-F1, and does the encoder↔decoder ordering flip between the
in-distribution (RQ1), leakage (RQ2), and adversarial (RQ3) regimes?

**Operationalization.**
- Same splits, seeds, metric suite, and full-document exposure for every family (protocol §5–§9);
  LLMs evaluated as QLoRA fine-tune + zero-shot + 5-shot, with zero-/few-shot results reported
  separately as **contamination-suspect** and gated by the per-LLM contamination probe
  (protocol §4.7, §5). Scoped explicitly as an n=2 case study, not a general "decoder LLM" claim.
- Decision rule: paired bootstrap on macro-F1 differences + Cliff's delta + BH-FDR (protocol
  §10). Cross-regime ordering read jointly from RQ1/RQ2/RQ3 tables. Confirmatory family link:
  the pairwise encoder comparisons on WELFake (**H1** — no significant pairwise encoder
  difference after FDR).

**Justification.** Addresses **G9**: the encoder-vs-decoder question is "central but currently
muddied by prompt-sensitivity artifacts and inconsistent protocols" — P001 reports an internal
contradiction, P002 reports an implausible 12.3% zero-shot GPT-4o from label-parsing issues,
and P010 uses generative models for generation, not classification (`research_gap.md` G9). A
single controlled protocol with a contamination probe is the missing clean comparison; it also
answers the practical question of *whether the winner is stable* across the RQ1–RQ3 stress tests
rather than only on clean in-distribution data.

---

## RQ5 — Accuracy-per-compute under a fully disclosed protocol

**Question.** Holding the training protocol fixed and fully disclosed, which model family sits on
the accuracy-vs-compute Pareto frontier on WELFake, and do compact encoders (DistilBERT/ALBERT)
retain ≥ 95% of the best encoder's macro-F1 at < 50% of its parameters?

**Operationalization.**
- All timing on a single fixed accelerator (NVIDIA T4, 16 GB, Kaggle); report parameter count,
  on-disk size, peak VRAM, training wall-clock, inference latency (ms/sample, batch=1, seq=512),
  throughput; plot macro-F1 vs parameters and macro-F1 vs latency (protocol §14, RQ-D).
- Every hyperparameter (optimizer AdamW, wd=0.01; warm-up 6% → cosine decay; LR 2e-5 encoders /
  2e-4 QLoRA; batch 16; max 10 epochs, early stop) is fixed and recorded, remedying the
  reporting gaps below.
- Decision rule: 3 seeds for this secondary endpoint; confirmatory link **H6** (compact encoder
  ≥ 95% of best encoder macro-F1 at < 50% params).

**Justification.** Combines **G12** (efficiency benchmarking is not standardized — P002 reports
USD, P003 time, P010 params/size, P011 param counts, all on different axes) with **G7**
(systemic non-reporting of scheduler, batch size, and per-dataset sizes across P002–P012;
`pattern_analysis.md` §5–§6). Fixing every setting and measuring on one accelerator is the only
way to make accuracy-per-compute comparable, which is precisely what the field currently cannot
do. This is the "resource paper" complement the professor recommends bundling with G1/G6.

---

## Coverage summary

| RQ | Frozen protocol RQ | Primary gap(s) | Hypothesis | Confirmatory? |
|----|--------------------|----------------|------------|:-------------:|
| RQ1 | RQ-A | G1 | H2 | Yes |
| RQ2 | RQ-B | G1, G8 | H3, H4 | Yes |
| RQ3 | RQ-C | G6 | H5 | Yes |
| RQ4 | RQ-F (+ H1 encoder family) | G9 | H1 | Partly (encoder pairwise) |
| RQ5 | RQ-D | G12, G7 | H6 | No (secondary) |

RQ1–RQ3 constitute the primary, confirmatory thesis (cross-dataset generalization, leakage
attribution, adversarial robustness). RQ4–RQ5 are secondary case-study/resource questions,
reported only if the primary endpoints complete within the ≤ 120 GPU-hour budget (protocol §1).
All questions are answered with 5-seed (primary) or 3-seed (secondary) runs, paired bootstrap
tests with 95% CIs, Cliff's delta effect sizes, and Benjamini–Hochberg FDR at q = 0.05.
