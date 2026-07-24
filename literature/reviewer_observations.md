# Reviewer #2 — Critical Observations

Basis: the extracted notes for P001–P012 (`literature/notes/`) and the derived
`master_literature_database.csv` / `pattern_analysis.md`. Every criticism is tied to
specific paper IDs and to evidence recorded during extraction. Where a paper is a
review/survey (P006, P007, P008) it is judged as a survey, not an empirical study.

Tone: unsparing but evidence-bound. I do not speculate beyond what the notes support;
suspected-but-unconfirmed issues are labeled as such.

---

## 1. Common Mistakes (internal inconsistency, sloppy reporting)

- **P001** — Self-contradiction: the RQ1 discussion states "LLMs outperform BERT-like models," directly contradicting the abstract and all result tables (which show BERT-like models winning). Dataset size is reported three different ways (30k curated / 40k / 10k labeled). Classification threshold is stated as both 0.5 and 0.8.
- **P009** — Headline AUC "up to 0.945" (few-shot ">0.95") in the abstract vs. table maxima of ~0.933–0.935. Hardware is inconsistent (RTX A6000 48 GB in text vs. RTX 4050 32 GB in Table 3); PyTorch version 2.2 vs. 2.1. Equation cross-references mislabeled.
- **P012** — Accuracy reported as both 96% (validation) and 95.3% (test) without disambiguation in the narrative. The BERT description ("124-layer, 1024-hidden, 16-head, 340M parameter, bert-base-uncased") is factually wrong — it conflates BERT-base and BERT-large and invents a 124-layer model.
- **P003** — Table 3 GRU title-only row is internally impossible (Precision/Recall = 0.80 but F1 = 0.90). Feature-importance term lists differ between sections.
- **P006** — Screening counts do not reconcile (134 vs. 125 identified; 102 vs. 93 included). Table 3 lists r/fakeddit and M4 with CREDBANK's 5-value credibility scale, which is wrong for those datasets.
- **P008** — Screening arithmetic does not add up (1500 identified, "800 excluded" as duplicates yet "900 unique").
- **P010** — Abstract rounds dataset counts (43k/31k/80k vs. exact 43,372/31,340/80,409); reference [30] mislabels TuringBench as a dialogue benchmark.
- **P002** — Nonstandard, confusion-inducing label convention (0 = fake, 1 = not fake).

Verdict: Inconsistent numbers between abstract, text, and tables are **endemic** (P001, P003, P006, P008, P009, P012). At minimum this signals inadequate proofreading; at worst it undermines the headline claims.

---

## 2. Weak Evaluation

- **Single-dataset evaluation** — P002, P003, P012 evaluate *only* on WELFake; P011's per-dataset numbers are within-dataset only. No transfer/generalization is demonstrated.
- **Trivially separable classes inflate metrics** — P010's "AI-generated" class comes largely from a single source (TuringBench) + self-generated titles and is classified at ~100%, plausibly inflating the reported 96.66% F1.
- **Title/headline-only classification** — P010 classifies news *titles* only; robustness on full articles is untested.
- **Downsampling / truncation** — P002 reduces WELFake from ~72k to 5,000 articles truncated to 2,560 characters; P001 samples 10k. Near-perfect fine-tuned results (P002: 98.6%) on such easy subsets are suspicious.
- **Saturated benchmarks** — P011 (ISOT ≈ 100%) and P006's own analysis (ISOT > 99%) show evaluation on benchmarks that no longer discriminate between methods.
- **Unstable splits** — P011's ablations show some "no_attention" configurations *outperforming* the full baseline and LIAR at ~60%, symptomatic of small/noisy evaluation splits.
- **Clean-test-only** — Only P001 evaluates adversarial/perturbation robustness. P002, P003, P010, P011, P012 report clean-test accuracy exclusively, overstating deployment reliability.
- **Zero-shot artifact** — P002's zero-shot GPT-4o at 12.3% is implausibly low and likely a prompt/label-parsing failure rather than a genuine capability measurement.

---

## 3. Poor Methodology

- **P009** — The "multi-modal / sensor-derived metadata" framing is aspirational: no real sensor/IoT data is used (only structured Amazon fields as proxies), and anomalies are largely *synthetic* (token shuffling, synonym substitution). The core claim is thus weakly supported by the actual experiments.
- **P010** — Deduplication/overlap resolution "involved manual inspection," which is neither documented nor reproducible.
- **P005** — No explicit text/image preprocessing pipeline is reported at all; equations/Algorithm 1 contain OCR/typographical artifacts affecting the method's clarity.
- **P008** — The headline contribution (a CLIP + mBERT/XLM-R fusion framework) is purely conceptual — no experiments, datasets, or ablations validate it.
- **P001** — Draws a conclusion (RQ1) that contradicts its own evidence; the annotation pipeline is only partially validated (IRR ≈ 72% is moderate at best).
- **P012** — Presents an erroneous model specification and relies on a comparison table mixing results from *different datasets and models* as if comparable.

---

## 4. Data Leakage (confirmed mitigations vs. unaddressed risks)

Confirmed good practice:
- **P003** — Explicitly fits the TF-IDF vectorizer on the training set only "to prevent leakage," and uses stratified splits with fixed seeds. Best hygiene in the corpus.
- **P009** — States test reviews are disjoint from the autoencoder's training set.
- **P010** — Performs exact and near-duplicate removal (cosine ≥ 0.95 on TF-IDF) before splitting — important given its sources overlap (FakeNewsNet contains PolitiFact/GossipCop).

Unaddressed / suspected leakage risks (flagged, not confirmed):
- **P002, P012** — Use WELFake (a merge of Kaggle/McIntire/Reuters/BuzzFeed) but report **no near-duplicate removal**. WELFake is known to contain source-style regularities (e.g., Reuters real vs. fabricated), so models can exploit **source/style shortcuts** rather than veracity, and duplicated/near-duplicated articles may straddle the train/test split. This must be ruled out before trusting ~95–98% accuracy.
- **P001** — Applies **SMOTE** for class balance but does not specify that oversampling was confined to training folds. If SMOTE was applied *before* the split (or across CV folds), synthetic neighbors of test points leak into training — a classic and serious error. The note cannot confirm correct ordering, so this is a required clarification.
- **P011** — Small, possibly overlapping splits (evidenced by unstable ablations) raise leakage/variance concerns that the paper does not address.

Reviewer note: leakage is the single most under-examined threat here. Only 3 of 9 empirical papers (P003, P009, P010) demonstrate awareness; the WELFake trio (esp. P002, P012) needs an explicit dedup + source-disjoint analysis.

---

## 5. Missing Statistical Tests

Papers with adequate statistical treatment: **P003** (McNemar + Holm correction), **P005** (Kruskal–Wallis), **P011** (paired t-tests, Cohen's d, 95% CI, k-fold). **P004** and **P009** at least report variance (mean ± std / std of AUC) but no significance test.

Missing significance testing entirely:
- **P001** — Reports 5-fold mean ± std but performs **no significance test** between BERT-like models and LLMs, despite the paper's central claim resting on that comparison.
- **P002** — Point estimates only; no significance test, no confidence intervals — yet claims one model "matches" another (GPT-4o vs. GPT-4o-mini both 98.6%).
- **P010** — Single 80/20 split, no significance test across its 11+ models; the ensemble's 96.66% vs. RoBERTa's 96.45% is asserted as an improvement without any test of significance.
- **P012** — Only metric-correlation analysis; the key claim (+1.2% from progressive training) has **no significance test** and could be within run-to-run noise.

Verdict: The strongest claims in P001, P002, P010, and P012 are statistically unsupported. Differences of 0.1–1.2% accuracy without variance or a significance test are not publishable evidence.

---

## 6. Missing Baselines

- **P002** — No classical ML baselines beyond a single CNN (no Logistic Regression / SVM / RF), and no external prior-work method re-run; comparisons are internal to 4 models.
- **P011** — Baselines limited to four transformers and their ablations; no classical ML and no established FND SOTA baseline.
- **P010** — "Baselines" in Table 10 are prior works evaluated on *different datasets* (COVID-19, FA-KES, Kaggle, LIAR, ISOT), i.e., not re-run on the authors' combined D6 — an apples-to-oranges comparison.
- **P012** — Comparison models (TCN-URG, MPFN, FNR-S, SAFE, BERT+LSTM, etc.) are cited from other papers on other datasets, not re-run; only internal baseline is standard BERT.
- **P009** — Omits the strongest relevant baselines (GPT-4-based detectors, adversarially tuned RoBERTa-large) — acknowledged but still a gap.
- **P001** — No GPT-3.5/4 or Claude evaluated *as classifiers*, and non-instruction Llama/Mistral (capable of sequence classification) not tested — acknowledged, but leaves the encoder-vs-decoder claim under-powered.

Verdict: Cross-dataset "borrowed" baselines (P010, P012) are the worst offense — they create the illusion of SOTA comparison without controlled experiments.

---

## 7. Weak Discussion

- **P001** — The discussion actively contradicts the results (RQ1), which is disqualifying until corrected; it does not reconcile why LLMs lose on accuracy but win on robustness.
- **P012** — Related-work and results narration are "garbled/typo-laden" with duplicated/mismatched table captions; discussion does not engage with the 96% vs. 95.3% discrepancy or the erroneous model spec.
- **P002** — No dedicated future-work section; the discussion drifts into speculative "transformative effects on the news industry" rather than analyzing *why* zero-shot failed or whether results generalize.
- **P010** — Does not seriously interrogate the ~4-point accuracy drop from pruning (96.65% → 92.78%) or the near-100% AI class as a confound.
- **P007** — As a survey, provides no quantitative meta-analysis or synthesis; aggregates heterogeneous numbers and contains garbled passages.
- **P006, P008** — Present quantitative claims sourced from other works as if settled, without tracing reliability; P008's discussion promotes an untested framework.

---

## 8. Weak Reproducibility

- **P011** — The clearest irony: it *markets* reproducibility (fixed seeds, deterministic CUDA, pinned versions) yet **omits dataset sizes and the core hyperparameters** (learning rate, batch size, epochs). Reports only the best run by validation accuracy (cherry-picking risk).
- **P012** — Omits learning rate, batch size, and max sequence length; data available only via institutional contact (no public repo).
- **P005** — No preprocessing pipeline and no batch size reported.
- **P004** — Optimizer name and batch size not reported (though embedding/layer hyperparameters are tabulated and code is public).
- **P003** — Optimizer name not explicitly stated and batch size absent (mitigated by detailed grids and fixed seeds).
- **P002** — Fine-tuning performed via the OpenAI API (proprietary job IDs), which is **not reproducible** by third parties and incurs cost; API models can change/deprecate.
- **P001** — Dataset available only "upon request from the first author"; **P008** likewise restricts data to author request.
- **Corpus-wide** — Schedulers are unreported in P002, P003, P004, P005, P009, P010; batch sizes unreported in P003, P004, P005, P011, P012 (see pattern_analysis §5–6). Fair re-implementation is therefore impossible for most of these papers.

---

## Per-Paper Reviewer #2 Rap Sheet

| Paper | Most serious issues | Reviewer #2 disposition |
|-------|---------------------|-------------------------|
| P001 | Self-contradictory conclusion; SMOTE leakage risk unclear; no significance test; preprint (no venue/DOI) | Major revision |
| P002 | Single dataset + heavy downsampling; no dedup (leakage risk); no significance test; API-only fine-tuning; missing baselines | Major revision |
| P003 | Table typo (GRU); single dataset (acknowledged); optimizer/batch unreported | Minor revision (strongest of the empirical set) |
| P004 | Optimizer/batch unreported; only two small Twitter datasets; no significance test | Minor–Major revision |
| P005 | No preprocessing pipeline; batch size unreported; equation/OCR artifacts; text+image only | Major revision |
| P006 | Survey count inconsistencies; wrong rating-scale table; numbers borrowed from primaries | Minor revision (as a survey) |
| P007 | No meta-analysis; garbled text; borrowed heterogeneous numbers | Minor–Major revision (as a survey) |
| P008 | Adjacent domain; untested conceptual framework; screening arithmetic errors; restricted data | Major revision (as a survey) |
| P009 | Aspirational "sensor/multimodal" framing; synthetic anomalies; internal inconsistencies (AUC/hardware/version) | Major revision |
| P010 | Trivial AI class inflates metrics; title-only; borrowed cross-dataset baselines; manual dedup; no significance test | Major revision |
| P011 | Reproducibility claims undercut by missing hyperparameters/sizes; unstable splits; best-run reporting; no classical baselines | Major revision |
| P012 | Erroneous BERT spec; 96% vs 95.3% inconsistency; no dedup (leakage risk); missing hyperparameters; borrowed baselines; no significance test | Major revision |

## Reviewer #2 Bottom Line

Three systemic failures recur across the corpus and would, individually, justify a major-revision decision at any serious venue:

1. **Unquantified uncertainty** — the field routinely claims 0.1–1.2% "improvements" with **no significance tests and no confidence intervals** (P001, P002, P010, P012). Only P003, P005, and P011 test significance.
2. **Leakage left unexamined** — near-duplicate/source-style leakage on merged corpora (WELFake: P002, P012) and SMOTE-timing (P001) are not ruled out; only P003, P009, P010 show hygiene.
3. **Reproducibility theater** — papers assert rigor while omitting the exact settings needed to reproduce them (P011 most egregiously, plus P004, P005, P012 and corpus-wide missing schedulers/batch sizes).

If I were handling these submissions, only **P003** approaches acceptable-with-minor-revision on methodology; the remaining empirical papers require major revisions centered on significance testing, leakage audits, controlled (re-run) baselines, and complete hyperparameter disclosure.
