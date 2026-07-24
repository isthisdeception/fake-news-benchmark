# Pre-Implementation Peer Review — Fake News Detection Benchmark

Simulated **three anonymous reviewers** for a top-tier NLP venue (ACL/EMNLP/NAACL tier).
Scope of review: the **research design only** (`docs/final_protocol.md` v1.0, `research_questions.md`,
`hypotheses.md`, `experiment_matrix.md`, `evaluation_protocol.md`, `reproducibility_checklist.md`,
`literature/dataset_selection_report.md`). No code exists yet; every fix is actionable **before**
implementation.

Reviewer personas (distinct expertise, deliberately adversarial):
- **R1 — Methodology & statistics.** Experimental fairness, evaluation protocol, inference validity.
- **R2 — Data & leakage.** Dataset choice, leakage controls, construct validity.
- **R3 — Novelty & impact (acts as AC).** Contribution, scope, external validity, positioning.

Likelihood key: **VH** ≥ 85%, **H** 65–85%, **M** 40–65%, **L** 20–40%, **VL** < 20%
(probability that *at least one* reviewer at this venue raises the concern).

---

## Reviewer 1 — Methodology & Statistics

**Summary.** The protocol is unusually rigorous (pre-registration, BH-FDR, dedup, single-accelerator
efficiency). But several fairness and inference choices are internally inconsistent with the
"fair comparison" thesis, and two statistical decisions (equivalence vs non-significance;
attack comparability) can invalidate headline claims. **Score: 4/6 (borderline; fixable).**

### W1. "No difference" (H1) is inferred from non-significance, not equivalence *(Evaluation / Stats)*
- **Why it matters.** H1 claims the top encoders are "statistically indistinguishable" if pairwise
  BH-adjusted p ≥ 0.05. Failure to reject the null is **not** evidence of equivalence, especially
  with only 5 seeds on a saturated dataset (low power). A reviewer will read this as "underpowered,
  so of course nothing is significant." The headline "encoders are equivalent" is then unsupported.
- **Fix before implementation.** Replace/augment with a **TOST equivalence test** (or Bayesian ROPE)
  with a pre-registered equivalence margin (e.g., ±1.0 macro-F1). Report a power analysis for the
  5-seed design. State H1 as an equivalence hypothesis, not an absence of significance.
- **Likelihood: VH.**

### W2. Adversarial attacks are not budget-matched across model families *(Fairness / Threats)*
- **Why it matters.** RQ-C/H5 compares ASR across classical, encoder, and LLM families. TextAttack
  attacks are query/gradient-driven and per-model; without a **fixed query budget and identical
  attack recipe per sample**, a model that is simply queried more (or is white-box vs black-box)
  will show higher ASR for reasons unrelated to robustness. The "LLMs are more robust" claim (H5)
  could be a budget artifact — exactly the kind of confound P001's contradictory RQ1 already shows.
- **Fix.** Pre-register a **fixed per-example query cap**, identical attack search method and
  constraints for every model, and report ASR at matched budgets; consider transfer attacks
  (attack the best encoder, evaluate all) as a second, budget-neutral comparison.
- **Likelihood: H.**

### W3. Single frozen hyperparameter config gives classical models a tuning advantage *(Fairness)*
- **Why it matters.** Classical models get a **validation grid search** (`configs/classical_grids.yaml`),
  while every encoder is locked to one config (LR 2e-5, batch 16). This is an **asymmetric tuning
  budget**: encoders cannot adapt (DeBERTa-v3/ALBERT are known to be LR-sensitive), so an encoder
  losing to another may reflect a bad shared LR, not the model. This undercuts the "fair comparison"
  headline and RQ4/H1.
- **Fix.** Either (a) give every family an **equal, pre-registered small tuning budget** (e.g., LR ∈
  {1e-5, 2e-5, 3e-5} × warmup, selected on val macro-F1), or (b) remove the classical grid and fix
  classical hyperparameters too. Document the equalized budget explicitly.
- **Likelihood: H.**

### W4. Preprocessing is entangled with model family → confounded classical-vs-neural comparison *(Fairness)*
- **Why it matters.** Classical = lowercase + stopword removal + lemmatization + TF-IDF; neural =
  subword only. The protocol admits the entanglement (§6), but any classical-vs-neural difference
  then confounds **model** with **preprocessing**. Family-level claims (and the Friedman/Nemenyi
  ranking) are not clean.
- **Fix.** Add an **ablation cell**: run at least one strong encoder on the classical-cleaned text
  and one classical model on minimally-cleaned text, to bound the preprocessing effect. Explicitly
  scope all family comparisons as "model+preprocessing pipeline", not "model".
- **Likelihood: M.**

### W5. Downstream experiments depend on an arbitrary "best encoder" *(Design dependency)*
- **Why it matters.** RQ-C (adversarial), RQ-D (efficiency), EXP-B1/B2, EXP-G2/G3 all use "the best
  encoder from EXP-1". If H1 is true (encoders tied), "best" is **within noise / arbitrary**, so the
  entire robustness/efficiency/interpretability story rests on a coin-flip choice.
- **Fix.** Pre-register a **deterministic tie-break rule** (e.g., highest mean macro-F1; ties broken
  by lowest ECE, then fewest params), and run RQ-C/D on the **top-2** encoders to show the
  conclusions are stable to the choice.
- **Likelihood: M.**

### W6. Seed variance and bootstrap resampling variance are conflated *(Stats)*
- **Why it matters.** The paired bootstrap resamples the test set (10k), but only 5 seeds capture
  training stochasticity. Reporting a bootstrap CI on a difference while averaging over 5 seeds mixes
  two variance sources; a naive combination can understate uncertainty.
- **Fix.** Pre-register the exact estimand: e.g., **per-seed bootstrap then aggregate**, or a
  hierarchical/mixed-effects model with seed as a random effect. State how CIs combine the two levels.
- **Likelihood: M.**

### W7. Cross-domain ΔF1 confounds generalization loss with threshold miscalibration *(Evaluation / Threats)*
- **Why it matters.** Macro-F1 at a fixed 0.5 threshold penalizes a model that transfers well in
  ranking but needs a shifted operating point on an imbalanced target. RQ1/H2's "≥10-point drop" may
  partly measure threshold drift, not representational failure.
- **Fix.** Report ΔF1 **both** at the fixed threshold and at a **validation-tuned threshold**
  (tuned on source-domain val), plus threshold-free PR-AUC ΔAUC, to separate calibration from
  generalization.
- **Likelihood: M.**

### W8. Exploratory analyses are entirely uncorrected *(Stats)*
- **Why it matters.** Only the narrow confirmatory family gets BH-FDR; E1/E2 (Friedman/Nemenyi) and
  the many transfer cells are uncorrected, inviting false positives that readers over-interpret.
- **Fix.** Apply within-family FDR to exploratory analyses too (clearly separated from confirmatory),
  and label every exploratory p as such.
- **Likelihood: L.**

---

## Reviewer 2 — Data & Leakage

**Summary.** The leakage-audit framing is the paper's best idea, but it is fragile: the central
shortcut control depends on source metadata that may not exist, the transfer set is tiny and
partially self-overlapping, and "fake" is not the same construct across datasets. **Score: 3/6
(major revision).**

### W9. The headline shortcut control (H4/RQ-B) is conditional on unrecoverable metadata *(Leakage)*
- **Why it matters.** WELFake is a **merged** corpus (Kaggle/McIntire/Reuters/BuzzFeed); per-article
  source may not be recoverable. Both the source-disjoint split (EXP-B2) and the source/metadata-only
  baseline (EXP-B3) — the study's novel contribution — could collapse to a length/punctuation
  surrogate or to "not applicable". The most publishable claim then evaporates.
- **Fix.** **Before implementation**, verify source recoverability for WELFake; if weak, pre-register
  a concrete fallback: reconstruct source via the four constituent datasets, or run the shortcut
  baseline on a dataset where source is known (ISOT: Reuters-vs-fabricated), and state the surface-
  feature baseline as the guaranteed-available version.
- **Likelihood: VH.**

### W10. WELFake and ISOT share Reuters real news → transfer between them is confounded *(Leakage / Fairness)*
- **Why it matters.** WELFake merges Reuters; ISOT's "real" class is Reuters. WELFake↔ISOT transfer
  is therefore partly **same-source**, which either inflates transfer accuracy or causes the 0.90
  cross-dedup to delete large, non-random chunks of the test set, distorting class balance and ΔF1.
- **Fix.** Quantify and **report source overlap** between DS1 and DS2 up front; after cross-dedup,
  report the residual size and class balance; consider excluding the WELFake↔ISOT pair from
  confirmatory transfer or treating it as a labeled "same-source" control.
- **Likelihood: H.**

### W11. Lexical (TF-IDF cosine) dedup misses semantic/paraphrase duplicates *(Leakage)*
- **Why it matters.** Thresholds 0.95/0.90 on TF-IDF catch near-verbatim copies but not paraphrases
  or template-rewrites — precisely the leakage that survives across merged sources and across
  datasets. Residual semantic leakage would inflate both in-distribution and transfer numbers,
  weakening RQ-A and RQ-B.
- **Fix.** Add an **embedding-based near-dup pass** (e.g., Sentence-BERT cosine) as a second filter
  and report the incremental removals; keep TF-IDF as the primary, embeddings as the audit.
- **Likelihood: H.**

### W12. Generalization/leakage conclusions rest on very few datasets *(Dataset / External validity)*
- **Why it matters.** Article-type transfer uses only {WELFake, ISOT, FakeNewsNet} — one saturated
  (ISOT ≈ 100%), one link-rotted (FakeNewsNet). RQ-B leakage is measured on **WELFake alone (n=1)**.
  "Cross-dataset generalization degrades" and "benchmarks are leaky" from 2–3 flawed datasets is a
  thin empirical base for a top venue.
- **Fix.** Add ≥1 additional article dataset for transfer (e.g., a NELA-GT slice, or a recent
  post-2022 set), and replicate the leakage audit on ≥1 second dataset so RQ-B is not n=1. If
  infeasible, explicitly scope claims to "these datasets" and soften the generality.
- **Likelihood: H.**

### W13. FakeNewsNet bespoke snapshot is not externally reproducible *(Dataset / Reproducibility)*
- **Why it matters.** Because FakeNewsNet ships as IDs/URLs with link rot, your cached snapshot is
  unique to you. Others cannot reconstruct the exact test set, undermining the benchmark's
  reproducibility claim precisely where it is emphasized.
- **Fix.** Release the **snapshot content hash + the exact hydrated IDs and retrieval date**, and
  report what fraction was unrecoverable; consider replacing FakeNewsNet with a statically-hosted
  alternative if reproducibility is a headline claim.
- **Likelihood: M.**

### W14. "Fake" is a heterogeneous construct across datasets (construct validity) *(Threats)*
- **Why it matters.** WELFake fabricated/political, ISOT fabricated-vs-Reuters, FakeNewsNet
  fact-checked celebrity/political, COVID health claims, LIAR 6-way statements. Cross-dataset transfer
  measures a mix of **veracity, style, source, and topic** shift, not a single construct. ΔF1 is
  hard to attribute. Binarizing LIAR (half-true→real) is itself contestable.
- **Fix.** Add a **construct/label-definition table** per dataset; frame RQ-A as "transfer under
  combined construct+distribution shift" and, where possible, decompose (e.g., topic-controlled
  subsets). Justify the LIAR binarization with a sensitivity analysis (alternative cut).
- **Likelihood: M.**

### W15. Contamination probe (guided prefix-completion) is a weak memorization test *(Leakage / LLM)*
- **Why it matters.** A single prefix-completion probe can miss contamination (false "clean"),
  letting inflated zero-shot LLM numbers through despite the safeguard.
- **Fix.** Triangulate with ≥2 probes (e.g., membership-inference / perplexity gap + canary/quiz
  format) and pre-register the flag threshold; treat zero-shot as suspect regardless (already partly
  done).
- **Likelihood: L.**

---

## Reviewer 3 — Novelty & Impact (Area Chair)

**Summary.** Methodologically careful, but as a *contribution* this reads as an audit/resource paper
whose confirmatory findings are largely expected, on English pre-LLM-era data, with the single most
novel gap (AI-generated news) explicitly out of scope. It is a strong workshop/Findings paper; for
the main track the novelty case must be sharpened. **Score: 3/6 (lean reject unless reframed).**

### W16. Contributions are individually well-trodden; findings risk being "unsurprising" *(Novelty)*
- **Why it matters.** The team's own gap analysis scores G1 novelty 6/10 and G6 7/10, and notes "the
  problem is known." Confirming that transfer degrades (H2), dedup lowers accuracy (H3), and compact
  models are competitive (H6) are **expected outcomes**. Top venues discount confirmatory audits that
  tell the community what it already suspects.
- **Fix.** Reframe around the **quantitative decomposition** as the novelty: not "does it leak" but
  "how much of reported SOTA is attributable to leakage vs shortcut vs signal", delivered as a
  reusable, leakage-audited benchmark others can run. Lead with the source/metadata-only decomposition
  (the one genuinely novel control) and a public benchmark artifact.
- **Likelihood: VH.**

### W17. The most novel gap (G2, AI-generated news, novelty 8/10) is excluded *(Novelty / Scope)*
- **Why it matters.** The team's own analysis ranks AI-generated-news detection as the highest-novelty
  gap, and §17 rules it out. Reviewers will ask why the timely, high-impact direction was dropped in
  favor of a known problem, especially at an LLM-era venue.
- **Fix.** Either (a) add a **scoped AI-generated probe** (e.g., a small topic-matched machine-vs-human
  set) as a forward-looking section, or (b) preempt the question with an explicit, compelling scope
  justification (compute/data honesty) in the intro and limitations.
- **Likelihood: H.**

### W18. English-only, text-only, pre-LLM-era data limits external validity/impact *(Threats / Impact)*
- **Why it matters.** Conclusions may not transfer to multilingual, multimodal, or current
  LLM-generated misinformation — the deployment reality. Impact-oriented reviewers weight this heavily.
- **Fix.** Add a crisp **scope-and-limitations** paragraph tying the choice to reproducibility/compute,
  and (optionally) one multilingual or post-2022 stress cell to gesture at generality.
- **Likelihood: M.**

### W19. The "FROZEN/immutable protocol" framing can read as inflexible or defensive *(Positioning)*
- **Why it matters.** Pre-registration is a virtue, but "immutable, no post-hoc tuning" risks
  presenting **known-suboptimal** configurations as fair, and reviewers care about results, not
  governance ceremony. If a fixed LR clearly cripples a model, "we froze it" is not a satisfying reply.
- **Fix.** Keep pre-registration, but present it as **methodology**, not as an authority argument;
  pair it with the equalized tuning budget (W3) so "fixed" never means "handicapped". Move governance
  language to an appendix.
- **Likelihood: M.**

### W20. No positioning against prior published WELFake numbers *(Positioning)*
- **Why it matters.** By (correctly) not borrowing literature numbers, the paper forgoes a direct
  "we reproduce/refute P002/P003/P012" narrative; without explicit positioning, reviewers may ask
  "what do we learn relative to existing WELFake results?"
- **Fix.** Add a **reproduction contrast**: run prior configs where possible and show your
  leakage-audited numbers beside the reported ~95–98%, quantifying the inflation — a concrete,
  citable result.
- **Likelihood: M.**

---

## Consolidated Weakness Register (every weakness, all reviewers)

| ID | Dimension | Reviewer | Weakness (one line) | Fix (pre-implementation) | Likelihood |
|----|-----------|:--------:|---------------------|--------------------------|:---------:|
| W1 | Evaluation/Stats | R1 | H1 infers equivalence from non-significance (underpowered) | TOST/ROPE equivalence test + power analysis; restate H1 | VH |
| W2 | Fairness/Threats | R1 | Adversarial ASR not budget-matched across families | Fixed per-example query cap + identical recipe; add transfer attack | H |
| W3 | Fairness | R1 | Classical get grid search, encoders locked → unequal tuning | Equal pre-registered tuning budget for all families | H |
| W4 | Fairness | R1 | Preprocessing entangled with family → confounded comparison | Cross-preprocessing ablation; scope claims to model+pipeline | M |
| W5 | Design dependency | R1 | RQ-C/D built on arbitrary "best encoder" (tie under H1) | Deterministic tie-break + run top-2 encoders | M |
| W6 | Stats | R1 | Seed variance vs bootstrap variance conflated | Pre-register estimand; per-seed bootstrap or mixed-effects | M |
| W7 | Evaluation/Threats | R1 | ΔF1 confounds generalization with threshold drift | Report fixed + val-tuned threshold + ΔPR-AUC | M |
| W8 | Stats | R1 | Exploratory analyses uncorrected for multiplicity | Within-family FDR for exploratory; label clearly | L |
| W9 | Leakage | R2 | Novel shortcut control depends on maybe-unrecoverable source | Verify source now; pre-register fallback (ISOT/surface) | VH |
| W10 | Leakage/Fairness | R2 | WELFake↔ISOT share Reuters → confounded transfer | Report source overlap; residual size/balance; consider excluding pair | H |
| W11 | Leakage | R2 | TF-IDF dedup misses semantic/paraphrase duplicates | Add Sentence-BERT embedding dedup audit pass | H |
| W12 | Dataset/External | R2 | Transfer on 3 flawed datasets; RQ-B leakage n=1 | Add article dataset(s); replicate leakage on ≥2 datasets | H |
| W13 | Dataset/Repro | R2 | FakeNewsNet snapshot not externally reproducible | Release hashes + hydrated IDs + date; report recovery rate | M |
| W14 | Threats (construct) | R2 | "Fake" heterogeneous across datasets; LIAR binarization | Construct table; reframe transfer; LIAR-cut sensitivity | M |
| W15 | Leakage/LLM | R2 | Single contamination probe is weak | Triangulate ≥2 memorization tests; pre-register threshold | L |
| W16 | Novelty | R3 | Contributions well-trodden; confirmatory findings expected | Lead with quantitative leakage/shortcut decomposition + benchmark artifact | VH |
| W17 | Novelty/Scope | R3 | Highest-novelty gap (AI-generated, G2) excluded | Add scoped AI-gen probe or strong scope justification | H |
| W18 | Threats/Impact | R3 | English-only, text-only, pre-LLM-era limits validity | Scope/limitations paragraph; optional multilingual/post-2022 cell | M |
| W19 | Positioning | R3 | "Immutable protocol" reads inflexible/defensive | Present pre-registration as method; pair with equal tuning budget | M |
| W20 | Positioning | R3 | No contrast vs prior published WELFake numbers | Reproduce prior configs; quantify inflation beside ~95–98% | M |

## Priorities before writing any code

1. **Verify WELFake source recoverability now (W9)** — the novel RQ-B contribution depends on it; if
   weak, the whole leakage-audit story needs a pre-registered fallback.
2. **Fix the two comparability confounds (W2 adversarial budget, W3/W4 tuning + preprocessing)** — these
   directly threaten the "fair comparison" thesis that underpins RQ4/H1/H5.
3. **Convert H1 to an equivalence test (W1)** and de-arbitrary the "best encoder" (W5) — otherwise the
   in-distribution and downstream conclusions are statistically unsupported.
4. **Strengthen leakage detection (W10 same-source overlap, W11 semantic dedup)** and **broaden the
   empirical base (W12)** — so the leakage/generalization claims survive scrutiny.
5. **Sharpen the novelty framing (W16/W17)** — reposition as a quantitative, reusable leakage-audited
   benchmark; justify or partially relax the AI-generated-news exclusion.

**Meta-assessment.** Two reviewers (R1, R2) recommend *major revision but fixable*; the AC (R3)
leans reject unless the contribution is reframed. Every issue above is addressable **pre-implementation**
by amending `configs/` and the protocol (with the §0 version bump), which is far cheaper now than after
120 GPU-hours are spent.
