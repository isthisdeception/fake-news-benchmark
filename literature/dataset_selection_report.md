# Dataset Selection Report — Fake News Detection Benchmark

Evidence base: the reviewed literature database (notes P001–P012). "Papers using it"
counts **empirical** use (training/evaluation); survey cataloging (P006, P007, P008)
is noted separately. All sample counts, languages, and licensing are taken from the
extracted notes; anything absent is marked "Not Reported".

---

## Summary Comparison

| Dataset | Empirical users | Survey-cataloged | Task | Samples | Classes | Language |
|---------|:---------------:|:----------------:|------|---------|:-------:|----------|
| WELFake | P002, P003, P012 (3) | — | Article veracity | 72,134 | 2 | English |
| FakeNewsNet (PolitiFact+GossipCop) | P010, P011 (2) | P006, P007 | Article veracity | ~23,196 (P010 subset); PolitiFact 1,056 / GossipCop 22,140 (P007) | 2 | English |
| ISOT | P010, P011 (2) | P006 | Article veracity | 44,898 | 2 | English |
| LIAR | P010, P011 (2) | P006 | Short-statement veracity | ~12,800 (3,500 in P010 subset) | 6 (often binarized) | English |
| NELA-GT-2022 | P001 (1) | P006 (NELA-GT-2018) | Article veracity (distant) | 1,778,361 (10k subset used) | Multi-label | English |
| COVID-19 Fake News | P010 (1) | P006 (CoAID related) | Claim/article veracity | 3,118 | 2 | English |
| TuringBench | P010 (1) | — | AI-generated text detection | 70,409 | 2 (human/machine) | English |
| Twitter15 / Twitter16 | P004 (1) | — | Rumor/propagation veracity | Twitter15 742 graphs | 4 (binary subset used) | Not Reported (English-oriented) |
| Twitter (MediaEval) | P005 (1) | P007 | Multimodal (text+image) | train 8,617 / test 2,059 | 2 | Multilingual |
| Weibo | P005 (1) | P007 | Multimodal (text+image) | ~9,528 | 2 | Chinese |
| Fakeddit | P005 (1) | P006, P007 | Multimodal (text+image) | 1,063,106 (30k/10k subset used) | 2 / 3 / 6 | English |
| P001 own GPT-4-annotated set | P001 (1) | — | Article veracity | 10,000 labeled | 2 | Not Reported |
| Amazon Reviews / Yelp | P009 (1) | — | Fake-review anomaly (adjacent) | 570M / 1.2M subset | 2 | English |

---

## Per-Dataset Analysis

### WELFake
- **Papers using it:** P002, P003, P012 (3 empirical — the most-used primary dataset in the corpus).
- **Task type:** Binary article veracity (title + full text).
- **Samples:** 72,134 (35,028 real, 37,106 fake) — P002/P003/P012 agree.
- **Classes:** 2. **Language:** English (P003 explicit).
- **Advantages:** Large, class-balanced, includes both title and body (enables P003's granularity axis); merges four sources (Kaggle, McIntire, Reuters, BuzzFeed Political) for topical diversity; used by three corpus papers → **direct fair comparison**.
- **Limitations:** Merged-source construction risks **source/style shortcuts and near-duplicate leakage** (Reviewer flag for P002, P012, which did no dedup); saturates easily (P002/P012 ≈ 95–98%); label noise acknowledged (P012).
- **Licensing/availability:** Attribution-NonCommercial 4.0 International (P002); publicly available on Kaggle and Zenodo (P003 cites Zenodo 4561253).
- **Transformer suitability:** **High** — full-text, large, balanced, public; runs on a single GPU. Its leakage susceptibility is a liability for naive use but an opportunity for a leakage-audited study.

### FakeNewsNet (PolitiFact + GossipCop)
- **Papers using it:** P010, P011 (empirical); cataloged by P006, P007; referenced by P012.
- **Task type:** Binary article veracity (with social context in full version).
- **Samples:** Inconsistent across corpus — P010 subset 23,196 (17,441 real / 5,755 fake); P007 lists PolitiFact 1,056 and GossipCop 22,140; P006 lists 422. 
- **Classes:** 2. **Language:** English.
- **Advantages:** Widely used → comparability; contains title, body, and social/propagation signals; two domains (political vs celebrity).
- **Limitations:** **Reproducibility risk** — distributed as IDs/URLs requiring crawling; link rot makes content partly unrecoverable (P004 explicitly warns about unstable URLs/IDs); reported sizes conflict across papers.
- **Licensing/availability:** Public via crawler/IDs (content not directly redistributable).
- **Transformer suitability:** **Moderate** — good text if fully hydrated, but link-rot undermines exact reproduction.

### ISOT
- **Papers using it:** P010, P011 (empirical); cataloged by P006.
- **Task type:** Binary article veracity. **Samples:** 44,898 (21,417 real / 23,481 fake, P010). **Classes:** 2. **Language:** English.
- **Advantages:** Large, balanced, clean full-text, easily downloadable.
- **Limitations:** **Saturated** — P011 reports ≈100% and P006 >99%; near-perfect scores mean it no longer discriminates between methods; strong source-style artifacts (Reuters-real vs fabricated).
- **Licensing/availability:** Publicly available (University of Victoria ISOT lab).
- **Transformer suitability:** **Low as a discriminative benchmark** (too easy); **useful only as an easy anchor / saturation reference**.

### LIAR
- **Papers using it:** P010, P011 (empirical); cataloged by P006.
- **Task type:** Short-statement veracity (fact-checked political claims). **Samples:** ~12,800 (P010 uses a 3,500 subset). **Classes:** 6 (pants-fire … true), frequently binarized. **Language:** English.
- **Advantages:** Genuinely **hard** (P011 ≈ 60%) → strong discrimination; fine-grained labels; small and cheap to run.
- **Limitations:** Very short text (statements, not articles) → transformers underperform and it is **not comparable to article-length datasets** without care; no article body for granularity studies.
- **Licensing/availability:** Publicly available.
- **Transformer suitability:** **Moderate** — excellent as a difficulty stress-test, poor as a stand-alone article benchmark.

### NELA-GT-2022
- **Papers using it:** P001 (empirical, 10k subset); NELA-GT-2018 cataloged by P006.
- **Task type:** Article veracity via distant/source-level supervision. **Samples:** 1,778,361 articles / 361 outlets (10,000 subset used). **Classes:** Multi-label (veracity from Media Bias/Fact Check). **Language:** English.
- **Advantages:** Very large; real news outlets; temporal coverage.
- **Limitations:** **Distant labels** (source-level, not article-level) introduce source bias (P001); requires subsetting; label reliability lower than human/AI-verified sets.
- **Licensing/availability:** Publicly available (Dataverse).
- **Transformer suitability:** **Moderate** — scale is attractive but noisy distant labels limit clean benchmarking.

### COVID-19 Fake News
- **Papers using it:** P010 (empirical). **Task:** Binary claim/article veracity. **Samples:** 3,118 (2,061 real / 1,057 fake). **Classes:** 2. **Language:** English.
- **Advantages:** Distinct health domain → good for domain-shift probing.
- **Limitations:** Small; topically narrow; imbalanced.
- **Licensing/availability:** Publicly available.
- **Transformer suitability:** **Moderate** — best as a domain-shift/transfer probe, not a primary set.

### TuringBench
- **Papers using it:** P010 (empirical, as the AI-generated class). **Task:** AI-generated vs human text. **Samples:** 70,409 AI-generated. **Classes:** 2. **Language:** English.
- **Advantages:** Enables the timely AI-generated-news dimension (gap G2).
- **Limitations:** In P010 the AI class was **near-trivially separable (~100%)**, inflating metrics; single-generator style.
- **Licensing/availability:** Publicly available.
- **Transformer suitability:** **Moderate** — valuable for AI-text detection but requires a harder, topic-matched construction to be non-trivial.

### Twitter15 / Twitter16
- **Papers using it:** P004 (empirical). **Task:** Rumor/propagation veracity on social graphs. **Samples:** Twitter15 = 742 graphs (per-class counts Not Reported). **Classes:** 4 (fake/true/non-rumor/unverified; binary subset used). **Language:** Not Reported (English-oriented; BERTweet used).
- **Advantages:** Includes **propagation trees** (social-context modeling).
- **Limitations:** Small; requires graph/propagation data; binary subset discards two classes; language unstated.
- **Licensing/availability:** Publicly available (Ma et al. 2017).
- **Transformer suitability:** **Low for text-only transformer benchmarking** (its value is graph/propagation, not text).

### Twitter (MediaEval), Weibo, Fakeddit
- **Papers using them:** P005 (empirical, multimodal); cataloged by P006/P007.
- **Task type:** **Multimodal** (text + image) veracity. **Samples:** Twitter train 8,617/test 2,059; Weibo ~9,528; Fakeddit 1,063,106 (30k/10k subset used). **Classes:** 2 (Fakeddit also 3/6). **Language:** Twitter multilingual, **Weibo Chinese**, Fakeddit English.
- **Advantages:** Enable multimodal research; Fakeddit is large with multi-granularity labels.
- **Limitations:** Require images (out of our text-only, compute-limited scope per frozen protocol §17); Weibo is Chinese (language shift); image pipelines exceed Kaggle free tier.
- **Licensing/availability:** Publicly available (with image download).
- **Transformer suitability:** **Low for our text-only study**; high for multimodal work (out of scope).

### P001 Own GPT-4-Annotated Dataset
- **Papers using it:** P001. **Task:** Binary article veracity (GPT-4 + human labels). **Samples:** 10,000 labeled. **Classes:** 2. **Language:** Not Reported.
- **Advantages:** High-quality AI+human labels (IRR ≈ 72%).
- **Limitations:** **Not openly available** (only "upon request from first author") → fails reproducibility.
- **Licensing/availability:** Restricted (author request).
- **Transformer suitability:** **Low** for us — cannot be obtained reliably.

### Amazon Reviews / Yelp (P009)
- **Papers using it:** P009. **Task:** Fake-**review** anomaly detection (adjacent domain, not news). Not suitable as a fake-**news** benchmark.

---

## Evidence-Based Recommendation

**Primary recommendation: WELFake as the core in-distribution + leakage-audit dataset, embedded in a small cross-dataset transfer set {WELFake → ISOT, FakeNewsNet, COVID-19}, with LIAR as a separate short-statement difficulty control.** This matches `docs/final_protocol.md` §4 and is justified criterion-by-criterion below — not by popularity.

| Criterion | Assessment | Winner |
|-----------|------------|--------|
| **Fair comparison with prior work** | WELFake is the only dataset used by ≥3 corpus papers (P002, P003, P012) spanning classical ML, ALBERT/GRU, CNN, BERT, and GPT-4o — the broadest head-to-head baseline set available. | WELFake |
| **Dataset quality** | Large (72,134), balanced, full title+body, topically diverse. LIAR is cleaner-labeled but too short; ISOT is clean but saturated; FakeNewsNet suffers link rot. | WELFake (with caveats) |
| **Reproducibility** | WELFake ships as static files on Kaggle/Zenodo (fixed content) — unlike FakeNewsNet (crawl/link-rot, P004) or P001's request-only set. | WELFake |
| **Availability** | Public, CC BY-NC 4.0 (P002); non-commercial research use is permitted. | WELFake |
| **Computational feasibility** | Text-only, fits a single Kaggle T4; no image/graph pipeline (unlike P005's multimodal sets or P004's propagation graphs). | WELFake |
| **Publication potential** | WELFake's known leakage susceptibility (P002/P012 did no dedup) is turned into a **novel contribution**: a leakage-audited re-evaluation (protocol RQ-B) — higher novelty than another saturated-benchmark result. | WELFake |

### Why not the alternatives (evidence)
- **ISOT** — saturated (P011 ≈100%, P006 >99%); cannot discriminate modern models. Keep only as an easy anchor.
- **FakeNewsNet** — link rot / ID-based distribution harms reproducibility (P004 warning); conflicting sizes across P006/P007/P010. Keep only as a transfer target from a frozen snapshot.
- **LIAR** — genuinely hard (good), but short statements are not comparable to article-length data and lack a body for granularity analysis. Keep as a separate difficulty control.
- **NELA-GT-2022** — distant/source-level labels are noisy (P001); scale is attractive but label quality is inferior for a clean benchmark.
- **Multimodal sets (Twitter/Weibo/Fakeddit) & Twitter15/16** — require images/graphs and (Weibo) Chinese; out of our text-only, single-GPU scope (protocol §17).
- **P001 own set / P009 reviews** — restricted availability / wrong domain.

### Critical caveat carried into the study
WELFake's greatest weakness — source-style and near-duplicate leakage that inflated P002/P012's ~95–98% — is precisely what makes it valuable *for us*: the protocol mandates (a) within-dataset dedup (cosine ≥ 0.95), (b) a source/metadata-only shortcut baseline, and (c) before/after-dedup reporting (final_protocol §4.5, §11). This converts a common corpus flaw into a measurable, publishable result rather than repeating it.

**Bottom line:** WELFake is recommended not because it is popular, but because it uniquely maximizes fair comparability (3 corpus papers), static reproducibility, public availability, and single-GPU feasibility — while its leakage risk is convertible into a novel contribution. ISOT/FakeNewsNet/COVID-19 serve as transfer targets and LIAR as a difficulty control, exactly as fixed in the frozen protocol.
