# Systematic Literature Pattern Analysis

Scope: This analysis is derived exclusively from the extracted literature database
(`literature/notes/P001.md`–`P012.md` and `literature/master_literature_database.csv`).
Every observation is supported by paper IDs. Where the underlying notes recorded
`Not Reported`, that absence is itself reported rather than inferred.

Corpus composition (for context):
- Primary experimental studies: P001, P002, P003, P004, P005, P009, P010, P011, P012
- Review/survey papers: P006, P007, P008 (P008 is adjacent-domain: hate speech)

---

## 1. Most Common Datasets

| Dataset | Papers | Count |
|---------|--------|-------|
| WELFake | P002, P003, P012 | 3 (primary) |
| FakeNewsNet | P010, P011 (primary); cataloged in P006, P007 (surveys) | 2 primary + 2 surveys |
| ISOT | P010, P011 | 2 |
| LIAR | P010, P011 (primary); cataloged in P006 (survey) | 2 primary + 1 survey |
| Twitter (MediaEval / Twitter15/16) | P004 (Twitter15/16), P005 (MediaEval); cataloged in P007 | 2 primary + 1 survey |
| Fakeddit | P005 (primary); cataloged in P006, P007 | 1 primary + 2 surveys |
| Weibo | P005 (primary); cataloged in P007 | 1 primary + 1 survey |
| NELA-GT (2018/2022) | P001 (NELA-GT-2022); cataloged in P006 | 1 primary + 1 survey |
| COVID-19 Fake News | P010 | 1 |
| Amazon Reviews / Yelp | P009 | 1 (fake-review domain) |

Observations:
- **WELFake is the single most common primary-study dataset** (P002, P003, P012), making it the strongest cross-paper comparison anchor.
- **FakeNewsNet, ISOT, and LIAR co-occur** as a benchmark trio in the combined/ablation studies (P010, P011).
- Survey papers P006 and P007 independently confirm that Twitter, Weibo, Fakeddit, FakeNewsNet, PHEME, and LIAR are the field's recurring benchmarks.
- P009 uses review/e-commerce data (Amazon/Yelp) rather than news, so it is dataset-adjacent to the benchmark.

---

## 2. Most Common Transformer Models

| Model | Papers | Count |
|-------|--------|-------|
| BERT (bert-base-uncased and variants) | P001, P002, P004, P005, P010, P011, P012 | 7 |
| RoBERTa | P001, P010, P011 | 3 |
| DistilBERT | P001, P010, P011 | 3 |
| DeBERTa | P010, P011 | 2 |
| ALBERT (albert-base-v2) | P003 | 1 |
| BERTweet | P004 | 1 |
| XLNet | P010 | 1 |
| GPT-4o / GPT-4o-mini | P002 | 1 |
| Llama2-7B / Mistral-7B | P001 | 1 |
| FLAN-T5 | P009 | 1 |
| Swin-T / CLIP (multimodal) | P005 | 1 |
| GPT-2 / GPT-NEO / Distil-GPT-2 (generation) | P010 | 1 |

Observations:
- **BERT is the dominant backbone**, used or fine-tuned in 7 of the 9 primary studies (P001, P002, P004, P005, P010, P011, P012).
- **RoBERTa, DeBERTa, and DistilBERT are the most common "beyond-BERT" encoders**, and P010 and P011 both benchmark the full BERT/RoBERTa/DeBERTa/DistilBERT set.
- Surveys P006 and P007 corroborate that transformer models (BERT/GPT family) are the prevailing architecture class.
- Decoder-only LLMs (P001 Llama2/Mistral; P002 GPT-4o) and instruction-tuned encoder-decoder (P009 FLAN-T5) appear but are less common than encoder-only transformers.

---

## 3. Most Common Preprocessing Pipeline

Per-paper reported steps:
- P002: column removal, empty-row removal, title+text merge, label standardization, length restriction (2560 chars), tokenization
- P003: lowercasing, noise removal (URLs, HTML, mentions, hashtags, punctuation, digits), whitespace normalization, stopword removal (NLTK), lemmatization (WordNet), TF-IDF
- P004: remove URLs, remove punctuation, remove stopwords (Word2Vec only), integer tokenization
- P009: remove HTML/emojis/special chars, lowercasing, punctuation normalization, NLTK tokenization, langdetect filtering, SentencePiece
- P010: cleaning (HTML/numbers/symbols), lowercasing, stopword removal, tokenization, lemmatization, deduplication
- P011: bert-base-uncased tokenizer, text normalization, truncation (128–512 tokens)
- P012: EDA, data cleaning, text cleaning (special chars, lowercasing), stemming/lemmatization, stopword removal, tokenization
- P001: Google RSS curation, length filter (>150 words), GPT annotation, SMOTE class balancing
- P005: **Not Reported** as an explicit text-cleaning pipeline

**Most common pipeline (the modal sequence):**
`lowercasing → noise/URL/HTML/punctuation removal → stopword removal → stemming/lemmatization → tokenization`

Support:
- Lowercasing: P003, P009, P010, P012 (and P002)
- Stopword removal: P003, P004, P010, P012
- Lemmatization/stemming: P003 (WordNet), P010, P012
- URL/HTML/special-character removal: P003, P004, P009, P010, P012
- Tokenization: all primary studies

Observation: The classic **NLP-cleaning stack (lowercase + noise removal + stopword removal + lemmatization + tokenization)** is most fully instantiated in P003, P010, and P012, with P004 and P009 as partial matches. Note that transformer-native pipelines (P011) rely mainly on subword tokenization/truncation, and P005 does not report a cleaning pipeline.

---

## 4. Most Common Optimizer

| Optimizer | Papers |
|-----------|--------|
| AdamW | P001, P010, P012 |
| Adam | P002, P005 |
| Not Reported / grid-search only | P003, P004, P009, P011 |

Observation: **The Adam family (Adam/AdamW) is the de facto standard**, explicitly used in 5 papers (P001, P002, P005, P010, P012). No paper reports a non-Adam optimizer as its primary choice (P012 mentions SGD only as a general alternative). Optimizer is `Not Reported` in P003, P004, P009, and P011.

---

## 5. Most Common Scheduler

| Scheduler | Papers |
|-----------|--------|
| Cosine annealing (incl. warm restarts) | P011, P012 |
| Learning-rate warm-up | P001 (6% steps, RoBERTa), P011 |
| ReduceLROnPlateau (mentioned) | P012 |
| Not Reported | P002, P003, P004, P005, P009, P010 |

Observation: **Schedulers are rarely reported.** The most common named scheduler is **cosine annealing** (P011, P012), with warm-up used in P001 and P011. A majority of papers (P002, P003, P004, P005, P009, P010) report no scheduler at all, which is a reproducibility gap across the corpus.

---

## 6. Most Common Batch Size

| Batch size | Papers |
|------------|--------|
| 16 | P001 (16–32 for BERT), P010 (16 for all NLP models) |
| 32 | P001 (upper bound of 16–32 range) |
| 8 / 16 | P001 (LLM train 8 / eval 16) |
| 6 | P002 |
| Not Reported | P003, P004, P005, P011, P012 |

Observation: **Batch size 16 is the most common reported value** (P001, P010). P002 is an outlier at 6. Batch size is `Not Reported` in five papers (P003, P004, P005, P011, P012) — notably including P011, despite its explicit reproducibility emphasis.

---

## 7. Most Common Evaluation Metrics

| Metric | Papers |
|--------|--------|
| Accuracy | P001, P002, P004, P005, P009, P010, P011, P012 (P003 uses macro P/R/F1) |
| Precision | P001, P002, P003, P004, P005, P009, P010, P011, P012 |
| Recall | P001, P002, P003, P004, P005, P009, P010, P011, P012 |
| F1-score | P001, P002, P003, P004, P005, P009, P010, P011, P012 |
| AUC / AUC-ROC | P003, P009; surveyed in P007, P008 |
| Statistical significance tests | P003 (McNemar + Holm), P005 (Kruskal–Wallis), P011 (paired t-test, Cohen's d, 95% CI) |
| Efficiency (training time / cost / params) | P002 (time + USD cost), P003 (time), P004 (mean±std), P010 (size/params/time), P011 (params) |

Observation: **Accuracy, Precision, Recall, and F1 are near-universal** (all primary studies report the P/R/F1 set). **AUC-ROC** is the most common secondary metric (P003, P009; endorsed by surveys P007, P008). Only P003, P005, and P011 add formal **statistical significance testing** — a differentiator of methodological rigor.

---

## 8. Common Hyperparameter Ranges

Learning rate (transformers/encoders):
- 2e-5 to 5e-5 — P001 (BERT), P002 (2e-5), P010 (3e-5, distillation)
- 1e-5 to 1e-4 — P003 (ALBERT log-uniform)
- 1e-5 to 3e-5 — P001 (LLMs)
- 0.001 — P004 (GETAE), P005 (Adam)
- 1e-4 to 1e-3 — P003 (GRU log-uniform)
- Not Reported (numeric) — P011, P012

Epochs:
- 3 — P002; 3–5 — P003 (ALBERT)
- 10 — P010; 20 — P001 (early stopping); 25 — P012 (10+10+5 progressive)
- 30 (8 for Word2Vec) — P004; 100 (best ~70) — P005

Batch size: 6 (P002); 16 (P010); 16–32 (P001).

Dropout:
- 0.2 — P004; 0.1–0.5 — P003; 0.1–0.2 — P011

Max sequence length:
- 128 — P004; ~200 — P003; 512 — P002, P009; 128–512 — P011

Model/param scale:
- 0.3M / 1.2M / 3.1M compact variants — P011
- ~66M–265M (DistilBERT→RoBERTa+DeBERTa ensemble) — P010
- ~110M BERT-base — P002

Observation: **The modal transformer configuration is learning rate ≈ 2e-5–5e-5, batch size 16, dropout ~0.1–0.2, max length 512, trained for a small number of epochs (3–10)** (P001, P002, P010; ranges from P003, P011). Graph/multimodal models use a higher LR of 0.001 and many more epochs (P004: 30; P005: 100).

---

## 9. Frequently Reported Limitations

| Limitation theme | Papers |
|------------------|--------|
| English-only / monolingual restricts generalization | P003, P010, P011, P012; surveys P006, P007, P008 |
| Single-dataset / cross-domain generalization untested | P003, P005, P011, P012 |
| No / limited multimodal coverage (esp. video, audio) | P001, P005 (text+image only), P011, P012 |
| Interpretability challenges | P003 (SHAP computationally prohibitive), P009 |
| Synthetic / non-real anomalies limit realism | P009 |
| Dataset quality: instability, imbalance, missing metadata | P004, P006 |
| Not real-time / precollected data only | P010; surveyed in P006, P007 |
| Missing hyperparameters / sizes hurt reproducibility | P011, P012 (recorded in their notes' weaknesses) |

Observation: The **two most frequent limitations are (a) English-only scope** (P003, P010, P011, P012 + all three surveys) **and (b) single-dataset evaluation with untested cross-domain generalization** (P003, P005, P011, P012). Lack of multimodality (P001, P005, P011, P012) is the third recurring theme.

---

## 10. Frequently Suggested Future Work

| Future-work theme | Papers |
|-------------------|--------|
| Multilingual / cross-lingual detection | P003, P010, P011, P012; surveys P006, P007, P008 |
| Multimodal (text + image + audio + video) | P004, P005, P010, P011, P012; surveys P006, P007, P008 |
| Explainability / XAI (SHAP, LIME, attention) | P003, P011, P012; surveys P007, P008 |
| Real-time / dynamic detection | P010; surveys P006, P007 |
| Larger / dynamic / continually-updated datasets | P004, P006 |
| Detecting AI-generated / LLM misinformation | P006, P010 |
| Cross-domain adaptation / continual learning | P003, P011 |
| Hybrid / ensemble / MoE architectures | P003, P004, P010, P012 |

Observation: **Multilingual extension and multimodal fusion are the two most frequently suggested future directions** (each spanning 5+ papers including all surveys). **Explainability/XAI** is the third recurring recommendation (P003, P011, P012, P007, P008).

---

## 11. Papers With Strongest Methodology

Ranked by reproducibility, statistical rigor, ablation depth, and completeness of reported settings (as documented in the notes):

1. **P003** — multi-input-granularity benchmark; **McNemar's tests with Holm correction**; fixed seeds, held-out test set, reported training time; grid/randomized search. Most statistically rigorous.
2. **P005** — thorough **ablations** of each module; **Kruskal–Wallis significance testing**; computational-cost analysis (params, train/test time); three datasets; demonstrates plug-in transferability of its MIB module.
3. **P004** — **10-fold stratified cross-validation**, ablation matrix (word × node embedding × recurrent unit), full hyperparameter tables, mean±std reporting, public code.
4. **P011** — systematic **seven-component ablation** across four transformers; strong reproducibility protocol (fixed seeds, deterministic CUDA, multiple runs, pinned versions, paired t-tests, Cohen's d, 95% CI). (Caveat below.)
5. **P010** — broad model comparison (6 ML + 5 transformers + ensemble), **efficiency benchmarking** (pruning/quantization/distillation), public dataset + code, deployed system.

Common strength signals: statistical significance testing (P003, P005, P011), cross-validation/multiple runs (P004, P011), ablation studies (P004, P005, P011), and public code/data (P004, P010).

---

## 12. Papers With Weakest Methodology

Flagged for internal inconsistencies, reproducibility gaps, or evaluation limitations (as recorded in the notes' Weaknesses/Reviewer Notes):

1. **P012** — reporting inconsistencies (**96% validation vs 95.3% test** stated interchangeably), an **erroneous BERT specification** ("124-layer, 1024-hidden, 340M, bert-base-uncased" mixes BERT-base/large), garbled related-work text, missing LR/batch size/max length, and single-dataset evaluation.
2. **P001** — **internal contradiction** (RQ1 discussion says "LLMs outperform BERT-like models," contradicting its own tables/abstract), **dataset-size discrepancies** (30k vs 40k vs 10k), inconsistent thresholds (0.5 vs 0.8), and no peer-reviewed venue/DOI (preprint).
3. **P002** — down-samples WELFake from ~72k to **5,000** truncated articles; **implausibly low zero-shot GPT-4o accuracy (12.3%)** likely reflecting prompt/label-parsing issues; near-identical near-perfect fine-tuned results suggest an easy subset; nonstandard label convention.
4. **P009** — **"multi-modal/sensor" framing is aspirational** (structured metadata proxies, not real sensor data); anomalies are largely **synthetic**; internal inconsistencies in hardware (RTX A6000 vs RTX 4050), PyTorch version, and headline AUC (0.945 abstract vs ≤0.935 tables).
5. **P011** — although methodologically strong in design, it **omits dataset sizes and core training hyperparameters** (learning rate, batch size, epochs) despite its reproducibility claims, and its LIAR results (~60%) and noisy FakeNewsNet ablations suggest small/unstable splits. (Listed here specifically for its reproducibility gap.)

Note on surveys: P006, P007, and P008 are review papers and are not ranked on experimental methodology; their recorded weaknesses are internal count inconsistencies (P006: 134 vs 125, 102 vs 93; P008: 1500/900/800 screening mismatch) and, for P008, an untested conceptual framework in an adjacent domain (hate speech).

---

## Cross-Cutting Summary

- **Convergence points**: WELFake dataset (P002, P003, P012); BERT-family transformers (P001, P002, P004, P005, P010, P011, P012); Adam/AdamW optimizer (P001, P002, P005, P010, P012); Accuracy/Precision/Recall/F1 metric set (all primary studies); LR ≈ 2e-5–5e-5 with batch size 16 (P001, P002, P010).
- **Systemic reporting gaps**: schedulers (unreported in P002, P003, P004, P005, P009, P010), batch sizes (unreported in P003, P004, P005, P011, P012), and per-dataset sizes (P004, P011) — a consistent reproducibility weakness across the corpus.
- **Shared open problems** (limitations → future work): multilingual coverage, multimodality, explainability, and cross-domain generalization recur in the large majority of papers (P003, P004, P005, P006, P007, P008, P010, P011, P012).
