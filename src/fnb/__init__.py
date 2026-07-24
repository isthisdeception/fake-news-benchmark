"""fnb — Fake News Benchmark.

A reproducible, cross-dataset fake-news detection benchmark spanning classical
TF-IDF models, transformer encoders, and 4-bit QLoRA-fine-tuned LLMs. See
``MASTER_IMPLEMENTATION_GUIDE.md`` for the project constitution and ``README.md``
for the Kaggle run flow.

Sub-packages:
    config       config loading + schema validation + typed constants
    utils        seeding, logging, hashing, run registry, IO helpers
    data         acquisition, binarization, preprocessing, dedup, splits, stats
    models       classical, encoder, source-only, QLoRA LLM
    training     torch datasets, custom training loop, scheduler, early stopping
    evaluation   metric suite + statistical stack
    adversarial  TextAttack recipes under label-preserving constraints
    analysis     calibration, error analysis, interpretability, figures
    experiments  end-to-end experiment drivers (EXP-*)
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
