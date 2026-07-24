# Prompt Log — This Session

A verbatim list of every user prompt in this conversation, in order.

---

## Prompt 1 — Research questions

You are a senior NLP researcher.

Read:

- docs/final_protocol.md
- literature/research_gap.md
- literature/pattern_analysis.md

Our dataset has already been selected (WELFake).

Generate publication-quality research questions.

Requirements

- 3–5 research questions
- Each must be answerable through experiments
- Each must directly address the research gap
- Avoid vague or descriptive questions
- Include a justification for every research question

Save as:

docs/research_questions.md

---

## Prompt 2 — Testable hypotheses

Using

- research_questions.md
- final_protocol.md
- literature database

Generate testable hypotheses.

For each hypothesis include:

- Hypothesis statement
- Scientific rationale
- Related literature
- Experiment(s) that will test it

Save:

docs/hypotheses.md

---

## Prompt 3 — Experiment matrix

Using the protocol,

design every experiment required.

For each experiment specify:

- Experiment ID
- Objective
- Dataset version
- Model
- Hyperparameters
- Metrics
- Expected outputs

Generate:

docs/experiment_matrix.md

This document becomes the master plan for implementation.

---

## Prompt 4 — Evaluation protocol

Review every planned experiment.

Recommend a fixed evaluation protocol.

For every metric explain:

- Why it is included
- Which research question it answers
- How it will be interpreted

Generate:

docs/evaluation_protocol.md

---

## Prompt 5 — Reproducibility checklist

Create a reproducibility checklist.

Include:

- Random seeds
- Library versions
- Hardware
- Dataset version
- Train/validation/test split
- Tokenizer version
- Model checkpoints
- Configuration files
- Logging
- Experiment tracking

Generate:

docs/reproducibility_checklist.md

---

## Prompt 6 — Pre-implementation review (three reviewers)

Act as three anonymous reviewers for a top NLP conference.

Review ONLY our research design.

Critique:

- Novelty
- Dataset choice
- Experimental fairness
- Leakage controls
- Evaluation protocol
- Threats to validity

List every weakness.

For each weakness:

- Explain why it matters.
- Suggest how to fix it before implementation.

Estimate the likelihood that a reviewer would raise this concern.

Save:

docs/pre_implementation_review.md

---

## Prompt 7 — This prompt log

up until now, can you create a file called prompt1.md and list down all the promt in this context window from the begin with?
