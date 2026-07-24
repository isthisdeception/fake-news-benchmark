# Prompt Log

All prompts issued in this session, in order from the beginning.

---

## Prompt 1 — Build the research database

You are not an AI assistant.

You are the literature review team in our NLP research lab.

Your objective is to convert our PDF collection into a structured research database.

Read every paper inside

literature/papers/

For EACH paper create

literature/notes/PXXX.md

using EXACTLY this template.

------------------------------------------------

Paper ID

Title

Authors

Year

Venue

DOI

Research Objective

Research Questions

Dataset(s)

Dataset Size

Language(s)

Task

Model(s)

Architecture Type

Preprocessing Pipeline

Training Strategy

Hyperparameters

Evaluation Metrics

Baseline Models

Main Results

Strengths

Weaknesses

Limitations

Future Work

Interesting Ideas

Possible Inspiration For Our Paper

Potential Citation Usage

Reviewer Notes

------------------------------------------------

Never guess.

If something is unavailable write

Not Reported

Do not summarize.

Extract information precisely.

---

## Prompts 2–5 — Subagent completion follow-ups (auto-generated, repeated after each of the 4 extraction workers finished)

Perform any necessary follow-up actions in response to the subagent completion above. If no follow-up work is needed, no further action is required. [Standard linking/formatting instructions omitted.]

---

## Prompt 6 — Master CSV database

Read every markdown file inside

literature/notes/

Create

literature/master_literature_database.csv

Include

PaperID

BibKey

Year

FirstAuthor

Venue

Dataset

Models

Metrics

Task

Contribution

Keywords

ReadyForCitation

One row per paper.

---

## Prompt 7 — Systematic pattern analysis

Using only the extracted literature database,

perform a systematic literature analysis.

Identify

1. Most common datasets

2. Most common transformer models

3. Most common preprocessing pipeline

4. Most common optimizer

5. Most common scheduler

6. Most common batch size

7. Most common evaluation metrics

8. Common hyperparameter ranges

9. Frequently reported limitations

10. Frequently suggested future work

11. Papers with strongest methodology

12. Papers with weakest methodology

Save as

literature/pattern_analysis.md

Support every observation using paper IDs.

---

## Prompt 8 — Research gap analysis (senior NLP professor)

Act as a senior NLP professor.

Read

literature_database

pattern_analysis

Identify every possible research gap.

Rules

A research gap MUST be supported by multiple papers.

For every gap provide

Gap

Supporting Papers

Evidence

Why it matters

Novelty Score (1-10)

Difficulty

Publication Potential

Feasibility for one researcher using Kaggle

Rank every gap.

Save

literature/research_gap.md

---

## Prompt 9 — Reviewer #2 observations

Pretend you are Reviewer #2.

Read every extracted paper.

Identify

Common mistakes

Weak evaluation

Poor methodology

Data leakage

Missing statistical tests

Missing baselines

Weak discussion

Weak reproducibility

Generate

literature/reviewer_observations.md

---

## Prompt 10 — Design the benchmark study

Using every reviewed paper,

design our entire benchmark study.

Output

Research Questions

Hypotheses

Dataset Justification

Model Selection

Experimental Protocol

Training Strategy

Evaluation Metrics

Statistical Analysis

Error Analysis

Efficiency Analysis

Threats to Validity

Expected Contributions

Save

docs/experimental_design.md

---

## Prompt 11 — Critique the design

Critique experimental_design.md.

Find every weakness.

Suggest improvements.

Do not rewrite.

Only critique.

Think like a top-tier reviewer.

---

## Prompt 12 — Generate the immutable protocol

Review experimental_design.md.

Generate

docs/final_protocol.md

This protocol becomes immutable.

All future code must follow it.

Do not modify experimental settings after this point unless explicitly instructed.

---

## Prompt 13 — Dataset selection report

You are an NLP researcher.

Read every reviewed paper in our literature database.

Analyze all datasets used for fake news detection.

For each dataset report:

- Name
- Number of papers using it
- Task type
- Number of samples
- Number of classes
- Language
- Advantages
- Limitations
- Licensing or availability (if reported)
- Suitability for benchmarking modern transformer models

Finally, recommend the best dataset for OUR study.

Your recommendation must be evidence-based.

Do not choose based on popularity alone.

Consider:
- Fair comparison with prior work
- Dataset quality
- Reproducibility
- Availability
- Computational feasibility
- Publication potential

Save the results as:

literature/dataset_selection_report.md

---

## Prompt 14 — This request

up until now, can you create a file called prompt.md and list down all the promt in this context window from the begin with?
