# Kaggle Operating Runbook — How to Actually Run This Project

This is the **step-by-step operating guide** (the "how do I run it"). It complements
`MASTER_IMPLEMENTATION_GUIDE.md` (the "what to build"). If you ever feel lost, come back here.

---

## 1. The Big Picture (3 places, 1 loop)

```
   ┌─────────────┐   git push    ┌─────────────┐   git clone/pull   ┌──────────────┐
   │   CURSOR    │ ────────────▶ │   GITHUB    │ ─────────────────▶ │    KAGGLE    │
   │ (write code)│               │ (code home) │                    │  (runs code, │
   │             │ ◀──────────── │             │                    │   has GPU +  │
   └─────────────┘  you commit   └─────────────┘                    │   datasets)  │
         ▲          outputs here                                     └──────┬───────┘
         │                                                                  │
         │                     copy/paste the output text                   │
         └──────────────────────────────────────────────────────────────────┘
```

- **Cursor** = write code (with my prompts) and push it to GitHub. Almost no running here.
- **GitHub** = stores your code so Kaggle can download it.
- **Kaggle** = the machine that actually runs the code (free GPU + your datasets live here).
- **This chat** = where you paste Kaggle's output so we continue to the next step.

**Key idea:** code flows Cursor → GitHub → Kaggle. Results flow Kaggle → (you copy) → Cursor/chat.

---

## 2. One-Time Setup (do this ONCE, before Step S1)

You only do these five things a single time.

### 2.1 Put the project on GitHub
1. Create a free account at github.com if you don't have one.
2. Create a new **private** repository, e.g. `fake-news-benchmark`.
3. In Cursor's terminal, from the project folder, run (I can do this for you when you're ready):
   ```
   git init
   git add .
   git commit -m "chore: initial project (docs + guides)"
   git branch -M main
   git remote add origin https://github.com/<your-username>/fake-news-benchmark.git
   git push -u origin main
   ```
Now GitHub has your project.

### 2.2 Create a Kaggle account and unlock the GPU
1. Sign up at kaggle.com (free).
2. Go to **Settings → Phone Verification** and verify your phone number. This unlocks:
   - **GPU** (the T4 we need), and
   - **Internet access** inside notebooks (needed to `pip install` and download models).

### 2.3 Find and add the datasets (one time, then reused)
The datasets already exist on Kaggle. For each of DS1–DS5:
1. Click **Datasets** (left menu) → search, e.g. `WELFake`.
2. Open the dataset page and note its **slug** (the `owner/dataset-name` in the URL) and its
   **version**. We record these in `configs/datasets.yaml`.
Datasets to locate (search these names): **WELFake**, **ISOT Fake News**, **FakeNewsNet**,
**COVID-19 Fake News**, **LIAR**. (We'll confirm the exact best copies together during Step S6.)

### 2.4 Create the Kaggle Notebook (your "run button")
1. Click **Create → New Notebook**.
2. On the right panel:
   - **Accelerator** → set to **GPU T4 x1** (only when running experiments; keep it **None/CPU**
     for lightweight steps like tests to save your GPU quota).
   - **Internet** → **On**.
3. **Add Data** (right panel) → search and attach each dataset from 2.3. They appear under
   `/kaggle/input/<slug>/` (read-only).
4. Save the notebook (e.g., name it `fnb-runner`). You will **reuse this same notebook** for every
   step — you just change what it runs.

That's the entire one-time setup. From here on it's the repeating loop below.

---

## 3. The Repeating Loop (do this for EVERY step S1…S32)

### Step A — In Cursor (write code)
1. Tell me which step we're on (e.g., "do Step S1"). I generate the files.
2. When it looks good, commit and push (I can run these for you):
   ```
   git add .
   git commit -m "feat(S1): project foundation"
   git push
   ```
Now GitHub has the new code.

### Step B — In Kaggle (run code)
Open your `fnb-runner` notebook. The **first cell** always pulls your latest code and installs it:

```python
# Cell 1 — get the code and install it (run this at the start of every session)
!git clone https://github.com/<your-username>/fake-news-benchmark.git
%cd fake-news-benchmark
!pip install -e . -q
```

> If the repo is **private**, use a GitHub Personal Access Token in the URL:
> `!git clone https://<TOKEN>@github.com/<your-username>/fake-news-benchmark.git`
> (Create the token at GitHub → Settings → Developer settings → Fine-grained tokens, read-only.)

Then a **second cell** runs the actual step. Examples:

```python
# Run unit tests (use CPU accelerator for this — cheap)
!pytest -q
```

```python
# Run a data-pipeline stage or an experiment (use GPU T4 for training experiments)
!python scripts/run_data_pipeline.py        # e.g. after Milestone 2
# or
!python scripts/run_experiment.py --exp EXP-1 --seed 13
```

Click **Run All** (or run the cells top to bottom).

### Step C — Get the output back to Cursor/chat
Kaggle writes outputs into `/kaggle/working/`. You have two easy ways to bring them back:

- **Small text outputs (test results, a CSV, a log):** just **select the cell's output text and
  copy it**, then paste it into this chat. That's usually all I need.
- **Files you want to keep in the repo (result CSVs, hashes, split indices):** in the notebook's
  right panel under **Output**, download the file, then in Cursor drop it into the correct folder
  (see the mapping in §4) and commit it.

### Step D — Repeat
Move to the next step. Done.

---

## 4. Where does each output go? (the mapping)

When you bring a file back from Kaggle, put it in the repo like this (commit small ones with git;
DO NOT commit big ones — see §5):

| Kaggle produces… | Put it here in the repo | Commit to git? |
|------------------|--------------------------|:---:|
| Test results / console logs | (just paste into chat) | no |
| `SNAPSHOT_HASHES.txt` | `data/SNAPSHOT_HASHES.txt` | ✅ yes (tiny) |
| Split index files `*.idx` | `data/splits/` | ✅ yes (small) |
| Result tables (`results/*.csv`) | `results/` | ✅ yes (small) |
| Run logs (`logs/*.log`) | `logs/` | ✅ yes (small) |
| Processed data (`*.parquet`) | stays on Kaggle | ❌ no (too big) |
| Model checkpoints / QLoRA adapters | stays on Kaggle | ❌ no (too big) |

Rule of thumb: **small text/CSV → repo; big binary/data → keep on Kaggle** (see §5).

---

## 5. Keeping heavy files between Kaggle sessions (important)

Kaggle sessions are **temporary** — when a session ends, `/kaggle/working/` is wiped. So after a
long run (e.g., training a model), you must **save the outputs as a Kaggle Dataset** so the next
session/experiment can use them:

1. After the run finishes, in the notebook's right panel click **Output → Create Dataset** (or
   "Save Version" which snapshots the output).
2. Give it a name like `fnb-outputs-milestone5`.
3. In your NEXT notebook run, **Add Data** and attach that output dataset. It shows up under
   `/kaggle/input/fnb-outputs-milestone5/`, and your next experiment reads from there.

This is how, for example, `EXP-C1` (adversarial) reuses the trained checkpoints from `EXP-1`
without retraining, and how you stay within the GPU-hour budget across many sessions.

---

## 6. GPU quota & sessions (so you don't run out)

- Kaggle gives a **weekly GPU quota** (roughly ~30 hours) and each session lasts up to ~9–12 h.
- Use **GPU only for training/experiments**; use **CPU (Accelerator = None)** for tests, the data
  pipeline, and figure-building — those don't need a GPU and don't burn your quota.
- The whole project targets **≤120 GPU-hours total**, so it is spread across several weeks/sessions.
  Run one experiment per session, save its output dataset, then continue next session.

---

## 7. Quick answers to your exact questions

- **"When do I log into Kaggle?"** After I generate code in Cursor and you push it to GitHub. You
  log into Kaggle to *run* that code (Section 3, Step B).
- **"How do I add the dataset?"** Once, via **Add Data** in the notebook (Section 2.3 / 2.4). After
  that it's always attached at `/kaggle/input/`.
- **"How do I clone the code?"** The first Kaggle cell does `git clone …` (Section 3, Step B, Cell 1).
  ("Clone the dataset" = clone the *repo*; the datasets are *attached*, not cloned.)
- **"How do I run the whole code?"** Run the notebook cells top-to-bottom (**Run All**). Cell 1 =
  clone+install; next cell = the step's command.
- **"How do I export the output?"** Copy the cell's text into chat for small stuff, or download the
  file from the **Output** panel and commit it into the repo folder shown in §4.
- **"Where do I paste it in the file structure?"** See the table in §4 (mostly `results/`, plus
  `data/` and `logs/`). Big files never go in the repo — they stay as Kaggle output datasets (§5).

---

## 8. Your immediate next action

Nothing to run yet — there's no code until we do Step S1. When you're ready, just say
**"do Step S1"** and I'll generate the project scaffolding. Then:
1. push it to GitHub (I'll give you the commands),
2. open your Kaggle notebook and run the Cell 1 clone+install + `pytest -q`,
3. paste the output here.

We'll go one step at a time — exactly one next step, always.
