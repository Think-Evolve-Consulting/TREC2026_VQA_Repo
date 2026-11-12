# Session Notes — 2026-10-12 → 2026-11-09 — Speaker abstract, notebook paper, official-results analysis, markdown archive

A writing-and-analysis session with no GPU work. It reviewed every markdown
document in the project, read the **official NIST results** for both tasks for
the first time, and produced six deliverables:

- a speaker-proposal abstract;
- a TREC notebook paper;
- a markdown inventory (text and Excel);
- a Q&A answer on debate versus voting;
- a zip archive of all project markdown;
- this record.

Everything that existed only in the chat (the abstract, the Q&A answer, the
official-results analysis) is reproduced here so that it survives the session.

**Status: complete.** No model was run and no run file was changed. One package
(`openpyxl`) was installed into `.venv`, and one memory note was corrected.

---

## Contents

1. [Requests, in order](#1-requests-in-order)
2. [Sources read](#2-sources-read)
3. [Speaker-proposal abstract](#3-speaker-proposal-abstract)
4. [Official NIST results — first analysis](#4-official-nist-results--first-analysis)
5. [The notebook paper](#5-the-notebook-paper)
6. [Markdown inventory and virtualenv files](#6-markdown-inventory-and-virtualenv-files)
7. [Q&A answer: debate vs. voting at equal budget](#7-qa-answer-debate-vs-voting-at-equal-budget)
8. [Excel inventory and zip archive](#8-excel-inventory-and-zip-archive)
9. [Environment and memory changes](#9-environment-and-memory-changes)
10. [Files created](#10-files-created)
11. [Open items — things to fill in or verify](#11-open-items--things-to-fill-in-or-verify)
12. [How to reproduce](#12-how-to-reproduce)

---

## 1. Requests, in order

| # | Date | Request | Outcome |
|---|---|---|---|
| 1 | 12 Oct | Review all markdowns; write an abstract for a speaker proposal titled *"Multi-agent debate for VLMs: Does it survive honest evaluation?"* | Abstract delivered in chat (§3), with a warning that debate itself was never run at scale |
| 2 | 4 Nov | Review all markdowns, including subfolders; list them in a txt file; document them all in one markdown file written as a notebook paper for proceedings | `MARKDOWN_INVENTORY.txt` and `TREC2026_VQA_NOTEBOOK_PAPER.md` (§5, §6). Official results discovered and analysed (§4) |
| 3 | 4 Nov | "What are these 45 files inside the virtualenvs?" | Explained (§6.2) |
| 4 | 5 Nov | Answer *"How does debate compare with independent answers plus voting at the same inference budget?"* with a positive spin on local LLM setups | Q&A answer delivered in chat (§7) |
| 5 | 8 Nov | List all markdown files in an Excel file and zip them with the Excel file inside | `MARKDOWN_INVENTORY.xlsx` and `TREC2026_markdown_archive_2026-11-08.zip` (§8) |
| 6 | 9 Nov | Export this session to a markdown file, documented thoroughly | This file |

---

## 2. Sources read

**Request 1:** the five debate and judge documents read in full:

- `# Two-judge debate over the ranker swarm.md`
- `SESSION.2026-09-17.debate-system.md`
- `SESSION.2026-09-18.debate-mac-local.md`
- `Multi_Agent_VQA_Session_Doc.md`
- `SESSION.2026-09-17.orchestrator-judge.md`

Plus targeted reads of `MODEL_SCORECARD.2026-09-17.md`,
`SESSION.2026-09-16.task2-extra-rankers.md` and `ARCHITECTURE.reconciled.md`.

**Request 2:** a full recursive scan found 97 markdown files. Every one of the
42 project documents was read. The three `TRANSCRIPTION.language-review*.md`
files were read in their summary sections; the rest of each is a per-video
listing. The 10 vendor model cards were read for model facts only. The 45 files
inside the virtualenvs were listed but not read.

**The official results**, found while checking what was submitted:

| File | Contents |
|---|---|
| `FINAL RESULTS/VQA_AG_effectiveness_scores.xlsx` | Task 1 (AG), all 44 runs. Sheets: *Summary*, *vqa mcq scores*, *Human Answer Only*, *All References*, *Definitions* |
| `FINAL RESULTS/vqa.mcq.scores.txt` | Task 2 (MC), all 42 runs: Top-1 and MRR over 2,428 scored queries (15 excluded) |

`openpyxl` was not installed at the time, so the workbook was read by parsing
its XML with the standard library (`zipfile` + `xml.etree`). Memory notes that
were consulted: `trec2026-ensemble-submission`, `task2-mc-ranking-pipeline`,
`task2-mc-dev-harness` and `judge-rerank-gate`.

---

## 3. Speaker-proposal abstract

Delivered in chat for an audience of conference programme reviewers.
Reproduced verbatim:

> **Multi-Agent Debate for VLMs: Does It Survive Honest Evaluation?**
>
> Multi-agent debate is often presented as a cheap way to get more out of
> vision-language models: let models disagree, let them argue, and keep the
> answer they converge on. This talk tests that idea on a real benchmark, the
> TREC 2026 Video Question Answering track, under conditions where a positive
> result has to be earned.
>
> We ran eight open-weight VLMs over 2,443 multiple-choice video questions.
> Then we spent a month trying to beat the best single model (Qwen3.8-27B,
> 0.800 Top-1 on a 400-query dev set with ground truth) by combining the others.
> Every attempt was held to a gate fixed before results were read: an exact
> McNemar test plus a minimum gain, scored alongside a null judge and
> majority-vote controls. Rank fusion lost or tied. A search over 781,248
> weighted fusions reached 0.825 in-sample but fell to 0.801 under nested
> cross-validation. A VLM judge that re-watched the video, with every
> panellist's ranking in front of it, overruled the baseline 39–61 times. Those
> overrules split almost evenly between fixes and breaks (16 vs 17, 24 vs 27),
> so the judge ended level with the baseline.
>
> The useful finding came from comparing two judges. Where they disagreed, at
> least one was right 38 times out of 50, yet each alone got only 18–20, and
> the baseline got 24. That headroom is what debate is meant to capture. We show
> how to build a debate that cannot make results worse: it can only change an
> answer on contested queries where the judges converge, and a deadlock falls
> back to the baseline. We also show where debate cannot help. On counting
> questions, even an oracle over both judges stays low, because the frames don't
> contain the evidence and arguing about them won't add it.
>
> Attendees will leave with an evaluation protocol for multi-agent systems:
> pre-registered gates, null controls, nested CV and paired tests. They will
> also take away a clear lesson. Disagreement between models usually means the
> question is hard, not that one model knows the answer.

**The caveat given with it:** the debate system was built and verified against
mock models and against the real InternVL judge, but it **never ran at scale**.
The real smoke test stopped because the Mac-hosted Qwen judge returned nothing
in 30 minutes. No debate scores exist on dev or on 2026. Two options were
offered:

1. **Run it before the talk**, over the 203 disputed dev queries, with the Qwen
   judge on the 3090.
2. **Keep the abstract as written**, presenting debate as the remaining
   headroom rather than as a measured result.

---

## 4. Official NIST results — first analysis

This is the most consequential new information from the session. The project's
working notes predate these scores.

### 4.1 Our submitted runs

| Task | Run tag (as NIST lists it) | Local file (inferred from the name; to verify) |
|---|---|---|
| AG | `gemma4_run01` | `gemma4.run.task1.json` (Gemma 4 via Ollama, 10 prompts × 3) |
| AG | `gemma4_expand_run1` | `gemma4.expanded.run.task1.json` |
| AG | `qwen3vl_expand_run2` | `qwen3vl.expanded.run.task1.json` (Qwen3-VL-8B-Thinking) |
| AG | `fused_expan_run1` | `fused.expanded.run.task1.json` |
| AG | `fused_judge_exp_run1` | `fused.judged.expanded.run.task1.json` |
| MC | `qwen38` | `qwen38.run.task2.json` |
| MC | `qwen38.resampled` | `qwen38.resampled.run.task2.json` |
| MC | `qwen3vl.run` | `qwen3vl.run.task2.json` |
| MC | `internvl.run` | `internvl.run.task2.json` |
| MC | `fused.weighted` | `fused.weighted.run.task2.json` |

The MC set differs from the five the session notes recommended (qwen38 →
qwen3vl → fused.weighted → gemma4.vllm → fused.rrf). The submission used
`qwen38.resampled` and `internvl.run` in place of `gemma4.vllm` and `fused.rrf`.

### 4.2 The scoring change that matters

The 2026 AG headline metrics are **@1**, the score of the rank-1 answer alone.
NDCG@10 is reported alongside, with each metric used as graded relevance. The
workbook's *Definitions* sheet defines its terms:

- **Best answer score:** at k=1, the rank-1 score; at k=5 or 10, the best score
  among the top k answers.
- **Mean top-k score:** the mean across the top k ranks, with missing ranks
  counted as zero.
- **Averaging:** all scores are macro-averaged.

The 2025 scorer (`eval.vqa.ag.py`), which the whole project developed against,
scores a query as the **mean over all ten answers**. That makes answer order
irrelevant. The project's conclusion that ordering cannot move METEOR was
therefore true for 2025 and **false for 2026**.

There are also two reference modes: *Human Answer Only* and *Human +
Alternative Answers*. The second raises every score by about 0.02–0.03 and
leaves our relative order unchanged.

### 4.3 Task 1 (AG) — human-answer-only, 44 runs, rank in parentheses

| Run | STS@1 | METEOR@1 | BERTScore@1 | STS NDCG@10 | METEOR NDCG@10 | BERTScore NDCG@10 |
|---|---|---|---|---|---|---|
| `gemma4_run01` | 0.4573 (23) | 0.1761 (39) | 0.8887 (18) | 0.8896 (35) | 0.7038 (44) | 0.9924 (35) |
| `gemma4_expand_run1` | 0.4228 (30) | 0.2140 (28) | 0.8860 (23) | 0.8876 (38) | 0.7688 (36) | 0.9942 (28) |
| `fused_judge_exp_run1` | 0.4067 (35) | 0.2132 (31) | 0.8829 (31) | 0.8906 (34) | 0.7779 (32) | 0.9934 (30) |
| `fused_expan_run1` | 0.4019 (38) | 0.2115 (32) | 0.8827 (32) | 0.8877 (37) | 0.7789 (31) | 0.9934 (31) |
| `qwen3vl_expand_run2` | 0.3620 (44) | 0.1675 (43) | 0.8802 (36) | 0.8894 (36) | 0.7426 (43) | 0.9950 (27) |
| field median | 0.4582 | 0.2340 | 0.8860 | 0.9263 | 0.8104 | 0.9956 |
| field best | 0.5112 | 0.2976 | 0.8988 | 0.9648 | 0.8813 | 0.9992 |

**Best@1 against Best@10:**

| Run | STS Best@1 → Best@10 (rank) | BERTScore (rank) | METEOR (rank) |
|---|---|---|---|
| `gemma4_run01` | 0.457 (23) → 0.639 (**9**) | 0.889 (18) → 0.917 (**9**) | 0.176 (39) → 0.335 (23) |
| `gemma4_expand_run1` | 0.423 (30) → 0.600 (14) | 0.886 (23) → 0.907 (23) | 0.214 (28) → 0.368 (**13**) |
| `fused_expan_run1` | 0.402 (38) → 0.593 (16) | 0.883 (32) → 0.909 (18) | 0.212 (32) → 0.364 (14) |
| `qwen3vl_expand_run2` | 0.362 (44) → 0.512 (39) | 0.880 (36) → 0.899 (36) | 0.168 (43) → 0.290 (37) |

**Readings:**

1. **The model ordering held.** Gemma 4 beat Qwen3-VL on every @1 metric
   (compared expanded against expanded), as dev predicted.
2. **Expansion traded STS for METEOR.** From `gemma4_run01` to
   `gemma4_expand_run1`, METEOR@1 rose by **+0.038**, matching the dev
   prediction of +0.039, and METEOR NDCG@10 rose by +0.065. But STS@1 fell by
   **−0.035** and BERTScore@1 by −0.0027. STS was the one official metric the
   dev loop could not compute, because the UMBC service is dead. Our unexpanded
   run is our best on STS and BERTScore.
3. **Fusion lost.** `fused_expan_run1` trails `gemma4_expand_run1` on every @1
   metric: METEOR −0.0025, STS −0.021, BERTScore −0.003. Its only gain is
   METEOR NDCG@10 (+0.010). This matches the post-hoc dev result in
   `MODEL_SCORECARD.2026-09-17.md`, where fused scored 0.133 against Gemma 4's
   0.148 (p = 0.041).
4. **The judge rerank changed little.** It moved the fused run by +0.0017
   METEOR@1 and +0.0048 STS@1, which cannot be distinguished from zero without
   per-query scores.
5. **The candidate sets were stronger than the rank-1 choices.** Our Best@10
   values sit in the top quarter of the field (Gemma: 9th of 44 on STS and
   BERTScore), while our rank-1 answers place in the middle or below. Some of
   this reflects Best@10 rewarding diverse answer sets, which the paraphrase
   ensemble was built to produce. But it is also the direct cost of having
   developed against a scorer that made ordering irrelevant. Gemma's METEOR
   NDCG@10 is last of 44.

### 4.4 Task 2 (MC) — 42 runs, 2,428 scored queries

| Run | Dev Top-1 | Official Top-1 (rank) | Official MRR (rank) | Correct |
|---|---|---|---|---|
| `qwen38` | 0.800 | **0.8027** (20) | **0.8767** (15) | 1,949 |
| `qwen38.resampled` | — | 0.8019 (21) | 0.8757 (17) | 1,947 |
| `fused.weighted` | 0.795 | 0.7846 (27) | 0.8659 (22) | 1,905 |
| `qwen3vl.run` | 0.790 | 0.7537 (32) | 0.8433 (32) | 1,830 |
| `internvl.run` | 0.760 | 0.7401 (33) | 0.8355 (33) | 1,797 |
| field median | | 0.8015 | 0.8665 | |
| field best | | 0.8476 | 0.9081 | |

**Readings:**

1. **Our best model transferred almost exactly:** 0.800 on dev, 0.803
   official.
2. **The dev "tie" was not a tie.** Qwen3-VL was tied with Qwen3.8 on dev
   (−1.0 points, p = 0.70) but finished **4.9 points behind** on 2026, 119
   fewer correct answers. The dev set is four-option, short video and unrouted;
   2026 is 81% eight-option, has a long tail, and was routed.
3. **Fusion fell below its best member** (0.785 against 0.803), as nested CV
   had predicted.
4. **The resampling workaround made no difference** (0.8019 against 0.8027),
   confirming the dev finding.

---

## 5. The notebook paper

**File:** `TREC2026_VQA_NOTEBOOK_PAPER.md`, about 8,600 words.
**Title:** *Measure Before You Ship: An Open-Weight, Single-GPU System for the
TREC 2026 Video Question Answering Track.*
**Audience:** TREC proceedings readers and reviewers.

**Structure:**

| § | Section | Main content |
|---|---|---|
| — | Abstract | Constraints, method, negative results, official placement |
| 1 | Introduction | One GPU, open weights, short timeline; *measure before you ship*; contributions |
| 2 | Task and data | Both tasks; the @1 scoring change; 2026 corpus profile; dev sets and how they differ from 2026 |
| 3 | Hardware, serving, models | 22.72 GiB usable; schedule by resident model; ten-model table |
| 4 | Shared preprocessing | Dual ASR, frames, paraphrases, six-agent router, the shift from a multi-agent design to a schedule |
| 5 | Development methodology | Scorer parity, oracle and null controls, paired tests, pre-registered gates, nested CV, silent-failure guards; the STS gap |
| 6 | Task 1 | Three architectures and their error profiles; dev results; expansion A/B/C; ordering and noise floor; ensemble (gate skipped); judge rerank; official results and their reading |
| 7 | Task 2 | Ranking contract; eight rankers; model-specific traps; fusion with nested CV; error anatomy; judge, two judges and debate; official results and dev→test transfer |
| 8 | Throughput | Task 2 prefill/decode table with caveats |
| 9 | Engineering lessons | Twelve silent failures: how each presented and how it was caught |
| 10–11 | Discussion, conclusion, future work | Counting, audio, rank-1 selection, measuring debate on dev, local STS |
| A–B | Appendices | Section-to-source map; reproducibility notes |

**Editorial decisions:**

- **Results lead with the official scores**, with dev numbers presented as
  predictions to compare against.
- **Every number comes from a project document or the NIST files.** Nothing was
  estimated.
- **Debate is described as built but unmeasured.** The two-judge oracle figure
  (0.843) is labelled as a ceiling, never as a debate result.
- **The crowd-density session and the vendor model cards are excluded**, as
  unrelated or third-party.
- **References were limited to works whose details are certain.** The ITI-CERTH
  and HLTCOE 2025 notebooks carry `[citation to be completed]` rather than
  invented titles.
- **Authors and team are placeholders.**

---

## 6. Markdown inventory and virtualenv files

### 6.1 `MARKDOWN_INVENTORY.txt`

Recursive scan, 4 Nov: **97 markdown files**.

| Section | Count | Treatment |
|---|---|---|
| A. Project documents | 42 | All reviewed. Grouped A.1 track material, A.2 data prep and audio, A.3 Task 1, A.4 dev evaluation, A.5 Task 2, A.6 unrelated (crowd density). Each has a one-line description |
| B. Vendor model and weight cards | 10 | Reviewed for model facts only |
| C. Virtualenv package files | 45 | Listed only |

A counting slip was corrected before delivery. The first draft said 41 project
documents and 11 vendor cards; the correct figures are 42 and 10.

### 6.2 What the 45 virtualenv files are

They are documentation that `pip` installs along with third-party packages:
40 files in `.venv/` and 5 in `.venv-videollama3/`. None was written for this
project.

| Kind | Count | Examples |
|---|---|---|
| Licence and author files | 27 | numpy (11 alone), httpx, idna, librosa, nltk `AUTHORS.md` |
| Templates for generated model or dataset cards | 8 | `huggingface_hub`, `sentence_transformers` |
| Internal readmes for library developers | 8 | plotly `figure_factory`, torchgen, onnxruntime mobile op lists |
| Other bundled notes | 2 | onnxruntime `Privacy.md`; typer's `SKILL.md` (instructions for AI coding agents) |

They are harmless, and they disappear and return whenever an environment is
rebuilt.

---

## 7. Q&A answer: debate vs. voting at equal budget

The user asked for a positive framing of local LLM setups. The answer framed
local inference favourably but **did not claim a debate result**, because no
budget-matched comparison was ever run. Reproduced verbatim:

> "We haven't run debate against voting at a matched budget yet, so I won't
> claim a winner. What we have measured is the voting side, and it's a high
> bar.
>
> On our hardware, 'independent answers plus voting' is what every ranker
> already does. Each model ranks every question three times, with the options
> presented in a different order each time, and we combine the three with a
> Borda vote. That costs three calls per question. Our best model run that way
> scored 0.800 on our development set and 0.803 in the official results.
>
> The debate we built costs about the same. Two opening opinions, plus rebuttal
> rounds only on the questions where the judges still disagree, comes to
> roughly 2.8 calls per disputed question.
>
> What we know about the arguing side so far: a single judge that re-examined
> the video with every model's answer in front of it was a coin flip. It
> overruled the baseline 39 times, fixed 16 and broke 17. But when two judges
> disagreed, at least one of them was right 38 times out of 50, while the
> baseline got 24. That gap is exactly the headroom debate is designed to
> capture.
>
> Running everything locally is what made this measurable. There were no
> per-token bills, so we could run null-judge and majority-vote controls, a
> 781,248-fusion search with nested cross-validation, and a debate whose output
> is guaranteed never to be worse than the baseline. The budget-matched
> experiment is a single overnight run on hardware we already own."

**Where the 2.8 comes from:** 2 opening calls per disputed query, plus the
rounds R1 and R2, which the debate plan estimates at about 40% of the opening
cost. So 2 × 1.4 ≈ 2.8 calls.

**Backup facts prepared for follow-up questions:**

| Question | Answer |
|---|---|
| Plain majority voting across different models? | 129 of 203 against the baseline's 133. Every model loses more queries to the leader than it wins |
| Why expect debate to do better? | New evidence rather than more opinions: 32–64 fresh frames, a report-the-maximum-count instruction, CLAP scores for 135 sound questions. Oracle headroom about +17 queries |
| Where it can't help | Counting: the two-judge oracle gets only 24 of 38 |
| Risk | None by construction: a deadlock falls back to the baseline, verified with never-conceding mocks |
| Why not run yet | The 32 GB Mac judge was too slow. On the 3090, with judges taking turns on the card, the speed is already known |
| What local buys | Free repetition, data stays on site, bit-identical cached reruns, free model swaps |

**The line to avoid:** saying "debate scores 0.843". That figure is an oracle
ceiling, not a result.

---

## 8. Excel inventory and zip archive

**Scope:** the 53 markdown files outside the virtualenvs (the 42 project
documents, the newly written paper, and 10 vendor cards). The 45 package files
were excluded, and the user was offered their inclusion.

### 8.1 `MARKDOWN_INVENTORY.xlsx` (15 KB)

| Sheet | Contents |
|---|---|
| **Markdown files** | One row per file: #, Category, Group, File name, Folder, Relative path, Path in archive, Size (bytes), Last modified, Description, SHA-256. Bold header row, frozen panes and an autofilter on every column |
| **Summary** | Files and bytes per category (42 project, 1 side project, 10 vendor), the total, a note on the excluded files, the source folder and the generation date |

Descriptions and groups were taken from `MARKDOWN_INVENTORY.txt`. The paper was
added as group *A.7 Write-up*. The build script refuses to write the workbook
if any row lacks a description.

### 8.2 `TREC2026_markdown_archive_2026-11-08.zip` (399 KB; the markdown totals 871 KB)

```
TREC2026_markdown_archive_2026-11-08.zip
├── MARKDOWN_INVENTORY.xlsx
└── markdown/
    ├── <42 project documents and the paper, at their original paths>
    ├── VQA/…  colab/…
    ├── models/…            (vendor cards)
    └── crowd_density/…     (weights stub)
```

**Verification performed by reopening the archive:**

- `testzip()` reported no corrupt members.
- 54 members: 53 markdown files plus the workbook.
- The SHA-256 of every markdown member matched its source file and its row in
  the workbook.

**Note:** this session file was written after the archive and is **not inside
it**. Re-running the build script (§12) would include it.

---

## 9. Environment and memory changes

| Change | Detail |
|---|---|
| Package installed | `openpyxl 3.1.5` into `E:\TREC_2026\.venv` (needed to write the workbook). Nothing else changed |
| Memory corrected | `trec2026-ensemble-submission.md` said ordering cannot change the score. A superseding paragraph now records that 2026 AG scoring is @1, so ordering matters, and adds the official expansion and fusion results. The `MEMORY.md` index line was updated to match |
| Not changed | No run file, cache, model, server or GPU state was touched |

---

## 10. Files created

| File | Location | Purpose |
|---|---|---|
| `TREC2026_VQA_NOTEBOOK_PAPER.md` | project root | The notebook paper |
| `MARKDOWN_INVENTORY.txt` | project root | Text inventory of all 97 markdown files |
| `MARKDOWN_INVENTORY.xlsx` | project root, and inside the zip | Excel inventory of the 53 non-virtualenv files |
| `TREC2026_markdown_archive_2026-11-08.zip` | project root | The 53 markdown files plus the workbook |
| `SESSION.2026-10-12.notebook-paper-and-archive.md` | project root | This record |

**Helper scripts**, all in the session scratchpad
(`C:\Users\ADMIN\AppData\Local\Temp\claude\371e7b77365f263e\5eca8fca-6546-40d0-b98a-f8fcb0172daa\scratchpad\`).
The scratchpad is temporary: copy anything worth keeping.

| Script | Purpose |
|---|---|
| `build_md_archive.py` | builds the workbook and zip, then verifies them |
| `%TEMP%\readxlsx.py` | stdlib-only xlsx reader used before openpyxl existed |
| `%TEMP%\rank.py` | ranks our AG runs against the field per metric |
| `%TEMP%\detail.py` | extracts Best@k and Mean@k detail for our AG runs |
| `%TEMP%\rank10.py` | ranks Best@1 against Best@10 |

---

## 11. Open items — things to fill in or verify

1. **Paper placeholders:** author names and team ID.
2. **Run-tag mapping (§4.1):** confirm that each NIST tag corresponds to the
   local file inferred from its name.
3. **Two incomplete references:** the ITI-CERTH and HLTCOE TREC 2025 VQA
   notebook papers.
4. **The judge gate on dev was never recorded.** If `judge.2025.cache.jsonl`
   was ever produced, its result belongs in paper §6.6.
5. **The speaker-proposal framing:** either run debate on the 203 disputed dev
   queries before the talk, or present it as the remaining headroom. The
   abstract in §3 is accurate under the second option.
6. **The archive excludes this file and the 45 virtualenv files.** Rebuild it
   if either should be included.
7. **Disk:** the earlier notes flag `models/Muse-Glimmer-30B/` (55.5 GB, unused)
   as deletable. That is unchanged by this session.

---

## 12. How to reproduce

```powershell
# Re-scan for markdown files (bash form, excluding the virtualenvs)
find . -type f -iname "*.md" -not -path "./.venv*" | sort

# Rebuild the workbook and zip (verifies SHA-256 of every member)
$env:PYTHONIOENCODING="utf-8"
E:\TREC_2026\.venv\Scripts\python.exe -I `
  "C:\Users\ADMIN\AppData\Local\Temp\claude\371e7b77365f263e\5eca8fca-6546-40d0-b98a-f8fcb0172daa\scratchpad\build_md_archive.py"
```

To read the official AG workbook now that openpyxl is installed:

```python
import openpyxl
wb = openpyxl.load_workbook(r"E:\TREC_2026\FINAL RESULTS\VQA_AG_effectiveness_scores.xlsx", read_only=True)
print(wb.sheetnames)  # Summary, vqa mcq scores, Human Answer Only, All References, Definitions
```

Two notes on the build script:

- It **reads descriptions from `MARKDOWN_INVENTORY.txt`**. A new markdown file
  needs a line added there, or a default in the script, before it will build.
- **The output names carry a fixed date** (`STAMP = "2026-11-08"`). Change it
  to avoid overwriting the existing archive.
