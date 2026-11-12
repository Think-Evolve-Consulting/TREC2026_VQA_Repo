# Paper structure: Multi-Agent Debate for VLMs: Does It Survive Honest Evaluation?

The paper version of the speaker proposal in
`SESSION.2026-10-12.notebook-paper-and-archive (1).md` §3. It is a focused research paper, not a second
notebook paper. It treats the TREC 2026 VQA Multiple Choice task as the testbed and covers one question:
can combining or arbitrating between VLMs beat the best single model under honest evaluation?

Every number below comes from the notebook paper (`markdown/TREC2026_VQA_NOTEBOOK_PAPER.md`, cited as
*NP §x*), from the official NIST files in `FINAL RESULTS/` (cited as *OR*: `vqa.mcq.scores.txt` for
multiple choice, `VQA_AG_effectiveness_scores.xlsx` for answer generation), or from the source note
named. Sources are given per section so drafting can work from them directly.

**What the official files add (checked 9 Nov 2026).** All our numbers in the notebook paper match the
files. On top of them:
- **The field:** 42 MC runs from 10 teams, scored on 2,428 of the 2,443 queries (NIST excluded 15).
  Median Top-1 0.8015, best 0.8476. Our best run (`qwen38`, 0.8027) ranks 20th of 42 runs and **6th of 10
  teams** by each team's best run. It ranks 15th of 42 on MRR (0.8767).
- **Resolution:** at n = 2,428 the standard error of one run's Top-1 is about 0.8 points. The file holds
  only totals, so comparisons between official runs are unpaired (§8).
- **Other teams' fusion runs** give the paper outside evidence on when fusion helps (§7.4). That
  evidence qualifies the thesis rather than simply supporting it.

---

## Scope: what this paper claims about debate

Debate is **designed, built and verified for safety, but not measured.** It can't be run before
submission because the infrastructure isn't available. The Mac-hosted 27B judge couldn't return a
single answer in 30 minutes, and the earlier Colab path has been retired
(`SESSION.2026-09-18.debate-mac-local.md`). The paper is written around that fact rather than hiding it:

- **What is measured:** five cheaper ways to combine the models (voting, two kinds of rank fusion, a
  781,248-fusion search, a single VLM judge). Each has controls and paired tests, and two are confirmed
  on the official test set.
- **What is estimated:** the headroom debate is meant to capture, from comparing two real judges.
- **What is shown but not scored:** a debate design whose output can't be worse than the baseline,
  with that property checked against mock agents.

**Wording rules for the whole paper.** Use "headroom", "ceiling" or "designed to capture". Never write
"debate improves" or "debate scores 0.843". The 0.843 figure (150/203 on the disputed queries, scaled to
the full 400) is an oracle ceiling, not a result.

**Title.** Keep it. The paper answers it in two parts: the cheaper alternatives to debate don't survive
honest evaluation, and the gain debate could offer is real but small and bounded. If a reviewer pushes
back, an optional subtitle makes the scope explicit: *"…: Measured Negatives and a Bounded Headroom on
TREC 2026 VQA."*

---

## Abstract (revised, ≈290 words)

> **Multi-Agent Debate for VLMs: Does It Survive Honest Evaluation?**
>
> Multi-agent debate is often presented as a cheap way to get more out of vision-language models: let
> models disagree, let them argue, and keep the answer they converge on. We test the simpler forms of
> that idea on a real benchmark, the TREC 2026 Video Question Answering multiple-choice task, under
> conditions where a positive result has to be earned, and we measure how much headroom is left for
> debate itself.
>
> On one consumer GPU, we ranked a 400-question labelled development set with eight open-weight VLMs
> and the 2,443-question test set with seven. We then tried to beat the best single model (Qwen3.8-27B,
> 0.800 Top-1 on development) by combining the others. Every attempt was held to a gate written before
> results were read, with null-judge and majority-vote controls, exact paired tests, and nested
> cross-validation wherever we searched. Rank fusion tied or lost. The best of 781,248 weighted fusions
> reached 0.825 in-sample but 0.801 under nested cross-validation. A VLM judge that re-watched the video
> with every panellist's ranking in front of it overruled the baseline 39 and 61 times, but its fixes
> and breaks split evenly (16 vs 17, 24 vs 27). The official results confirmed the development
> estimates: the best single model scored 0.803 (field median 0.801), and the weighted fusion 0.785, below its own best
> member. Across 42 runs from 10 teams, other teams' fusions gained only where their members were close
> in strength.
>
> Comparing two judges showed where headroom remains. Where they disagreed, at least one was right 38
> times out of 50, yet each alone got only 18–20, and the baseline 24. That gap is what debate is meant
> to capture. We present a debate built to capture it that cannot make results worse: it changes an
> answer only on contested questions where the judges converge, and a deadlock falls back to the
> baseline. We verified that property but have not yet run debate at scale. We also show where
> arbitration cannot help. On counting questions, even an oracle over both judges gets 24 of 38,
> because the frames don't contain the evidence.
>
> The lesson: across these models, disagreement usually means the question is hard, not that one model
> knows the answer.

**What changed from the speaker abstract:**
1. "Eight VLMs over 2,443 questions" is now "eight on development, seven on the test set". Muse Glimmer
   failed its gate and never ran on 2026 (NP §7.2).
2. The official confirmation is added (0.800 → 0.803; fusion 0.785) (NP §7.7).
3. "We show how to build a debate" is now "we present a debate… verified that property but have not yet
   run debate at scale". Nothing implies a debate score.
4. The closing line now reports the paper's finding. The protocol itself is listed under contributions.
5. The field median and the cross-team fusion sentence come from `FINAL RESULTS/`. **The fusion
   sentence rests on what other teams named their runs** (§7.4). Keep it in the abstract only once
   their notebook papers confirm what those runs are. Otherwise cut it and keep the point in §7.4 as
   an observation.

---

## 1. Introduction (≈1 page)

- **Hook:** multi-agent debate and ensembling are presented as cheap ways to get accuracy out of
  existing models [Du et al. 2023]. Most evaluations lack null controls, paired tests or protection
  against search optimism.
- **The setting:** one consumer GPU (RTX 3090, 22.7 GiB usable), open weights only, a real benchmark
  with official held-out scoring. The single GPU makes every "agent" a sequential pass. That is a
  constraint, and it also makes runs cheap to repeat, so controls cost nothing extra (NP §1, §3).
- **The question:** given a panel of VLMs, does any way of combining them beat the best member? And if
  nothing cheap does, how much is left for debate?
- **Contributions:**
  1. An evaluation protocol for multi-model VLM systems: pre-registered gates, null and majority
     controls, paired tests, nested CV.
  2. Five combination methods measured under it. All tie or lose, and the two submitted to NIST
     (the single model and the weighted fusion) confirm the development predictions.
  3. A diagnostic: *disagreement means a hard query, not a recoverable one*, shown by a wins-and-losses
     table against the leader and a two-judge oracle.
  4. An estimate of debate's headroom (≈17 queries on development) and a debate design that can't make
     the output worse, with that property verified. Measuring it at scale is future work.
  5. Where arbitration cannot help: counting, where the frames lack the evidence.
  6. A boundary condition from the whole field: fusion helped other teams only where their members
     were close in strength, which matches what the wins-and-losses table predicts (§7.4).

## 2. Related work (≈0.5–0.75 page)

- Multi-agent debate for LLMs [Du et al. 2023], and later critiques that debate gains shrink against
  equal-budget self-consistency or voting. *Every citation must be verified before it goes in. The
  notebook paper only cites works whose details are certain. Keep that rule.*
- LLM/VLM-as-judge and reranking.
- Rank fusion: RRF [Cormack et al. 2009], Borda.
- Evaluation hygiene: selection bias in model search, nested cross-validation, McNemar tests.
- Video QA benchmarks, TREC VQA 2025/2026. The ITI-CERTH and HLTCOE references are still incomplete
  (session notes §11).

## 3. Testbed (≈0.75 page)

- **Task:** TREC 2026 VQA Task 2. Rank every option. Top-1 and MRR (NP §2.1).
- **Dev set:** 400 labelled 2025 queries, four options, short video, unrouted. **Test set:** 2,443
  queries, 81% eight-option, a long-tailed duration distribution, routed. NIST scored 2,428 of them and
  excluded 15 (OR). The difference between the two sets matters in §7 (NP §2.3, §7.7).
- **The field:** 42 MC runs from 10 teams (OR). Give the median and best once here, so later sections
  can refer back.
- **The panel:** a table of eight rankers with dev Top-1 and MRR, from Qwen3.8-27B (0.800 / 0.884) down
  to Cosmos-Reason2-2B (0.650), plus the random control (0.253 / 0.517). Mark Muse Glimmer as dev-only,
  since it failed its gate (NP §7.2).
- **The shared ranking contract:** rank option numbers rather than text, three permuted passes combined
  by Borda, position bias at chance (NP §7.1). This is why per-model differences reflect vision and
  reasoning, not output format.
- **Hardware:** one model resident at a time (NP §3; `ARCHITECTURE.reconciled.md` §1).

## 4. Evaluation protocol (≈1 page). This is the methods contribution

Each practice comes with the failure it prevents (NP §5):

| Practice | What it prevents | Evidence it matters |
|---|---|---|
| Scorer parity with the official MC scorer | Phantom gains from rounding | MRR disagreed on 19/400 queries before matching the official 3-dp rounding |
| Null and random controls | A harness that hands out wins | Random run 0.253 / 0.517 vs theory 0.250 / 0.521; null judge 0.650 |
| Majority-vote control | Mistaking "more opinions" for signal | Majority 129/203 vs baseline 133 |
| Exact paired McNemar test | Noise read as signal | Two *random* runs differed by 3.0 points (p = 0.36); ≈4.5 points resolvable at n = 400 |
| Pre-registered gate | Moving goalposts | Judge gate: p < 0.05 **and** ≥ 8 queries gained, written before reading |
| Nested CV for any search | Search optimism | 0.825 in-sample → 0.801 nested (§5.3) |
| Silent-failure guards | Runs that exit 0 while being the baseline | A dead judge engine produced a baseline-identical run for 15.7 min before the 10-fallback abort (`SESSION.2026-09-17.orchestrator-judge.md`) |
| Official held-out confirmation | Development-set overfitting | Fusion 0.795 dev → 0.785 official; best model 0.800 → 0.803 (NP §7.7) |

**Figure 1:** a protocol diagram: candidate method → gate written → controls run alongside → paired test
→ promote or reject → official confirmation.

## 5. Cheaper combination methods (≈1.5 pages)

### 5.1 Voting and rank fusion
- Accuracy-weighted RRF over four models tied (0.795, p = 0.86), with weights fitted on the same 400
  queries. Equal-weight fusion over three models *lost* 3.5 points (p = 0.029) (NP §7.4).

### 5.2 Why: the wins-and-losses table (the diagnostic figure)
- At least one model has the right answer first on 93.5% of queries, yet every model loses more queries
  to Qwen3.8 than it wins: Qwen3-VL 29 vs 33, InternVL 25 vs 41, … Cosmos 26 vs 86 (NP §7.4).
- **Figure 2:** a wins-vs-losses bar chart against the leader, per model. The key line: *there is no
  majority to harvest.*

### 5.3 Searching harder: 781,248 fusions
- Every subset × integer weights 0–4, RRF and Borda, under 10-fold nested CV × 20 splits.
- **Table:** in-sample 0.825 / 0.896 → nested **0.801 / 0.883**. 9/20 repeats above baseline, median
  p = 0.87. Equal-weight subsets: 0.810 → 0.778, 0/20 (NP §7.4).
- **Figure 3:** the in-sample vs nested-CV distribution. The gap is pure search optimism.

### 5.4 Official confirmation
- **Table** (OR; ranks out of 42):

  | Run | Dev Top-1 | Official Top-1 | Correct of 2,428 | Rank (Top-1 / MRR) |
  |---|---|---|---|---|
  | `qwen38` | 0.800 | 0.8027 | 1,949 | 20 / 15 |
  | `qwen38.resampled` | — | 0.8019 | 1,947 | 21 / 17 |
  | `fused.weighted` | 0.795 | 0.7846 | 1,905 | 27 / 22 |
  | `qwen3vl.run` | 0.790 | 0.7537 | 1,830 | 32 / 32 |
  | `internvl.run` | 0.760 | 0.7401 | 1,797 | 33 / 33 |
  | field median / best | | 0.8015 / 0.8476 | | |

- The fusion fell 44 queries (1.8 points) below its best member. Dev had predicted it would not beat that
  member (NP §7.7).
- **State the strength of this honestly.** Unpaired, the gap is about 1.6 standard errors of a
  difference. A paired McNemar test needs per-query correctness, which needs the 2026 MC ground truth
  (not in `FINAL RESULTS/`) scored against our local run files on `E:\TREC_2026`. If NIST releases it,
  run the test and report it. Otherwise present the official gap as consistent with dev, not as
  significant on its own.

## 6. Arbitration: a judge, two judges, a debate (≈2 pages). The core of the paper

### 6.1 A single VLM judge (measured)
- Acts only on the 203/400 queries where the panel disagrees on rank 1 (the 197 unanimous ones are
  already 95% correct). It re-watches 16 frames and sees every panellist's ranking plus a reliability
  card (`SESSION.2026-09-17.orchestrator-judge.md`).
- **Table:** baseline 133 | Qwen3.8 judge 132 | InternVL judge 130 | InternVL with no card 127 |
  majority 129 | null 73 | perfect adjudicator 187 (NP §7.6).
- **The key finding:** the judges are active, not timid. They overruled 39 and 61 times, but the
  overrules are a coin flip: 16 fixed vs 17 broken, and 24 vs 27.
- **Box with qualitative examples:** Q358 (won: counted three rays of light), Q177 (won: inferred sound
  from what was visible), Q14 (lost: described the right answer, then picked the wrong option), Q115
  (lost: defensible, but not what the annotator marked). From the orchestrator-judge note.

### 6.2 Two judges read against each other: the headroom (measured)
- **Table:** judges agree on 153 queries (112 correct vs baseline 109). They disagree on 50
  (judge 1: 20, judge 2: 18, **baseline 24**, **either right: 38**). Oracle over both: 150 vs 133,
  so ≈17 queries of headroom on development.
- Three design rules follow directly:
  1. Debate only where the judges split.
  2. A deadlock falls back to the *baseline*, because it beats both judges on the contested set.
  3. Counting is out of reach: the two-judge oracle gets only 24 of 38.
- Be precise about what the 17 means. It is the gain from *always picking the right one of the two
  judges*. A debate that sees new evidence could in principle pick an answer neither judge chose, so
  the oracle measures the headroom rather than setting a strict bound. Say that in one sentence.

### 6.3 A debate that cannot make results worse (designed and verified, not run)
- Protocol: R0 independent opinions → R1 hold or switch, citing frame timestamps → R2 answer the
  opponent's citation → converged wins, otherwise the baseline (`SESSION.2026-09-17.debate-system.md`).
- New evidence the panel never had: 32 freshly cut frames (64 for counting; the panel averaged 14.5),
  a report-the-maximum-count instruction, CLAP audio scores for 135 sound questions (median
  best-to-worst spread 6.6).
- **The safety property, verified:** with two agents that never concede, the output was SHA-256
  identical to the baseline run. `--max-overrules` caps how far it can drift.
- What was verified beyond mocks: the full pipeline ran against the real InternVL judge. Three silent
  bugs were caught in testing (over-dense frame requests, a renamed CLAP keyword that scored 0 of 138,
  CLAP ranking the numeral "1" first on counting questions). These belong here because each would have
  biased a real run.
- **Figure 4:** the round flowchart with the fallback edge highlighted.
- Cost: ≈2.8 calls per disputed query (2 openings × ≈1.4 for the rebuttal rounds) vs 3 calls for the
  baseline's three-pass vote, so a *roughly matched budget*. Show the arithmetic (session notes §7).

### 6.4 Status: headroom, not a result
- One short paragraph. State plainly that debate wasn't run at scale and why (the infrastructure, per
  `SESSION.2026-09-18.debate-mac-local.md`). Give the headroom (≈17 queries on development) as the
  number to beat, and point to §10 for how it will be measured.
- Why the section still matters: the measured part of §6 (the judge, the two-judge comparison) is what
  says debate *could* help and where it can't. That is the useful input for anyone deciding whether to
  build one.

## 7. Where debate cannot help, and what dev cannot tell you (≈1 page)

### 7.1 Counting is an evidence problem
- Qwen3.8 error rate: 43.8% on counting vs 16.8% elsewhere (p = 0.0001, Fisher, Bonferroni). Misses are
  systematic undercounts: 15 of 18 readable cases too low, mean −1.2. Every model undercounts
  (NP §7.5).
- Also: "what can be heard" questions asked of video whose audio the ranker never received.

### 7.2 Agreement is a confidence signal, not a routing signal
- Split presentation passes: 46.4% error vs 14.5%. But every other model errs 48–59% on the same
  queries, so there is no better model to route them to (NP §7.5). Self-reported confidence was useless
  everywhere it was measured (NP §10).

### 7.3 Dev-to-test transfer
- The best model transferred within 0.3 points. But the dev "tie" (Qwen3-VL −1.0, p = 0.70) became a
  4.9-point gap on 2026 (1,830 vs 1,949 correct). The dev set couldn't test eight options, long video or
  routing (NP §7.7).
- **Figure 5:** a dev-vs-official scatter for the five submitted runs.
- The lesson for debate claims: even a gated dev result needs a held-out confirmation, which is one
  more reason not to report debate from dev alone.

### 7.4 The field: when does fusion help? (new, from the official results)
Four teams, ours included, submitted a run whose name says it is a fusion or average, alongside runs
that appear to be its members (OR):

| Team | Fusion run | Top-1 | Best non-fusion run | Top-1 | Fusion gain | Spread of the team's other runs |
|---|---|---|---|---|---|---|
| ISU | `isu-mc-fusefga-v2n` | 0.8476 (best in field) | `isu-mc-astra-v2n` | 0.8398 | **+19 queries (+0.8)** | astra 0.840, fable 0.836: close |
| NII_UIT | `NII_UIT-avg2_q38` | 0.8431 | `NII_UIT-huydq_qwen` | 0.8143 | **+70 (+2.9)** | members unknown from the name |
| BiTeM | `BiTeM_mcfusion` | 0.7611 | `BiTeM_mc122b` | 0.7657 | **−11 (−0.5)** | 0.766 vs 0.715–0.731: one leader |
| ours | `fused.weighted` | 0.7846 | `qwen38` | 0.8027 | **−44 (−1.8)** | 0.803 vs 0.740–0.754: one leader |

- **The reading:** fusion gained where a team's runs were close in strength, and lost where one run led
  the rest by 3–6 points. That is the pattern the wins-and-losses table (§5.2) predicts. When every member
  loses more queries to the leader than it wins, voting adds error.
- **This changes the thesis from absolute to conditional.** Not "fusion doesn't work", but "fusion
  doesn't rescue a panel with one dominant member, and you can see that in advance from dev". It also
  makes the paper stronger, because the diagnostic now predicts results outside our own runs.
- **Caveats that must go in the text:** the mapping of runs to fusions and members comes from the run
  names only. Confirm it from each team's notebook paper before publishing the table. Four teams is a
  pattern, not a test, and the ISU gain is within about one standard error.
- **Two side observations to check in the same papers.** Both names below are guesses until confirmed:
  - `g37q36_mc_text` (0.7875) looks like a text-only run. If it is, a model that never saw the video
    came within 0.3 points of our fusion. That fits the answer-length artefact in §7.5 and suggests part
    of every score here is a text prior.
  - `g38_mc_direct` (0.8431) and `NII_UIT-avg2_q38` look like Qwen3.8-based runs that scored 4 points
    above our `qwen38`. If they use the same base model, our panel leader is well below that model's
    ceiling. The serving and input choices (INT4, 16 frames, three-pass Borda) would then matter more
    than any way of combining models. That would belong in §9.

## 8. Threats to validity (≈0.5 page)

- **The central mechanism is unmeasured.** Lead with this so reviewers find it stated, not discovered.
  The paper's measured claims are about the cheaper alternatives and the headroom.
- One panel, one benchmark, one dev set of 400 (≈4.5 points resolvable).
- The reliability card and the debate rules were derived from the same dev queries the judge arms are
  scored on. The primed judge arms are mildly optimistic (removing the card costs 3 queries). A future
  debate run on those queries would carry the same optimism (§10).
- The judges saw 16 frames as images, not native video. Identical evidence for both isolates the model,
  but it caps what either can see.
- Asymmetric serving: the Qwen3.8 judge needed a Triton/bf16 workaround (18.1 s/call vs InternVL 5.4 s).
  LM Studio returns no logprobs.
- **Official comparisons are unpaired.** `vqa.mcq.scores.txt` gives totals only. Single-run standard
  error is about 0.8 points at n = 2,428. Paired tests on 2026 need the ground truth.
- **The cross-team analysis (§7.4) is inferred from run names** and covers four teams.

## 9. Discussion and practical guidance (≈0.5 page)

- **The thesis:** with a panel where every member is weaker than the leader on contested queries,
  disagreement marks hardness, not recoverable knowledge. Arguing over the same evidence doesn't add
  any.
- **When debate or fusion is worth trying:** the agents have *different evidence*, or the wins-and-losses
  table shows members that beat the leader on net. Here neither held until the debate design added new
  frames and audio. The field data (§7.4) points the same way: teams with closely matched runs gained
  from fusion, and teams with one dominant run lost.
- **Where the points may really be** (only if §7.4's side observations are confirmed): if other teams got
  4 points more from the same base model, a better single model is worth more than any arbitration over
  a weaker panel.
- **A local-inference angle** (session notes §7, framed positively but honestly): no per-token cost is
  what made the controls, the 781,248-fusion search and cached, bit-identical reruns affordable.
- **A checklist for readers** evaluating any multi-agent VLM claim: gate written first? null and
  majority controls? paired test? nested CV if anything was searched? confirmed on held-out data?

## 10. Conclusion and future work (≈0.25–0.5 page)

- **Conclusion, four sentences:** the cheaper combination methods tie or lose on this panel, and the
  official test confirms it. Across the field, fusion helped only where members were matched in
  strength, as the dev diagnostic predicts. There is real headroom only where judges disagree, and it is
  about 17 queries on development. Counting needs evidence, not argument.
- **Future work, measuring debate.** Give it as a concrete plan, since the pipeline exists:
  - Run on one GPU with the two judges taking turns (the speed on the 3090 is already known), instead
    of the Mac/LAN topology that stalled.
  - Pass the existing gates in order: does the judge actually see the frames, latency on 20 queries,
    evidence arriving, the number of changed answers matching the reported overrules, and the validator
    (`# Two-judge debate over the ranker swarm.md`).
  - Score it under the same gate as the single judge, with a no-reliability-card arm. Prefer 2026
    labels if NIST releases them, because development is where the rules were derived.
- Also: denser frames for counting, and CLAP scored against labels.

---

## Figures and tables

| # | Content | Section | Source |
|---|---|---|---|
| Fig 1 | Evaluation protocol diagram | §4 | new |
| Fig 2 | Wins vs losses against the leader, per model | §5.2 | NP §7.4 |
| Fig 3 | In-sample vs nested-CV accuracy of the fusion search | §5.3 | `SESSION.2026-09-16.task2-extra-rankers.md` (needs the per-split numbers from `E:\TREC_2026`) |
| Fig 4 | Debate rounds with the fallback edge | §6.3 | `SESSION.2026-09-17.debate-system.md` |
| Fig 5 | Dev vs official Top-1, submitted runs | §7.3 | NP §7.7, OR |
| Fig 6 | All 42 official runs, Top-1 vs MRR, ours highlighted, fusion runs marked | §7.4 | OR |
| Tab 1 | Panel: eight rankers on dev, Muse marked dev-only | §3 | NP §7.2 |
| Tab 2 | Protocol practices and the failures they prevent | §4 | NP §5 |
| Tab 3 | Fusion search: in-sample vs nested | §5.3 | NP §7.4 |
| Tab 4 | Judge arms and controls | §6.1 | NP §7.6 |
| Tab 5 | Two-judge agreement split and oracle | §6.2 | NP §7.6 |
| Tab 6 | Official results of our five runs, with ranks | §5.4 | OR |
| Tab 7 | Cross-team fusion vs best member | §7.4 | OR |

## Length and venue

About 8 pages plus references in a two-column format (a workshop on evaluation of multimodal or agentic
systems fits best). §4 and §6 are the core. If space runs short, cut §7.2 and the qualitative box in
§6.1 first. Task 1 (answer generation) material stays out except for one sentence in §9, which keeps
the paper on a single task. That sentence, now checked against `VQA_AG_effectiveness_scores.xlsx`
(human-answer-only): the AG two-model fusion `fused_expan_run1` also trailed its best member
`gemma4_expand_run1` on every @1 metric (METEOR 0.2115 vs 0.2140, STS 0.4019 vs 0.4228, BERTScore
0.8827 vs 0.8860).

## Still to supply

- Author names and affiliations (placeholders in the notebook paper too).
- Verified citations for the debate and judge literature in §2, and the two incomplete TREC 2025
  references.
- Per-split nested-CV numbers for Fig 3, which live with the run caches on `E:\TREC_2026`, not in this
  archive.
- **Other teams' notebook papers (ISU, NII_UIT, BiTeM, g37/g38):** to confirm which runs are fusions,
  what their members are, and whether `g37q36_mc_text` is text-only and the `q38` runs are Qwen3.8-based
  (§7.4).
- **2026 MC ground truth, if NIST releases it:** enables paired tests on the official set (§5.4) and is
  the cleanest place to score debate later (§10).
