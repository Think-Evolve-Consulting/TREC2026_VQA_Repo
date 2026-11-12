# Reconciled architecture — Task 1, one RTX 3090

**15 August 2026.** Reconciles `SESSION.2026-08-13.query-routing.md` (accurate,
partly stale) with `Multi_Agent_VQA_Session_Doc.md` (aspirational, largely
unbuildable on this hardware), against what the pipeline actually is on disk and
what the run caches actually measure.

Written to settle one question: **how to cut the fallback and truncation slots in
the Qwen3-VL and Cosmos runs using the query variations already on disk**, inside
a single-GPU serial budget, before the **17 August 2026** deadline.

---

## 1. The constraint that decides everything

**One RTX 3090. 22.72 GiB usable** (Windows permanently holds ~1.3 GiB). Every
model in this pipeline takes essentially the whole card:

| Model | Where | Weights | Port |
|---|---|---|---|
| Gemma 4 12B (QAT INT4) | WSL2 `/home/aakash/gemma4/.venv` | ~7.5 GB + KV | 8010 |
| Qwen3-VL-8B-Thinking (bf16) | WSL2 `/home/aakash/cosmos-reason2/.venv` | 16.64 GB | 8009 |
| Cosmos-Reason2-2B | same venv, systemd unit | ~16 GB resident | 8008 |

**Consequence: there is no concurrency between models, only a schedule.** Two of
these cannot co-reside. Every "agent" in `Multi_Agent_VQA_Session_Doc.md` that is
drawn side-by-side with another agent is, on this box, a *sequential pass over
the whole dataset* with a server teardown and reload between them.

This single fact invalidates most of that document's architecture and rewrites
its execution plan — see §2 and §7.

---

## 2. Reconciliation: the two documents against reality

### 2.1 `Multi_Agent_VQA_Session_Doc.md` — what survives

| It proposes | Reality on disk | Verdict |
|---|---|---|
| **Orchestrator Agent** routing sub-tasks at query time | `VQA/route.query.inputs.py` is an **offline batch stage**, already run; 2,443 routes sit in `query.routing.task1.json` | **Keep the idea, drop the agent.** Routing is a precomputed table, not a live process. Zero GPU cost at answer time. |
| **Vision / Audio / Context agents in parallel** | One model resident at a time | **Impossible as drawn.** The only thing that ever ran "six agents" is the router — and those six are *text-only judgements about the question string*, never about the video. |
| **Debate & Consensus Engine**, triggered at confidence < 0.85 | Routing doc §7 measured **96.9 % of confidences at exactly 1.0**, none below 0.7 | **Dead trigger — delete.** At temperature 0 the model asserts, it does not calibrate. A gate on 0.85 fires for ~0 % of queries. |
| Debate resolves **audio-vs-OCR conflicts** | No OCR stage exists (still true, §2.2) | **Moot.** There is no second signal to conflict with. |
| **10-Variation Generator** producing 10 *output formats* (JSON schemas, chronological breakdowns) | The 10 outputs must be 10 ranked **answers**; `validate.ag.run.py` rejects duplicates and rejects format noise | **Direction reversed.** Variation belongs on the **input** side — and that is exactly the fix for §5. |
| **Phase 1:** run 168 audio + 10 OCR queries first | No OCR stage; the audio-only queries are already answered in all three runs | **Unrunnable as written.** |
| **Asymmetric resource allocation**, lazy-load OCR | Nothing can be partitioned — only scheduled | **Reframe as a serial schedule** (§7). |
| Dataset "2,411+ videos" | 2,443 queries over **1,956** videos | Corrected. |

**What does survive, and is worth keeping:** the *pooling* instinct — that
several independent looks at one query beat one look. That is the one idea in
that document which is already proven on this hardware (§4), and it is the basis
of the repair design in §6.

### 2.2 `SESSION.2026-08-13.query-routing.md` — what went stale

The routing analysis itself is sound and its output is in production. Three
premises have since changed:

| Claim | Status |
|---|---|
| "868 undownloaded YouTube videos still block 35.5 % of the *answers*" | **Stale — closed.** The Qwen3-VL cache shows `video_exists` true for **2,427 / 2,443**; only **16** queries genuinely lack media. |
| Next step 1: "Wire `Required_Inputs` into the answering pass — not yet connected" | **Done, for one of three.** `generate.answers.qwen3vl.py` consumes `--routing` via `plan_for()`. Gemma 4 and Cosmos still ignore routing entirely. |
| Next step 2: "Build the OCR stage — 57 queries need it" | **Still open.** No OCR anywhere. 57 queries (2.3 %), 16 of them with no viable path at all. |
| Next step 5: "Validate routing against 2025 ground truth if the GT videos ever land" | **Done** — `VQA/EVALUATION.2025.md`, 150-query sample. |

Its §6 finding — *concurrency pays when the prompt prefix is cheap to recompute*
— is still the governing rule for worker counts and is unchanged.

---

## 3. What the pipeline actually is

Three answering passes over the same inputs. **Each has half the machinery, and
none has both halves:**

| Script | Model | Uses routing | Uses variations | Calls / query | Fallback rate |
|---|---|---|---|---|---|
| `generate.answers.py` | Gemma 4 | ✗ | **✓ 10 prompts × k=3** | **10** | **0.8 %** |
| `generate.answers.qwen3vl.py` | Qwen3-VL | **✓** | ✗ | **1** (k=10) | 8.6 % |
| `generate.answers.cosmos.py` | Cosmos | ✗ | ✗ | **1** (k=10) | **74 %** |

The fallback rate tracks the **architecture**, not the model. That is the central
finding of this reconciliation.

---

## 4. The measured error profile

Computed directly from the 2026 run caches (`answers.task1.*.cache.jsonl`), not
estimated.

### Qwen3-VL — 210 / 2,443 queries (8.6 %) carry a fallback; 1,596 slots (6.5 %)

| Mode | Queries | What happened |
|---|---|---|
| **Truncation** | **62 (2.5 %)** | `finish_reason = "length"`. 60 of the 62 yielded **no parseable JSON at all** → 10/10 fallback = **600 dead slots**. **59 of those 60 had evidence available** — recoverable, not blind. |
| **Refusal collapse** | 82 (3.4 %) | Model returned 10 answers that were all `"unknown"`; dedupe leaves 1, fill_slots pads 9. |
| **Genuine shortfall** | 68 (2.8 %) | 1–9 real distinct answers, short of 10. e.g. Q66 → `['0']`, Q1147 → `['barbell bench press', 'unknown']`. |

**Duplicate collapse costs 879 slots across 137 queries** — the model emits ten
answers that normalise to one or two.

**The truncation is never retried.** `Qwen3VLClient.answer()` returns on the
first successful HTTP response and never inspects `finish_reason`; the degrade
ladder catches `BadRequestError` and `APIStatusError` only. A truncated-but-200
response flows straight through `extract_answers()` → `[]` → ten fallbacks. This
is a gap, not a design choice.

**Why it truncates:** `reasoning_chars` is **0 on all 2,443 rows** while median
`completion_tokens` is **1,447** for ten answers of ≤40 words. The reasoning
parser is not splitting the think block — the thinking is being spent *inside*
the completion budget, and the k=10 JSON has to fit in what remains of 8,192
tokens. 62 queries ran out before reaching it.

### Cosmos — 1,808 / 2,443 queries (74 %) carry a fallback; 11,289 slots (46.2 %)

| Mode | Queries | What happened |
|---|---|---|
| **No JSON at all** | **457 (18.7 %)** | 10/10 fallback = 4,570 dead slots. **348 had evidence.** A 2B model cannot hold a ten-item JSON schema. |
| **Genuine shortfall** | 1,196 (66.2 %) | 1–9 distinct real answers. |
| **Refusal collapse** | 155 (8.6 %) | |

Orthogonal and already known: **1,097 queries (45 %) were answered with no video**
— the `Video_ID`-is-not-a-filename bug, fixed 14 Aug, run dated 13 Aug. 469
queries had neither video nor transcript.

### Gemma 4 — the control

10 prompts × 3 candidates = 30 pooled candidates per query, `n_prompt_failures`
**0 across all 2,443**. Result: **19 queries (0.8 %) with any fallback, 24 slots
total** (0.01 / 10). Same fallback pool, same validator, same dataset.

---

## 5. Why query variations are the right fix

Gemma 4 is the existence proof: same 2,443 queries, same fallback pool, **24 dead
slots against Qwen3-VL's 1,596 and Cosmos's 11,289.** Two mechanisms, and *both*
matter:

**1. Independent draws.** Ten separate calls fail independently. A truncation or
a JSON malformation kills one call out of ten instead of the whole query. This
directly addresses the 60 + 457 total-loss queries — the ones where a single
call's failure took all ten slots with it.

**2. Short JSON per call.** Gemma 4 asks each prompt for **k=3**, not k=10. A
3-item list is reached long before the token budget runs out; a 10-item list is
what Qwen3-VL is truncating and what Cosmos cannot format at all. **4 calls × k=3
= 12 candidates > the 10 needed**, at a fraction of the per-call token pressure.

Asking for *fewer answers per call, more calls* is therefore both the truncation
fix and the fallback fix.

**And the input artifact is already on disk, free:** `query.variations.task1.json`
holds **exactly 10 variations for all 2,443 queries**. It is text-only, needs no
GPU, and is already validated by the Gemma 4 run.

### What variations will *not* fix

The 82 Qwen3-VL and 155 Cosmos **all-refusal** queries are the model asserting
`"unknown"` — a semantic failure, not a format one. Re-asking in different words
recovers some, but expect partial yield. Budget them last, and do not count them
as recovered until measured. The 16 zero-evidence queries are unrecoverable by
any prompting.

---

## 6. The design: a `--repair` second pass

Not a rewrite of either script — a second pass over the **existing cache**, so
the 2,233 clean Qwen3-VL queries are never re-inferred.

```
--repair                  only process queries whose cached fallback > 0
--variations PATH         default query.variations.task1.json
--repair-candidates 3     k per call (NOT 10) -- the truncation fix
--max-extra-calls 4       ceiling per query
--stop-when-full          stop calling once 10 distinct answers exist
```

Behaviour per selected query:

1. Load the cached row; keep every **model-sourced** answer already in it.
2. Issue up to `--max-extra-calls` calls, one per variation, each asking k=3,
   reusing the same video URI and sampling plan.
3. Pool old + new through Gemma 4's `pool_candidates()` — position-weighted
   `(1.0, 0.5, 0.25)` × confidence, refusals sorted last.
4. `fill_slots()` as today. Fallbacks only cover what pooling could not reach.
5. Write to a **new cache and a new run file**; never edit in place.

**Three cheap additions worth making in the same pass:**

- **Retry on `finish_reason == "length"`** at k=3 before anything else. 62
  queries, one call each — recovers up to 600 slots for ~17 min of GPU.
- **Do not send a k=10 request to Cosmos again.** k=3 should be its default, not
  just its repair mode.
- Keep the video and plan identical between the original call and the repair
  calls, so vLLM's multimodal-embedding and prefix cache serve the 2nd–4th call
  far cheaper than the 1st. Cost figures below are therefore a **ceiling**.

---

## 7. Serial schedule to 17 August

**Order the work by model, not by pipeline stage.** Every swap costs a teardown
and a reload, and `ensure_backend()` is the only code permitted to do it. This is
the direct opposite of the Multi_Agent doc's Phase 1/2/3 plan, which slices by
query type and would force repeated swaps across the same dataset.

Measured rates: Qwen3-VL **16.6 s/query** at 4 workers; Cosmos **13.3 s/query**
at 2 workers; Gemma 4 expansion 68–76 min for 2,443.

| # | Block | Model | Work | Cost | Recovers |
|---|---|---|---|---|---|
| 1 | **Qwen3-VL resident** | Qwen3-VL | truncation retry, 62 queries × 1 call @ k=3 | **~0.3 h** | up to **600 slots** |
| 2 | ″ | Qwen3-VL | variation repair, 148 queries × ≤4 calls | **~2.4 h** | most of the remaining ~996 |
| 3 | ″ | Qwen3-VL | *(optional)* 82 refusal queries × ≤4 calls | ~1.5 h | partial |
| — | **swap** | | teardown + load | ~0.2 h | |
| 4 | **Cosmos resident** | Cosmos | full re-run — video fix + k=3 + inline repair | **~9–12 h** | the 45 % blind **and** the 46 % dead slots |
| — | **swap** | | | ~0.2 h | |
| 5 | **Gemma 4 resident** | Gemma 4 | re-expand the repaired runs, `expand.answers.py` | ~2.5 h | — |
| 6 | CPU only | — | `validate.ag.run.py` × N | <0.1 h | — |

**Blocks 1–2 are the whole argument: ~2.7 GPU-hours to clear ~89 % of Qwen3-VL's
dead slots.** Do them first regardless of what else fits.

**On Cosmos — the honest read.** It scored METEOR 0.033 against Gemma 4's 0.148,
and `VQA/EVALUATION.2025.md` already attributes most of that gap to degradation
artifact rather than comprehension. Block 4 is the single largest GPU cost here
and it is being spent on the weakest of three runs, which is **not currently a
submission candidate**. If the 48 hours get tight, **cut block 4, not blocks
1–2.** A repaired Qwen3-VL is worth more than a repaired Cosmos.

**Sequencing note:** blocks 1–3 change `qwen3vl.run.task1.json`, so
`qwen3vl.expanded.run.task1.json` must be regenerated in block 5 — the existing
expansion is derived from the unrepaired run. Keep the current four validated
candidates on disk untouched as fallbacks until the repaired ones validate.

**Two environment traps that kill long unattended runs** (both already paid for
once): keep **one interactive WSL2 window open** for the duration — WSL2 tears
the distro down when the last session exits — and start servers only through
`launch_detached()` with `setsid nohup … < /dev/null`.

---

## 8. What to delete from `Multi_Agent_VQA_Session_Doc.md`

So a future session does not re-derive this:

1. **The Debate & Consensus Engine, entirely.** Its confidence trigger was
   measured dead before that document was written. If graded uncertainty is ever
   wanted, the honest source is **disagreement across the variation ensemble** —
   which the repair pass in §6 produces for free as `votes` in the pooled
   cluster. That is a real signal; a self-reported 1.0 is not.
2. **The parallel-agent topology.** One GPU, one model, a schedule.
3. **The 10-variation *output* generator.** The ten outputs are ranked answers
   the validator checks for duplicates, not ten formats.
4. **The Phase 1/2/3 execution plan.** Replaced by §7, ordered by resident model.

**Still open, unchanged from the routing doc:** the OCR stage. 57 queries need
it, 16 have no other path, nothing provides it. It remains the one genuine
capability gap — but at 2.3 % of the set and under 48 hours, it is not the place
to spend the GPU.
