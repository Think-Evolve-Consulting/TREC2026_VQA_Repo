# TREC 2026 VQA: open-weight VLM pipelines on a single GPU

Scripts and working notes for our TREC 2026 Video Question Answering submissions:
Task 1 (answer generation, ten ranked answers) and Task 2 (multiple-choice ranking).
Every model ran locally on one RTX 3090, served through vLLM, llama.cpp or Ollama.

## Layout

| Path | Contents |
|---|---|
| `VQA/` | The pipeline: ASR (`transcribe.*`), frame extraction, query routing and paraphrases, Task 1 answerers (`generate.answers.*`), expansion, fusion and judge reranking, Task 2 rankers (`rank.options.mc.*`), the multi-agent debate (`agent.debate.mc.py`, `debate.rank.mc.py`), the 2025 dev harness and the orchestrators (`run.ensemble.py`, `run.eval.2025.py`) |
| `VQA/diagnostics/` | Throughput and error-analysis scripts |
| `VQA/*.sh`, `*.ps1`, `*.service` | Serving, run and diagnostic wrappers |
| `colab/` | Standalone Colab versions of the Task 2 rankers |
| `notebooks/` | Colab serving notebooks (superseded) |
| Root `*.py` | Test-video download and retry scripts |
| Root `*.md` | Session notes, plans, runbooks, the model scorecard and the notebook paper (`TREC2026_VQA_NOTEBOOK_PAPER.md`) |

## Third-party files

These files come from the NIST TREC VQA track and keep their original terms:
`VQA/eval.vqa.ag.py`, `VQA/eval.vqa.mc.py`, `VQA/validate.ag.run.py`, `VQA/calc.sts.pl`,
`VQA/vqa_eval_readme.txt`, `VQA/vqa.gt.csv`, `VQA/README.md` and `validate_vqa_mc_2026.py`.

## Not included

Videos, transcripts, model weights, caches, run files and scores are not in this repo. The TREC data
agreement covers them. Scripts default to the original local paths (for example `E:\TREC_2026` or
`/mnt/e/TREC_2026`); override them with the script's flags where it has them, or edit them before running.
