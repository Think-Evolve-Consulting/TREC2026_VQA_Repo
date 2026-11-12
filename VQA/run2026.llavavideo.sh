#!/bin/bash
# The LLaVA-Video pass over the 2,443 released 2026 MC queries. Runs inside
# WSL2 because the server opens each clip off the /mnt/e path in the request.
# Routing is on, as in every other 2026 pass. Resumable: queries already in
# the cache are skipped. Pass extra flags through, e.g. --videos <dir>.
cd /mnt/e/TREC_2026 || exit 1
/home/aakash/cosmos-reason2/.venv/bin/python -u VQA/rank.options.mc.llavavideo.py \
    --out llavavideo.run.task2.json \
    --cache rankings.task2.llavavideo.cache.jsonl \
    "$@" 2>&1 | tee -a logs/rank.options.mc.task2.llavavideo.log
