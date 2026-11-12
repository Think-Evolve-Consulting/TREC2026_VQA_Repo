#!/bin/bash
# The LLaVA-Video pass over the 400-query 2025 dev sample. Runs inside WSL2
# because the server opens each clip off the /mnt/e path in the request.
# Resumable: queries already in the cache are skipped.
cd /mnt/e/TREC_2026 || exit 1
/home/aakash/cosmos-reason2/.venv/bin/python -u VQA/rank.options.mc.llavavideo.py \
    --queries eval.2025.mc/sample400.seed2026.queries.json \
    --frames videos_GT_2025/frames --videos videos_GT_2025/videos \
    --transcripts videos_GT_2025/transcription_openwhisper --ignore-routing \
    --out eval.2025.mc/llavavideo.dev.run.json \
    --cache eval.2025.mc/llavavideo.dev.cache.jsonl \
    2>&1 | tee -a logs/dev.mc.llavavideo.log
