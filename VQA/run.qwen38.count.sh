#!/usr/bin/env bash
# The counting-aware Qwen3.8 Task 2 pass. Arguments pass straight through, so:
#
#   bash run.qwen38.count.sh --count-only --limit 20     # smoke
#   bash run.qwen38.count.sh                             # full 2,443
#
# Resumable: every finished query is appended to
# rankings.task2.qwen38count.cache.jsonl and re-reading it is the first thing
# the pass does, so re-running the same command continues where it stopped.
set -u

PY=/home/aakash/gemma4/.venv/bin/python3
SCRIPT=/mnt/e/TREC_2026/VQA/rank.options.mc.qwen38.count.py
LOG=/mnt/e/TREC_2026/logs/rank.options.mc.task2.qwen38count.log

mkdir -p /mnt/e/TREC_2026/logs

# Unbuffered, so a watcher tailing the log sees progress as it happens rather
# than in 8 KB bursts.
exec "$PY" -u "$SCRIPT" "$@" 2>&1 | tee -a "$LOG"
