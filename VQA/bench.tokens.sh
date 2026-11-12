#!/bin/bash
# Run VQA/bench.tokens.py from inside WSL2 against a vLLM model, with the venv
# that matches the server. The video-payload models hand the server a
# /mnt/e/TREC_2026 path it opens itself, so the client has to live on that side
# of the boundary too.
#
#   wsl -d Ubuntu-24.04 -- bash /mnt/e/TREC_2026/VQA/bench.tokens.sh qwen3vl
#   wsl -d Ubuntu-24.04 -- bash /mnt/e/TREC_2026/VQA/bench.tokens.sh qwen38 --concurrency 4
#
# Extra flags are passed through. The server must already be up; start it with
# VQA/serve.vllm.sh in another window.
set -u

MODEL="${1:-}"
shift || true

case "$MODEL" in
cosmos|qwen3vl|llavavideo) VENV=/home/aakash/cosmos-reason2/.venv ;;
gemma4|qwen38)             VENV=/home/aakash/gemma4/.venv ;;
*)
    echo "usage: bench.tokens.sh {cosmos|qwen3vl|llavavideo|gemma4|qwen38} [flags]" >&2
    exit 2
    ;;
esac

cd /mnt/e/TREC_2026 || exit 1
mkdir -p logs
$VENV/bin/python3 -u VQA/bench.tokens.py --model "$MODEL" "$@" 2>&1 |
    tee -a "logs/bench.tokens.$MODEL.log"
