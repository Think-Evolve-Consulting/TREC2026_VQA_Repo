#!/usr/bin/env bash
# Serve Qwen3.8-27B INT4 for the counting-aware Task 2 pass.
#
# Runs vLLM in the FOREGROUND on purpose. Detaching it with setsid/nohup/disown
# does not survive here: when the launching `wsl.exe ...` invocation exits, WSL2
# tears the session down and the server dies before it writes its first log
# line. Keeping it in the foreground of a long-lived invocation holds both the
# session and the distro open for as long as the server runs.
#
# --max-model-len 24576, not the baseline's 16384: a 12,288-token counting
# video leaves only ~3,840 tokens for the prompt at 16k, so a transcript-bearing
# counting query 400s and the ladder halves its budget to 6,144 -- silently
# undoing the whole point of the variant.
set -u

LOG=/home/aakash/qwen38.count.server.log

systemctl --user stop cosmos-reason2 gemma4-vllm 2>/dev/null || true
pkill -f '[v]llm serve' 2>/dev/null || true
sleep 3

cd /home/aakash/gemma4 || exit 1

export VLLM_USE_V2_MODEL_RUNNER=0   # WSL2's GPU-PV exposes no UVA
export VLLM_USE_FLASHINFER_SAMPLER=0 # no nvcc here for its JIT sampler

exec ./.venv/bin/vllm serve RedHatAI/Qwen3.8-27B-INT4 \
  --host 0.0.0.0 --port 8011 \
  --served-model-name RedHatAI/Qwen3.8-27B-INT4 \
  --allowed-local-media-path /mnt/e/TREC_2026 \
  --max-model-len 24576 \
  --gpu-memory-utilization 0.93 \
  --enforce-eager \
  --skip-mm-profiling \
  --media-io-kwargs '{"video": {"num_frames": -1}}' \
  > "$LOG" 2>&1
