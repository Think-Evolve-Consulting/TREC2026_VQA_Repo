#!/usr/bin/env bash
# Can we serve with Triton attention (no JIT at all) and a non-FP8 KV cache?
set -u
systemctl --user stop qwen38count-vllm 2>/dev/null
systemctl --user stop qwen38-triton-test 2>/dev/null
systemctl --user reset-failed qwen38-triton-test 2>/dev/null
sleep 3
LOG=/home/aakash/qwen38.triton.test.log
rm -f "$LOG"

systemd-run --user --unit=qwen38-triton-test --collect \
  --setenv=VLLM_USE_V2_MODEL_RUNNER=0 \
  --setenv=VLLM_USE_FLASHINFER_SAMPLER=0 \
  --property=StandardOutput=append:"$LOG" \
  --property=StandardError=append:"$LOG" \
  --working-directory=/home/aakash/gemma4 \
  /home/aakash/gemma4/.venv/bin/vllm serve RedHatAI/Qwen3.8-27B-INT4 \
    --host 0.0.0.0 --port 8011 --served-model-name RedHatAI/Qwen3.8-27B-INT4 \
    --allowed-local-media-path /mnt/e/TREC_2026 \
    --max-model-len 24576 --gpu-memory-utilization 0.93 \
    --enforce-eager --skip-mm-profiling \
    --attention-backend TRITON_ATTN \
    --kv-cache-dtype auto \
    --media-io-kwargs '{"video": {"num_frames": -1}}'

until curl -s -m 5 http://127.0.0.1:8011/v1/models > /dev/null 2>&1; do
  if [ "$(systemctl --user is-active qwen38-triton-test)" != "active" ]; then
    echo "DIED AT INIT"
    grep -aE "ValueError|RuntimeError" "$LOG" | tail -3 | cut -c1-170
    exit 1
  fi
  sleep 5
done

echo "HTTP up"
grep -aE "Using .* backend|KV cache size|Maximum concurrency" "$LOG" | tail -3 | cut -c1-150
echo "=== generation test ==="
body='{"model":"RedHatAI/Qwen3.8-27B-INT4","messages":[{"role":"user","content":"Say OK"}],"max_tokens":8,"temperature":0}'
out=$(curl -s -m 600 http://127.0.0.1:8011/v1/chat/completions -H 'Content-Type: application/json' -d "$body")
echo "$out" | head -c 400; echo
echo "$out" | grep -q '"error"' && { echo "GENERATION FAILED"; grep -aE "RuntimeError|ValueError" "$LOG" | tail -3 | cut -c1-170; exit 2; }
echo "TRITON PATH WORKS"
