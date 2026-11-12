#!/usr/bin/env bash
# Wait for the server, then prove it can actually GENERATE before anything long
# is launched. /v1/models returning 200 is not proof: the FlashInfer JIT failure
# only fires on the first real prompt, and the client treats a dead server as
# "unprocessable", falling back to the official option order on every query and
# exiting 0 as if the run had succeeded.
set -u

UNIT=qwen38count-vllm
LOG=/home/aakash/qwen38.count.server.log
URL=http://127.0.0.1:8011

until curl -s -m 5 "$URL/v1/models" > /dev/null 2>&1; do
  if [ "$(systemctl --user is-active $UNIT)" != "active" ]; then
    echo "UNIT DIED before serving"
    tail -12 "$LOG"
    exit 1
  fi
  sleep 5
done

echo "HTTP up. Backend line:"
grep -aE "Using .* backend|attention backend" "$LOG" | tail -1 | cut -c1-150

echo "=== GATE: one real generation (first request may JIT-compile, be patient) ==="
body='{"model":"RedHatAI/Qwen3.8-27B-INT4","messages":[{"role":"user","content":"Reply with only this JSON and nothing else: {\"ranking\": [2, 1, 3, 4]}"}],"max_tokens":40,"temperature":0}'
out=$(curl -s -m 900 "$URL/v1/chat/completions" -H 'Content-Type: application/json' -d "$body")

echo "$out" | head -c 700
echo

if echo "$out" | grep -q '"error"'; then
  echo "GATE FAILED"
  grep -aE "RuntimeError|Error:" "$LOG" | tail -3 | cut -c1-160
  exit 2
fi

echo "GATE PASSED -- the model generates"
