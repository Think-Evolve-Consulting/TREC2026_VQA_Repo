#!/usr/bin/env bash
# Block until the server answers, or it stops making progress.
#
# Liveness is judged by the HTTP endpoint and by the log still growing --
# deliberately NOT by pgrep. In this distro `pgrep -f vllm` returns nothing
# even while `ps` clearly shows `vllm serve` and `VLLM::EngineCore` running, so
# a pgrep-based death check reports a healthy server as dead.
set -u

LOG=/home/aakash/qwen38.count.server.log
STALL=40          # consecutive 10 s polls with no log growth before giving up
last_size=-1
stalled=0

for i in $(seq 1 360); do
  if curl -s -m 5 http://127.0.0.1:8011/v1/models > /dev/null 2>&1; then
    echo "SERVER READY after $((i * 10))s"
    grep -aiE 'KV cache size|maximum concurrency|Model loading took' "$LOG" | tail -3
    exit 0
  fi

  size=$(stat -c %s "$LOG" 2>/dev/null || echo 0)

  if [ "$size" = "$last_size" ]; then
    stalled=$((stalled + 1))
  else
    stalled=0
    last_size=$size
  fi

  if [ "$stalled" -ge "$STALL" ]; then
    echo "SERVER STALLED: log has not grown for $((STALL * 10))s"
    tail -25 "$LOG"
    exit 1
  fi

  sleep 10
done

echo "TIMEOUT after 60 min"
tail -20 "$LOG"
exit 2
