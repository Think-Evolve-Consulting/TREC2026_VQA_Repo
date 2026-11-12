#!/usr/bin/env bash
# Launch the counting pass as a transient systemd user unit.
#
# Why not just run it: a client tied to a `wsl.exe` invocation is killed when a
# signal propagates from the Windows side -- that is what killed the first
# server at 13:18:45, cleanly and without warning. systemd is PID 1 here, so a
# transient unit outlives anything happening in the calling session. The pass
# is resumable from its cache either way, but a 9-hour run should not have to
# rely on that.
#
#   bash launch.qwen38.count.sh smoke     # --count-only --limit 20
#   bash launch.qwen38.count.sh full      # all 2,443
#   bash launch.qwen38.count.sh status    # is it alive, and where is it up to
#   bash launch.qwen38.count.sh stop
set -u

UNIT=qwen38count-run
PY=/home/aakash/gemma4/.venv/bin/python3
SCRIPT=/mnt/e/TREC_2026/VQA/rank.options.mc.qwen38.count.py
LOG=/mnt/e/TREC_2026/logs/rank.options.mc.task2.qwen38count.log
CACHE=/mnt/e/TREC_2026/rankings.task2.qwen38count.cache.jsonl

mode="${1:-status}"

case "$mode" in
  smoke) args=(--count-only --limit 20) ;;
  full)  args=() ;;
  status)
    echo "unit:    $(systemctl --user is-active $UNIT 2>/dev/null)"
    echo "cached:  $(wc -l < "$CACHE" 2>/dev/null || echo 0) queries"
    echo "log:"
    tail -6 "$LOG" 2>/dev/null
    exit 0
    ;;
  stop)
    systemctl --user stop "$UNIT" 2>/dev/null
    echo "stopped"
    exit 0
    ;;
  *) echo "usage: $0 {smoke|full|status|stop}"; exit 2 ;;
esac

mkdir -p /mnt/e/TREC_2026/logs
systemctl --user reset-failed "$UNIT" 2>/dev/null || true

# -u so a watcher tailing the log sees progress per query rather than in bursts.
systemd-run --user --unit="$UNIT" --same-dir --collect \
  --property=StandardOutput="append:$LOG" \
  --property=StandardError="append:$LOG" \
  "$PY" -u "$SCRIPT" "${args[@]}"

echo "launched $UNIT ($mode); log: $LOG"
