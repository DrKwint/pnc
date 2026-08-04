#!/usr/bin/env bash
# Serial job supervisor for the lambda study.
#
# Runs one job at a time (the GPU must never have two JAX processes on it) from
# results/banking77_distilbert_pnc/lambda_study/queue/*.cmd, in filename order.
# New .cmd files may be dropped in WHILE this is running and will be picked up.
#
# Each job writes <name>.log and, on completion, <name>.done containing its exit code.
# The supervisor exits once the queue has been empty for IDLE_EXIT_SECONDS (default
# 45 min), so it self-terminates overnight instead of spinning forever.

set -u
ROOT="/home/elean/pnc"
Q="$ROOT/results/banking77_distilbert_pnc/lambda_study/queue"
PY="$ROOT/.venv_bank/bin/python"
IDLE_EXIT_SECONDS="${IDLE_EXIT_SECONDS:-2700}"
SUP_LOG="$Q/supervisor.log"

mkdir -p "$Q"
cd "$ROOT" || exit 1
export XLA_PYTHON_CLIENT_PREALLOCATE=false

say() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "$SUP_LOG"; }

# PID file, not pgrep: a pgrep pattern matching this script's path also matches the
# command line of whatever shell is doing the checking, which silently reports a dead
# supervisor as alive.
PIDFILE="$Q/supervisor.pid"
if [ -e "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE" 2>/dev/null)" 2>/dev/null; then
  say "supervisor already running (pid $(cat "$PIDFILE")) — exiting"; exit 0
fi
echo $$ > "$PIDFILE"
trap 'rm -f "$PIDFILE"; say "supervisor stop (pid $$)"' EXIT

say "supervisor start (pid $$, idle-exit ${IDLE_EXIT_SECONDS}s)"
idle=0
while true; do
  next=""
  for f in "$Q"/*.cmd; do
    [ -e "$f" ] || continue
    [ -e "${f%.cmd}.done" ] && continue
    next="$f"; break
  done

  if [ -z "$next" ]; then
    if [ -e "$Q/NO_MORE" ]; then say "queue drained and NO_MORE set — exiting"; break; fi
    idle=$((idle + 10))
    if [ "$idle" -ge "$IDLE_EXIT_SECONDS" ]; then
      say "queue idle for ${idle}s — exiting"; break
    fi
    sleep 10
    continue
  fi

  idle=0
  name="$(basename "${next%.cmd}")"
  say "START $name"
  # shellcheck disable=SC2086
  PY="$PY" bash "$next" > "$Q/$name.log" 2>&1
  rc=$?
  echo "$rc" > "$Q/$name.done"
  say "END   $name rc=$rc"
done
