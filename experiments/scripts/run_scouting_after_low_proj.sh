#!/bin/bash
# Gate script: wait for run_pjsvd_low_proj.sh to finish, then launch the
# evidential + MC Dropout scouting sweep. Ensures we honor one-GPU-job-at-a-time.
#
# Polls with pgrep every 30s (cheap; no GPU involvement).

set -u
GATE_LOG=/home/elean/pnc/experiments/logs/scouting_gate.log
mkdir -p "$(dirname "$GATE_LOG")"

echo "[$(date)] gate: waiting for run_pjsvd_low_proj.sh to finish..." >> "$GATE_LOG"
while pgrep -f run_pjsvd_low_proj.sh > /dev/null; do
    sleep 30
done
echo "[$(date)] gate: low-proj sweep done; launching scouting sweep." >> "$GATE_LOG"

bash /home/elean/pnc/experiments/scripts/run_evidential_mcd_scouting.sh

echo "[$(date)] gate: scouting sweep returned." >> "$GATE_LOG"
