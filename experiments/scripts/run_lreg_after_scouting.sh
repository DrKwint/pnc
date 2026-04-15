#!/bin/bash
# Gate script: wait for run_evidential_mcd_scouting.sh to finish, then launch
# the PJSVD lambda_reg sweep. Chains after the scouting gate so the overall
# queue is: low-proj -> scouting -> lreg sweep. One GPU job at a time.

set -u
GATE_LOG=/home/elean/pnc/experiments/logs/lreg_gate.log
mkdir -p "$(dirname "$GATE_LOG")"

echo "[$(date)] gate: waiting for run_evidential_mcd_scouting.sh to finish..." >> "$GATE_LOG"
# Also guard against still-running predecessors in case the scouting hasn't started yet.
while pgrep -f 'run_pjsvd_low_proj\.sh|run_evidential_mcd_scouting\.sh' > /dev/null; do
    sleep 30
done
echo "[$(date)] gate: scouting sweep done; launching ridge sweep." >> "$GATE_LOG"

bash /home/elean/pnc/experiments/scripts/run_pjsvd_lreg_sweep.sh

echo "[$(date)] gate: ridge sweep returned." >> "$GATE_LOG"
