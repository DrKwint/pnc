#!/bin/bash
# Gate: wait for all upstream sweeps to finish, then launch the Evidential ESN
# (predictive-NLL early-stop) sweep. Chain position: #4 of 4.

set -u
GATE_LOG=/home/elean/pnc/experiments/logs/esn_gate.log
mkdir -p "$(dirname "$GATE_LOG")"

echo "[$(date)] gate: waiting for all upstream sweeps to finish..." >> "$GATE_LOG"
while pgrep -f 'run_pjsvd_low_proj\.sh|run_evidential_mcd_scouting\.sh|run_pjsvd_lreg_sweep\.sh' > /dev/null; do
    sleep 30
done
echo "[$(date)] gate: upstream sweeps done; launching ESN sweep." >> "$GATE_LOG"

bash /home/elean/pnc/experiments/scripts/run_evidential_esn_sweep.sh

echo "[$(date)] gate: ESN sweep returned." >> "$GATE_LOG"
