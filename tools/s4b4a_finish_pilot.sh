#!/usr/bin/env bash
# S4B.4A pilot finish chain: wait for the training grid, then E1 -> E2 ->
# reports -> verify.  Every step writes a log under reports/s4b4a_run_logs/.
#
# Nothing here changes the frozen protocol: the config is passed through and
# E1/E2 read only artefacts the grid produced.
set -u
cd /fj/Agro-JEPA || exit 1
CFG=agro_jepa/config/s4b4a_year_heldout.yaml
LOGS=reports/s4b4a_run_logs
GRID_PID_FILE=$LOGS/pilot1_grid.pid
mkdir -p "$LOGS"

if [[ -f "$GRID_PID_FILE" ]]; then
  GRID_PID=$(cat "$GRID_PID_FILE")
else
  GRID_PID=""
fi

echo "[chain] $(date -Is) waiting for grid pid=${GRID_PID:-unknown}"
while [[ -n "$GRID_PID" ]] && kill -0 "$GRID_PID" 2>/dev/null; do
  sleep 60
done
echo "[chain] $(date -Is) grid finished"

CUDA_VISIBLE_DEVICES=1 python3 tools/s4b4a_eval_e1.py --config "$CFG" --seeds 0 \
  > "$LOGS/pilot1_e1.log" 2>&1
echo "[chain] $(date -Is) E1 rc=$?"

CUDA_VISIBLE_DEVICES=1 python3 tools/s4b4a_eval_e2.py --config "$CFG" --seeds 0 \
  > "$LOGS/pilot1_e2.log" 2>&1
echo "[chain] $(date -Is) E2 rc=$?"

python3 tools/s4b4a_write_reports.py --config "$CFG" \
  > "$LOGS/pilot1_reports.log" 2>&1
echo "[chain] $(date -Is) reports rc=$?"

python3 tools/s4b4a_verify.py --config "$CFG" --seeds 0 \
  > "$LOGS/pilot1_verify.log" 2>&1
echo "[chain] $(date -Is) verify rc=$?"
echo "[chain] all steps finished"
