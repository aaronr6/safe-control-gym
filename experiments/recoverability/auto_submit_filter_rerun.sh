#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/dss/dsshome1/0A/go39bur2/safe-control-gym"
LOG_DIR="$REPO_DIR/experiments/recoverability/ppo_runs/slurm_final_v112_filter_rerun"
LOG_FILE="$LOG_DIR/auto_submit.log"
INTERVAL_SECONDS="${AUTO_SUBMIT_INTERVAL_SECONDS:-1}"
MAX_NEW="${AUTO_SUBMIT_MAX_NEW:-1}"
MAX_SUBMITTED="${AUTO_SUBMIT_MAX_SUBMITTED:-2}"

mkdir -p "$LOG_DIR"
cd "$REPO_DIR"

printf 'auto-submit started at %s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" >> "$LOG_FILE"

while true; do
  echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] submitting check" >> "$LOG_FILE"
  /dss/dsshome1/0A/go39bur2/miniconda3/envs/safe/bin/python \
    experiments/recoverability/submit_filter_rerun_sweep.py \
    --max-new "$MAX_NEW" \
    --max-submitted "$MAX_SUBMITTED" >> "$LOG_FILE" 2>&1 || true
  sleep "$INTERVAL_SECONDS"
done
