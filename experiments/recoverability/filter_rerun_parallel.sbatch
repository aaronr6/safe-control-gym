#!/bin/bash
#SBATCH --job-name=rerun_parallel
#SBATCH --partition=cm4_inter
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=experiments/recoverability/slurm_logs/rerun_parallel_%j.out
#SBATCH --error=experiments/recoverability/slurm_logs/rerun_parallel_%j.err

set -euo pipefail

REPO_ROOT="/dss/dsshome1/0A/go39bur2/safe-control-gym"
cd "$REPO_ROOT"

source /dss/dsshome1/0A/go39bur2/miniconda3/etc/profile.d/conda.sh
conda activate safe

export ACADOS_SOURCE_DIR=/dss/dsshome1/0A/go39bur2/acados
export LD_LIBRARY_PATH=/dss/dsshome1/0A/go39bur2/acados/lib:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

mkdir -p experiments/recoverability/slurm_logs

python experiments/recoverability/run_manifest_local_parallel.py \
  --manifest experiments/recoverability/sweeps_phase2/final_v112_filter_rerun_manifest.csv \
  --repo_root "$REPO_ROOT" \
  --workers "${FILTER_RERUN_WORKERS:-8}" \
  --log_dir experiments/recoverability/ppo_runs/slurm_final_v112_filter_rerun/parallel_logs \
  --status_csv experiments/recoverability/ppo_runs/slurm_final_v112_filter_rerun/parallel_status_v2.csv
