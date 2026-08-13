#!/bin/bash
#SBATCH --job-name=recov_sweep
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=112
#SBATCH --mem=224G
#SBATCH --output=experiments/recoverability/slurm_logs/recov_%A_%a.out
#SBATCH --error=experiments/recoverability/slurm_logs/recov_%A_%a.err

set -euo pipefail

if [[ -z "${MANIFEST:-}" ]]; then
  echo "MANIFEST env var is required"
  exit 1
fi

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "SLURM_ARRAY_TASK_ID is missing"
  exit 1
fi

REPO_ROOT="/dss/dsshome1/0A/go39bur2/safe-control-gym"
cd "$REPO_ROOT"

source /dss/dsshome1/0A/go39bur2/miniconda3/etc/profile.d/conda.sh
conda activate safe

export ACADOS_SOURCE_DIR=/dss/dsshome1/0A/go39bur2/acados
export LD_LIBRARY_PATH=/dss/dsshome1/0A/go39bur2/acados/lib:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

mkdir -p experiments/recoverability/slurm_logs

completed_run=$(/dss/dsshome1/0A/go39bur2/miniconda3/envs/safe/bin/python - <<'PY'
import csv
import os
from pathlib import Path

manifest = os.environ['MANIFEST']
idx = int(os.environ['SLURM_ARRAY_TASK_ID'])

with open(manifest, 'r', newline='') as f:
    rows = list(csv.DictReader(f))

row = rows[idx]
summary_csv = Path(row.get('summary_csv', ''))
run_name = str(row.get('run_name', ''))

if run_name and summary_csv.exists():
    with summary_csv.open('r', newline='') as f:
        if any(existing.get('run_name') == run_name for existing in csv.DictReader(f)):
            print('1')
            raise SystemExit(0)

print('0')
PY
)

if [[ "$completed_run" == "1" ]]; then
  echo "Skipping completed rerun task ${SLURM_ARRAY_TASK_ID}."
  exit 0
fi

python - <<'PY'
import csv
import os
import subprocess
import sys
from pathlib import Path

manifest = os.environ['MANIFEST']
idx = int(os.environ['SLURM_ARRAY_TASK_ID'])

with open(manifest, 'r', newline='') as f:
    rows = list(csv.DictReader(f))

if idx < 0 or idx >= len(rows):
    raise IndexError(f'SLURM_ARRAY_TASK_ID={idx} out of bounds for manifest size {len(rows)}')

row = rows[idx]

def opt(flag, value):
    if value is None or value == '':
        return []
    return [flag, str(value)]

cmd = [
    sys.executable,
    'experiments/recoverability/train_ppo_adversary.py',
    '--r', str(row['r']),
    '--reward_mode', str(row['reward_mode']),
    '--seed', str(row['seed']),
    '--max_env_steps', str(row['max_env_steps']),
    '--env_horizon', str(row['env_horizon']),
    '--dt', str(row['dt']),
    '--umax', str(row['umax']),
    '--x_limit', str(row['x_limit']),
    '--mpsc_horizon', str(row['mpsc_horizon']),
    '--mpsc_cost_horizon', str(row['mpsc_cost_horizon']),
    '--init_scale', str(row['init_scale']),
    '--output_root', str(row['output_root']),
    '--summary_csv', str(row['summary_csv']),
    '--run_name', str(row['run_name']),
]

if row.get('eval_episodes', '') not in ('', None):
    cmd += ['--eval_episodes', str(row['eval_episodes'])]

if row.get('num_workers', '') not in ('', None):
    cmd += ['--num_workers', str(row['num_workers'])]

cmd += opt('--train_mpsc_horizon', row.get('train_mpsc_horizon', ''))
cmd += opt('--eval_mpsc_horizon', row.get('eval_mpsc_horizon', ''))
cmd += opt('--train_mpsc_cost_horizon', row.get('train_mpsc_cost_horizon', ''))
cmd += opt('--train_sf_cost_function', row.get('train_sf_cost_function', ''))
cmd += opt('--eval_sf_cost_function', row.get('eval_sf_cost_function', ''))
cmd += opt('--checkpoint_path', row.get('checkpoint_path', ''))
cmd += opt('--regularization_weight', row.get('regularization_weight', ''))
cmd += opt('--blended_cost_alpha', row.get('blended_cost_alpha', ''))
cmd += opt('--max_w', row.get('max_w', ''))
cmd += opt('--v_limit', row.get('v_limit', ''))

if str(row.get('eval_from_checkpoint', '')).strip().lower() in ('1', 'true', 'yes', 'y'):
    cmd += ['--eval_from_checkpoint']

print('Launching:', ' '.join(cmd))
subprocess.run(cmd, check=True)
PY
