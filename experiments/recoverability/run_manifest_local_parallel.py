#!/usr/bin/env python3
"""Run recoverability manifest entries locally with parallel workers.

This runner is intended for long unattended sessions (e.g., overnight) when
cluster submission is unavailable. It skips runs already present in the
summary CSV and executes the remaining entries with a worker pool.
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path
from typing import Dict, Iterable, List


def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open('r', newline='') as f:
        return list(csv.DictReader(f))


def _completed_run_names(summary_csv: Path) -> set[str]:
    if not summary_csv.exists():
        return set()
    rows = _read_csv_rows(summary_csv)
    return {r.get('run_name', '') for r in rows if r.get('run_name', '')}


def _build_cmd(row: Dict[str, str], summary_csv: str | None = None) -> List[str]:
    cmd = [
        sys.executable,
        'experiments/recoverability/train_ppo_adversary.py',
        '--r',
        str(row['r']),
        '--reward_mode',
        str(row['reward_mode']),
        '--seed',
        str(row['seed']),
        '--max_env_steps',
        str(row['max_env_steps']),
        '--env_horizon',
        str(row['env_horizon']),
        '--dt',
        str(row['dt']),
        '--umax',
        str(row['umax']),
        '--x_limit',
        str(row['x_limit']),
        '--mpsc_horizon',
        str(row['mpsc_horizon']),
        '--mpsc_cost_horizon',
        str(row['mpsc_cost_horizon']),
        '--init_scale',
        str(row['init_scale']),
        '--output_root',
        str(row['output_root']),
        '--summary_csv',
        str(summary_csv or row['summary_csv']),
        '--run_name',
        str(row['run_name']),
    ]
    if row.get('max_w', '') != '':
        cmd += ['--max_w', str(row['max_w'])]
    if row.get('v_limit', '') != '':
        cmd += ['--v_limit', str(row['v_limit'])]
    if row.get('train_mpsc_horizon', '') != '':
        cmd += ['--train_mpsc_horizon', str(row['train_mpsc_horizon'])]
    if row.get('eval_mpsc_horizon', '') != '':
        cmd += ['--eval_mpsc_horizon', str(row['eval_mpsc_horizon'])]
    if row.get('train_mpsc_cost_horizon', '') != '':
        cmd += ['--train_mpsc_cost_horizon', str(row['train_mpsc_cost_horizon'])]
    if row.get('train_sf_cost_function', '') != '':
        cmd += ['--train_sf_cost_function', str(row['train_sf_cost_function'])]
    if row.get('eval_sf_cost_function', '') != '':
        cmd += ['--eval_sf_cost_function', str(row['eval_sf_cost_function'])]
    if row.get('eval_from_checkpoint', '') != '':
        if str(row['eval_from_checkpoint']).strip().lower() in {'1', 'true', 'yes', 'y'}:
            cmd += ['--eval_from_checkpoint']
    if row.get('checkpoint_path', '') != '':
        cmd += ['--checkpoint_path', str(row['checkpoint_path'])]
    if row.get('regularization_weight', '') != '':
        cmd += ['--regularization_weight', str(row['regularization_weight'])]
    if row.get('blended_cost_alpha', '') != '':
        cmd += ['--blended_cost_alpha', str(row['blended_cost_alpha'])]
    if row.get('use_terminal_set', '') != '':
        if str(row['use_terminal_set']).strip().lower() in {'1', 'true', 'yes', 'y'}:
            cmd += ['--use_terminal_set']
    if row.get('terminal_set_scale', '') != '':
        cmd += ['--terminal_set_scale', str(row['terminal_set_scale'])]
    if row.get('terminal_cost_weight', '') != '':
        cmd += ['--terminal_cost_weight', str(row['terminal_cost_weight'])]
    return cmd


def _run_one(
    row: Dict[str, str],
    repo_root: str,
    base_env: Dict[str, str],
    log_dir: str,
    private_summary_dir: str,
) -> Dict[str, str]:
    run_name = row['run_name']
    private_summary_path = Path(private_summary_dir) / f'{run_name}.csv'
    cmd = _build_cmd(row, summary_csv=str(private_summary_path))
    env = dict(base_env)
    log_path = Path(log_dir) / f'{run_name}.log'
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with log_path.open('w') as logf:
        logf.write('CMD: ' + ' '.join(cmd) + '\n\n')
        logf.flush()
        proc = subprocess.run(
            cmd,
            cwd=repo_root,
            env=env,
            stdout=logf,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    return {
        'run_name': run_name,
        'returncode': str(proc.returncode),
        'log_path': str(log_path),
        'private_summary_path': str(private_summary_path),
    }


def _merge_private_summary(summary_csv: Path, private_summary_path: Path) -> bool:
    if not private_summary_path.exists():
        return False
    with private_summary_path.open('r', newline='') as source:
        private_rows = list(csv.DictReader(source))
    if not private_rows:
        return False

    existing_rows = _read_csv_rows(summary_csv) if summary_csv.exists() else []
    existing_names = {row.get('run_name', '') for row in existing_rows}
    row = private_rows[0]
    if row.get('run_name', '') in existing_names:
        return False

    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(row.keys())
    write_header = not summary_csv.exists() or not existing_rows
    with summary_csv.open('a', newline='') as target:
        writer = csv.DictWriter(target, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    return True


def _iter_pending(rows: Iterable[Dict[str, str]], completed: set[str]) -> List[Dict[str, str]]:
    return [r for r in rows if r.get('run_name', '') and r['run_name'] not in completed]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Local parallel runner for recoverability manifests.')
    p.add_argument('--manifest', type=str, required=True)
    p.add_argument('--repo_root', type=str, default='/dss/dsshome1/0A/go39bur2/safe-control-gym')
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--max_runs', type=int, default=0, help='0 means all pending runs')
    p.add_argument(
        '--log_dir',
        type=str,
        default='experiments/recoverability/ppo_runs/canonical_runs/local_logs',
    )
    p.add_argument(
        '--status_csv',
        type=str,
        default='experiments/recoverability/ppo_runs/canonical_runs/local_runner_status.csv',
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    manifest = (repo_root / args.manifest).resolve() if not Path(args.manifest).is_absolute() else Path(args.manifest)

    rows = _read_csv_rows(manifest)
    if not rows:
        print('No rows in manifest')
        return 0

    # All canonical rows currently share the same summary destination.
    summary_csv = repo_root / rows[0]['summary_csv']
    completed = _completed_run_names(summary_csv)
    pending = _iter_pending(rows, completed)
    if args.max_runs > 0:
        pending = pending[: args.max_runs]

    print(f'manifest_rows={len(rows)} completed={len(completed)} pending_to_launch={len(pending)}')
    if not pending:
        return 0

    base_env = dict(os.environ)
    acados_root = '/dss/dsshome1/0A/go39bur2/acados'
    base_env['ACADOS_SOURCE_DIR'] = acados_root
    base_env['LD_LIBRARY_PATH'] = f'{acados_root}/lib:' + base_env.get('LD_LIBRARY_PATH', '')
    base_env['OMP_NUM_THREADS'] = '1'
    base_env['MKL_NUM_THREADS'] = '1'
    base_env['OPENBLAS_NUM_THREADS'] = '1'
    base_env['NUMEXPR_NUM_THREADS'] = '1'

    log_dir = str((repo_root / args.log_dir).resolve())
    private_summary_dir = str((repo_root / args.log_dir / 'private_summaries').resolve())
    Path(private_summary_dir).mkdir(parents=True, exist_ok=True)
    status_csv = (repo_root / args.status_csv).resolve()
    status_csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ['run_name', 'returncode', 'log_path', 'private_summary_path']
    write_header = not status_csv.exists()

    with status_csv.open('a', newline='') as status_f:
        writer = csv.DictWriter(status_f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()

        with ProcessPoolExecutor(max_workers=max(1, int(args.workers))) as pool:
            inflight = set()
            pending_iter = iter(pending)

            # Prime queue
            for _ in range(max(1, int(args.workers))):
                try:
                    row = next(pending_iter)
                except StopIteration:
                    break
                inflight.add(
                    pool.submit(
                        _run_one,
                        row,
                        str(repo_root),
                        base_env,
                        log_dir,
                        private_summary_dir,
                    )
                )

            done_count = 0
            while inflight:
                done_set, inflight = wait(inflight, return_when=FIRST_COMPLETED)
                for fut in done_set:
                    result = fut.result()
                    done_count += 1
                    writer.writerow(result)
                    status_f.flush()
                    if result['returncode'] == '0':
                        _merge_private_summary(
                            summary_csv,
                            Path(result['private_summary_path']),
                        )
                    print(
                        f"[{done_count}/{len(pending)}] {result['run_name']} rc={result['returncode']} log={result['log_path']}"
                    )

                    try:
                        row = next(pending_iter)
                    except StopIteration:
                        continue
                    inflight.add(
                        pool.submit(
                            _run_one,
                            row,
                            str(repo_root),
                            base_env,
                            log_dir,
                            private_summary_dir,
                        )
                    )

    print(f'Completed launches: {len(pending)}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
