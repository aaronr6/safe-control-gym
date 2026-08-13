#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import subprocess
from pathlib import Path

REPO = Path('/dss/dsshome1/0A/go39bur2/safe-control-gym')
MANIFEST = REPO / 'experiments/recoverability/sweeps_phase2/final_v112_filter_rerun_manifest.csv'
SUMMARY = REPO / 'experiments/recoverability/ppo_runs/slurm_final_v112_filter_rerun/summary.csv'
SBATCH = REPO / 'experiments/recoverability/recoverability_array.sbatch'
PARTITION = 'cm4_inter'
AUTOSUBMIT_STOP_FILE = REPO / 'experiments/recoverability/STOP_FILTER_RERUN_AUTOSUBMIT'


def run(cmd: list[str]) -> tuple[int, str, str]:
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO))
    return proc.returncode, proc.stdout, proc.stderr


def load_manifest() -> list[dict]:
    with MANIFEST.open(newline='') as handle:
        return list(csv.DictReader(handle))


def completed_run_names() -> set[str]:
    if not SUMMARY.exists():
        return set()
    with SUMMARY.open(newline='') as handle:
        return {row['run_name'] for row in csv.DictReader(handle)}


def active_indices() -> set[int]:
    rc, out, _ = run(['squeue', '-u', 'go39bur2', '-h', '-o', '%i %j'])
    if rc != 0:
        return set()
    active = set()
    for line in out.splitlines():
        parts = line.split()
        if len(parts) != 2:
            continue
        jid, name = parts
        if not name.startswith('rerun_sweep'):
            continue
        if name.startswith('rerun_sweep_'):
            try:
                active.add(int(name.split('_', 2)[2]))
            except Exception:
                pass
            continue
        if '_[' in jid and jid.endswith(']'):
            try:
                active.add(int(jid.split('_[', 1)[1][:-1]))
            except Exception:
                pass
    return active


def current_submitted_count() -> int:
    rc, out, _ = run(['squeue', '-u', 'go39bur2', '-h', '-o', '%i'])
    if rc != 0:
        return 0
    return len([line for line in out.splitlines() if line.strip()])


def next_candidates(rows: list[dict], done: set[str], active: set[int]) -> list[int]:
    candidates = []
    for idx, row in enumerate(rows):
        if row['run_name'] in done:
            continue
        if idx in active:
            continue
        candidates.append(idx)
    return candidates


def submit_single(idx: int) -> tuple[bool, str]:
    rc, out, err = run([
        'sbatch',
        f'--partition={PARTITION}',
        f'--job-name=rerun_sweep_{idx}',
        f'--export=ALL,MANIFEST={MANIFEST.relative_to(REPO)},SLURM_ARRAY_TASK_ID={idx}',
        str(SBATCH),
    ])
    if rc != 0:
        return False, (err.strip() or out.strip())
    return True, out.strip()


def main() -> int:
    parser = argparse.ArgumentParser(description='Submit filter-only rerun sweep jobs within strict queue limits.')
    parser.add_argument('--max-new', type=int, default=1)
    parser.add_argument('--max-submitted', type=int, default=2)
    parser.add_argument('--status', action='store_true')
    args = parser.parse_args()

    if AUTOSUBMIT_STOP_FILE.exists() and not args.status:
        print(f'Autosubmission disabled by {AUTOSUBMIT_STOP_FILE}')
        return 0

    rows = load_manifest()
    done = completed_run_names()
    active = active_indices()
    submitted = current_submitted_count()

    remaining = len([row for row in rows if row['run_name'] not in done])
    print(f'manifest_rows={len(rows)} done={len(done)} active_indices={len(active)} remaining={remaining} submitted_now={submitted}')

    if args.status:
        return 0

    slots = max(0, args.max_submitted - submitted)
    if slots <= 0:
        print('No submit slots available right now.')
        return 0

    candidates = next_candidates(rows, done, active)
    if not candidates:
        print('No candidates to submit.')
        return 0

    n_submit = min(args.max_new, slots, len(candidates))
    print(f'submitting={n_submit}')
    ok_count = 0
    for idx in candidates[:n_submit]:
        ok, msg = submit_single(idx)
        print(f'idx={idx} ok={ok} msg={msg}')
        if ok:
            ok_count += 1
        else:
            break

    return 0 if ok_count > 0 else 1


if __name__ == '__main__':
    raise SystemExit(main())
