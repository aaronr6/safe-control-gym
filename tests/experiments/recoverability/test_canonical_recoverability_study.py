import sys
from pathlib import Path

import pandas as pd
from canonical_recoverability_study import build_study_grid, write_manifest

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / 'experiments' / 'recoverability'))


def test_build_study_grid_returns_canonical_combinations(tmp_path):
    rows = build_study_grid(
        r_values=[1, 2],
        horizons=[4, 8],
        umax_values=[1.0],
        max_w_values=[0.002],
        v_limit_values=[None],
        seeds=[0],
        reward_modes=['constraint_seeking', 'horizon_exhaustion'],
    )

    assert len(rows) == 8
    assert all(row['reward_mode'] in {'constraint_seeking', 'horizon_exhaustion'} for row in rows)
    assert any(row['r'] == 2 and row['mpsc_horizon'] == 8 and row['reward_mode'] == 'horizon_exhaustion' for row in rows)


def test_write_manifest_creates_csv(tmp_path):
    rows = [{'run_name': 'demo', 'r': 3, 'reward_mode': 'horizon_exhaustion', 'mpsc_horizon': 12}]
    output_path = tmp_path / 'manifest.csv'
    write_manifest(output_path, rows)

    manifest = pd.read_csv(output_path)
    assert list(manifest.columns)[:4] == ['run_name', 'r', 'reward_mode', 'mpsc_horizon']
    assert manifest.iloc[0]['run_name'] == 'demo'
    assert 'output_root' in manifest.columns
    assert 'summary_csv' in manifest.columns
