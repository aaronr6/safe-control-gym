from pathlib import Path

import pandas as pd

from experiments.recoverability.build_systematic_recoverability_manifest import build_manifest, write_manifest


def test_build_manifest_contains_expected_rows(tmp_path: Path) -> None:
    rows = build_manifest(
        r_values=[1, 2],
        horizons=[4, 8],
        umax_values=[0.6],
        v_limit_values=[None, 1.2],
        seeds=[0],
        reward_modes=['constraint_seeking', 'horizon_exhaustion'],
    )
    assert len(rows) == 16
    assert rows[0]['run_name'].startswith('r1_')
    assert rows[0]['reward_mode'] == 'constraint_seeking'

    out_path = tmp_path / 'manifest.csv'
    write_manifest(out_path, rows)
    df = pd.read_csv(out_path)
    assert list(df.columns[:4]) == ['run_name', 'r', 'reward_mode', 'mpsc_horizon']
    assert 'output_root' in df.columns
    assert 'summary_csv' in df.columns
