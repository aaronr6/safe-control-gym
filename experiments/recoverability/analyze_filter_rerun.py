#!/usr/bin/env python3
"""Analyze completed checkpoint-based safety-filter reruns."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'ppo_runs/slurm_final_v112_filter_rerun/summary.csv'
OUT = ROOT / 'ppo_runs/paper_bundle/filter_rerun_analysis'
OUT.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(20260813)
df = pd.read_csv(DATA)
df = df.drop_duplicates('run_name').copy()
df['episode_violation_rate'] = df['violating_episodes'] / 3.0
df['cost_setting'] = np.where(
    df['eval_cost_function'].eq('regularized_cost'),
    'regularized rw=' + df['regularization_weight'].map(lambda x: f'{x:g}'),
    'blended alpha=' + df['blended_cost_alpha'].map(lambda x: f'{x:g}'),
)
df['filter_family'] = df['eval_cost_function'].map({'regularized_cost': 'Regularized', 'blended_cost': 'Blended'})

METRICS = ['episode_violation_rate', 'intervention_freq', 'intervention_mag', 'control_effort', 'elapsed_s']


def summarize(group):
    result = {'n': len(group)}
    for metric in METRICS:
        values = group[metric].dropna().to_numpy(float)
        result[metric + '_mean'] = values.mean() if len(values) else np.nan
        result[metric + '_sd'] = values.std(ddof=1) if len(values) > 1 else np.nan
        if len(values):
            boots = np.array([rng.choice(values, len(values), replace=True).mean() for _ in range(2000)])
            result[metric + '_ci_low'] = np.quantile(boots, 0.025)
            result[metric + '_ci_high'] = np.quantile(boots, 0.975)
        else:
            result[metric + '_ci_low'] = result[metric + '_ci_high'] = np.nan
    return pd.Series(result)


by_r = df.groupby('r').apply(summarize, include_groups=False).reset_index()
by_setting = df.groupby(['filter_family', 'regularization_weight', 'blended_cost_alpha'], dropna=False).apply(summarize, include_groups=False).reset_index()
by_cell = df.groupby(['r', 'reward_mode', 'cost_setting']).apply(summarize, include_groups=False).reset_index()
by_r.to_csv(OUT / 'aggregate_by_relative_degree.csv', index=False)
by_setting.to_csv(OUT / 'aggregate_by_filter_setting.csv', index=False)
by_cell.to_csv(OUT / 'aggregate_by_cell.csv', index=False)

pairwise = []
for low in sorted(df['r'].unique()):
    for high in sorted(df['r'].unique()):
        if high <= low:
            continue
        x = df.loc[df['r'].eq(low), 'episode_violation_rate'].to_numpy(float)
        y = df.loc[df['r'].eq(high), 'episode_violation_rate'].to_numpy(float)
        observed = y.mean() - x.mean()
        boot = np.array([
            rng.choice(y, len(y), replace=True).mean() - rng.choice(x, len(x), replace=True).mean()
            for _ in range(2000)
        ])
        pooled = np.concatenate([x, y])
        exceed = 0
        for _ in range(2000):
            shuffled = rng.permutation(pooled)
            exceed += abs(shuffled[:len(y)].mean() - shuffled[len(y):].mean()) >= abs(observed)
        pairwise.append({
            'r_low': low, 'r_high': high, 'difference_high_minus_low': observed,
            'bootstrap_ci_low': np.quantile(boot, 0.025),
            'bootstrap_ci_high': np.quantile(boot, 0.975),
            'permutation_p': (exceed + 1) / 2001,
        })
pd.DataFrame(pairwise).to_csv(OUT / 'relative_degree_pairwise_tests.csv', index=False)

# Difference from the regularized rw=1 reference within each (r, reward_mode).
ref = df[df['cost_setting'].eq('regularized rw=1')][['r', 'reward_mode', 'episode_violation_rate']].rename(columns={'episode_violation_rate': 'reference_violation_rate'})
comparison = df.merge(ref, on=['r', 'reward_mode'], how='left')
comparison['delta_vs_regularized_rw1'] = comparison['episode_violation_rate'] - comparison['reference_violation_rate']
comparison.to_csv(OUT / 'run_level_comparisons.csv', index=False)

# Plot 1: relative degree profile with bootstrap intervals.
fig, ax = plt.subplots(figsize=(8, 5))
ax.errorbar(by_r['r'], by_r['episode_violation_rate_mean'],
            yerr=[by_r['episode_violation_rate_mean'] - by_r['episode_violation_rate_ci_low'],
                  by_r['episode_violation_rate_ci_high'] - by_r['episode_violation_rate_mean']],
            fmt='o-', capsize=4, color='#b6422b', linewidth=2)
ax.set(xlabel='Relative degree r', ylabel='Episode violation rate', title='Adversarial exploitability remains dominated by relative degree')
ax.set_xticks(sorted(df['r'].unique()))
ax.set_ylim(-0.03, 1.03)
ax.grid(alpha=.25)
fig.tight_layout()
fig.savefig(OUT / 'relative_degree_violation_rate.png', dpi=180)
plt.close(fig)

# Plot 2: filter setting x relative degree heatmap.
pivot = df.pivot_table(index='cost_setting', columns='r', values='episode_violation_rate', aggfunc='mean').sort_index()
fig, ax = plt.subplots(figsize=(9, 5.5))
im = ax.imshow(pivot.to_numpy(), aspect='auto', cmap='YlOrRd', vmin=0, vmax=1)
ax.set_xticks(range(len(pivot.columns)), pivot.columns)
ax.set_yticks(range(len(pivot.index)), pivot.index)
ax.set(xlabel='Relative degree r', ylabel='Filter setting', title='Episode violation rate across all 4,800 completed reruns')
for i in range(pivot.shape[0]):
    for j in range(pivot.shape[1]):
        ax.text(j, i, f'{pivot.iloc[i, j]:.2f}', ha='center', va='center', color='black' if pivot.iloc[i, j] < .65 else 'white', fontsize=9)
fig.colorbar(im, ax=ax, label='Mean episode violation rate')
fig.tight_layout()
fig.savefig(OUT / 'filter_setting_relative_degree_heatmap.png', dpi=180)
plt.close(fig)

# Plot 3: intervention/safety tradeoff by setting.
fig, ax = plt.subplots(figsize=(8, 5.5))
for label, group in df.groupby('filter_family'):
    ax.scatter(group['intervention_freq'], group['episode_violation_rate'], s=10, alpha=.22, label=label)
    means = group.groupby('r')[['intervention_freq', 'episode_violation_rate']].mean()
    ax.plot(means['intervention_freq'], means['episode_violation_rate'], marker='o', linewidth=2, label=f'{label} means')
ax.set(xlabel='Intervention frequency', ylabel='Episode violation rate', title='Safety and intervention workload across reruns')
ax.set_ylim(-.03, 1.03)
ax.grid(alpha=.2)
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(OUT / 'intervention_safety_tradeoff.png', dpi=180)
plt.close(fig)

# Plot 4: runtime distribution.
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(df['elapsed_s'].dropna(), bins=40, color='#2e6f95', alpha=.85)
ax.axvline(df['elapsed_s'].median(), color='#b6422b', linestyle='--', label=f"median {df['elapsed_s'].median():.2f}s")
ax.set(xlabel='Evaluation runtime (s)', ylabel='Number of runs', title='Checkpoint-only rerun runtime distribution')
ax.grid(alpha=.2)
ax.legend()
fig.tight_layout()
fig.savefig(OUT / 'rerun_runtime_distribution.png', dpi=180)
plt.close(fig)

# Compact machine-readable headline summary.
headline = {
    'runs': len(df),
    'unique_runs': df['run_name'].nunique(),
    'mean_episode_violation_rate': df['episode_violation_rate'].mean(),
    'median_elapsed_s': df['elapsed_s'].median(),
    'mean_elapsed_s': df['elapsed_s'].mean(),
}
for r, row in by_r.set_index('r').iterrows():
    headline[f'r{r}_mean_episode_violation_rate'] = row['episode_violation_rate_mean']
pd.DataFrame([headline]).to_csv(OUT / 'headline_summary.csv', index=False)

print(f'Wrote analysis to {OUT}')
print(pd.DataFrame([headline]).to_string(index=False))
print('\nBy relative degree:')
print(by_r[['r', 'n', 'episode_violation_rate_mean', 'episode_violation_rate_ci_low', 'episode_violation_rate_ci_high']].to_string(index=False))
