# Recoverability Study: Supervisor Update

**Analysis date:** 2026-08-13
**Dataset:** 4,800 completed checkpoint-only safety-filter reruns
**Source:** `slurm_final_v112_filter_rerun/summary.csv`

## Executive Summary

Since the last meeting, the study progressed from a completed 800-run adversarial sweep to a controlled filter-comparison experiment. The adversary checkpoints were held fixed and the safety filter was re-evaluated under six intervention objectives: regularization weights 0.25, 1.0, and 4.0, plus blended objectives with alpha 0.25, 0.5, and 0.75. All 4,800 planned reruns completed successfully.

The main result remains robust: **relative degree is the dominant determinant of adversarial exploitability**. Filter-cost changes affect intervention behavior and control effort, but they do not erase the sharp increase in episode violations with increasing relative degree.

## Key Results

| Relative degree | Mean episode violation rate | Interpretation |
|---:|---:|---|
| `r=1` | **0.000** (95% bootstrap CI 0.000-0.000) | Predictive filter remains highly effective |
| `r=2` | **0.380** (95% bootstrap CI 0.353-0.407) | Intermediate recoverability regime |
| `r=3` | **0.807** (95% bootstrap CI 0.785-0.829) | Strong exploitability |
| `r=4` | **1.000** (95% bootstrap CI 0.999-1.000) | Near-complete loss of recoverability |

The exact means and 95% bootstrap intervals are in [aggregate_by_relative_degree.csv](aggregate_by_relative_degree.csv). The full setting-by-relative-degree matrix is in [aggregate_by_cell.csv](aggregate_by_cell.csv).

Pairwise relative-degree contrasts are in [relative_degree_pairwise_tests.csv](relative_degree_pairwise_tests.csv). Every contrast is positive and strongly separated under the permutation test, with the smallest reported p-value bounded by the Monte Carlo resolution of the test (`p <= 0.0005`).

Across filter settings, mean episode violation rate varies only from **0.5438 to 0.5492**, while relative-degree means span **0.000 to 1.000**. This supports the interpretation that the intervention objective changes workload and control behavior much less than the system relative degree changes exploitability.

## What Changed Since The Last Meeting

1. Completed the original 800-run confirmatory sweep.
2. Rebuilt the paper bundle with bootstrap confidence intervals, permutation testing, and FDR correction.
3. Added tunable regularization of safety-filter input-rate changes.
4. Added a blended safety-filter objective combining predictive-action matching and input-rate regularization.
5. Added checkpoint-only evaluation, holding the adversarial controller fixed while changing only the filter.
6. Generated and completed a 4,800-run factorial filter rerun sweep.
7. Diagnosed a major SLURM inefficiency: individual jobs requested 112 CPUs and 224 GB despite short evaluations.
8. Replaced per-experiment scheduling with one 8-worker, 8-CPU, 32-GB parallel SLURM allocation. It completed the full rerun sweep in about 80 minutes once the compatibility and bookkeeping issues were fixed.
9. Added resumable result merging and a stop switch to prevent stale autosubmitter processes from creating duplicate jobs.

## Recommended Plots For The Meeting

Show these in this order:

1. [relative_degree_violation_rate.png](relative_degree_violation_rate.png)
   Best first figure. It communicates the central paper result: exploitability rises sharply with relative degree, with uncertainty intervals.

2. [filter_setting_relative_degree_heatmap.png](filter_setting_relative_degree_heatmap.png)
   Shows the complete 4,800-run factorial comparison and makes clear whether filter-cost choices change the relative-degree pattern.

3. [intervention_safety_tradeoff.png](intervention_safety_tradeoff.png)
   Shows the safety versus intervention-workload tradeoff and helps motivate the intervention-objective discussion.

4. [rerun_runtime_distribution.png](rerun_runtime_distribution.png)
   Optional operational figure documenting that checkpoint-only evaluation is short enough to run efficiently in a consolidated worker allocation.

## Statistical And Reproducibility Artifacts

- [headline_summary.csv](headline_summary.csv): compact headline metrics.
- [aggregate_by_relative_degree.csv](aggregate_by_relative_degree.csv): relative-degree means, standard deviations, and 95% bootstrap intervals.
- [aggregate_by_filter_setting.csv](aggregate_by_filter_setting.csv): filter-setting aggregates.
- [aggregate_by_cell.csv](aggregate_by_cell.csv): complete relative-degree, reward-mode, and filter-setting cells.
- [relative_degree_pairwise_tests.csv](relative_degree_pairwise_tests.csv): bootstrap contrasts and permutation tests for relative-degree effects.
- [run_level_comparisons.csv](run_level_comparisons.csv): run-level differences from the regularized-weight-1 reference.

## Important Interpretation

The rerun sweep is a **fixed-adversary filter comparison**, not a new adversary-training study. This is intentional: it isolates how intervention design changes the safety filter's behavior under the same learned exploit. The result therefore supports claims about filter robustness and recoverability under a controlled adversarial policy, while the original confirmatory sweep supports the broader relative-degree claim.

The strongest current paper framing is:

> Predictive safety filters can be adversarially exploited, and the severity of exploitability is governed primarily by system relative degree. Changing the intervention objective changes filter workload and control behavior, but does not by itself remove the recoverability erosion at higher relative degree.

## Caveats To Mention

- The reruns reuse fixed adversarial checkpoints; they do not retrain the adversary for each filter cost.
- The blended objective uses a regularization-only fallback for the integrator-chain environment because benchmark trajectory-observation APIs are not available there.
- Follow-up work should test whether longer prediction horizons, terminal ingredients, or adversary retraining change the high-relative-degree result.
