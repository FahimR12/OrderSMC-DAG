# V1 effort-matched adaptive diagnostic and accuracy-gate correction

This is development follow-up to the completed V1 catch-up, not an untouched
confirmatory experiment or a change to the score, prior or package default.
All 90 datasets and four original seeds per dataset are retained: **360 fits**,
grouped into 90 serial Slurm tasks (12 concurrent). Keeping the serial layout
preserves the original random-number streams; parallelism is across tasks.
The large 64-rank mixed/metadata jobs remain untouched.

## Fixed question and rule

Does restoring the legacy number of MH proposals improve adaptive v0.7's
posterior accuracy and independent-seed graph stability?

For each dataset/seed, let L be the legacy total proposed moves, A the existing
adaptive total, and P the unchanged particle count. Freeze

    extra final-target steps = (L - A) / P

from the saved **proposal counts only**, without using graph truth, recovery,
exact-posterior error or seed agreement. Noninteger or negative top-ups are
rejected. Run the same adaptive path, save snapshot zero, and add that many
ordinary MH proposals per particle at beta=1. This preserves the final target
and particle weights. Counts match exactly, but placement along the bridge,
accepted moves and computation time need not match. This is not an equal-time
or MPI-speedup benchmark, and final-only mixing may not recover early lost
particle ancestry.

The primary new arm is `adaptive_matched_effort`. Always retain and report
`adaptive_original_budget`, `legacy`, and the unchanged `order-mcmc` comparator.
There is no automatic selection of the best-performing arm or promotion of
this policy to a universal default. Dataset-dependent counts implement one
shared rule; they are not per-dataset accuracy tuning.

## Replay and immutability

The model/sampler files must match the source catch-up freeze. Each run checks:

- snapshot-zero edge probabilities and adaptive schedule exactly replay the
  saved adaptive fit;
- tempering proposal counts equal A;
- final total proposal counts equal L;
- final weights equal the saved adaptive weights.

All checks must pass before the fit is accepted for reporting. Data, candidate
tables, references, original fits and source snapshots are content-addressed
and preserved. New outputs are in this separate results root. A failed replay
is an error, not an excluded observation.

## Accuracy-gate formula

For dataset d, seed s and directed edge e:

    r_ds = sqrt(mean_e((estimated_PIP_dse - exact_PIP_de)^2))
    r_d  = mean_s(r_ds)

The original protocol gate is `mean_d(r_d) <= 0.012` AND
`quantile_0.95({r_d}) <= 0.020`, using linear quantile interpolation and at
least three datasets. This is neither a percentile over individual seeds nor
RMSE of the seed-averaged PIPs. Extra reproducibility-only seeds are excluded
from primary accuracy gates. Per-seed p95 remains a clearly labelled diagnostic.
Missing exact references and underpowered conditions are not counted as passes.

The same correction is applied to each catch-up cohort after its original
report finishes. `analysis/<cohort>/exact_rmse_flags.json` is the corrected
report; `accuracy_gate_correction.json` points to its audit manifest. The old
per-seed gate is archived under
`reporting_corrections/dataset-gate-v1/<cohort>/original_per_seed_gate.json`.
No fitted result, posterior, source snapshot, score or other metric is changed.

## Outputs

- `plan.json`, `inventory.json`, `code_snapshot/`: frozen design and provenance.
- `submission.json`, `logs/`: accepted jobs and progress; re-submission skips
  recorded jobs instead of creating duplicates.
- `validation/replay/`: per-fit exact replay and effort-count checks.
- `analysis/per_run_metrics.csv`, `dataset_metrics.csv`, `cell_summary.csv`:
  posterior error and graph recovery for every arm.
- `analysis/seed_stability.csv`: directed/skeleton Jaccard, identical-graph
  indicator, and complete-edge-inventory pairwise PIP RMSE.
- `analysis/mean_pip_graphs.json`: graphs obtained from seed-averaged PIPs,
  kept distinct from averages of individual-run graph metrics.
- `analysis/exact_rmse_flags.json`: dataset-level gates for every arm.
- `analysis/paired_dataset_contrasts.csv`: matched-minus-comparator differences
  with 10,000 paired dataset bootstrap draws, separately by experimental cell.
- `analysis/completion.json`: fit/replay completion, separate from accuracy
  success and publication readiness.

The controller submits new fits only after regression and input preflight pass.
Reports use `afterany` and explicitly reject missing/unverified fits. Passing
this diagnostic would justify further validation, not claims that all graph
directions are identifiable or that every seed must return an identical DAG.
