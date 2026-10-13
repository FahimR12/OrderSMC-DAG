# Corrected v0.7 catch-up experiments

This is a versioned rerun of the previously completed scientific cohorts,
not a new untouched confirmatory study. No participant-level living-brain
observations are used. Old results, publication editions and source snapshots
remain unchanged.

## Scope

| Cohort | Coverage | Corrected SMC fits |
|---|---|---:|
| V1 | All 90 small mixed datasets, four method seeds each | 360 |
| V3 mixed | All 220 topology-preserving mixed datasets, four seeds each | 880 |
| V3 CLG | All 60 datasets from the corrected CLG-BIC cohort, four seeds each | 240 |
| Native/rank comparison | All 96 fresh score/noise/sample-size/measurement conditions, eight seeds each | 768 |
| Rank budget comparison | Eight n=5,000 datasets, two budgets, 20 seeds each | 320 |
| Metadata Section 2 | All 23 M1–M5 datasets; eight M1 seeds and four elsewhere | 112 |
| **Total** | Counts overlap in data/conditions; not independent datasets | **2,680** |

Sixteen metadata fits are shared with the separately frozen correction pilot,
so this plan schedules **2,664 new fits**. The pilot also schedules 16
proposal-only ablation fits, separate from the table above. Its original
v0.6 comparison uses saved fits. Avoid rerunning those shared 16 fits.

The original misspecified native-CLG V3 rows are retained historically, not
resubmitted as the current model. Their replacement is the complete CLG-BIC
corrective cohort. Earlier smoke tests, tuning trials, failed metadata
calibrations and superseded pilots are not additional publication cohorts.
The 120 historical rank exact-only conditions are unchanged model diagnostics;
they need no new sampling. The fresh native/rank measurement variants are
paired conditions, and rank invariance is algebraic, not independent recovery
evidence.

## What changes and what stays fixed

The corrected geometric/adaptive bridge starts uniformly and retains the same
final posterior target. MPI now uses the recorded proposal mixture. Conditional
ESS target is 0.9, minimum temperature increment 1e-8, maximum 2,000 stages.
Inability to progress fails explicitly rather than forcing a collapsing step.

Keep each original dataset, candidate family, local score, edge prior, inference
seed, particle count, per-stage MH counts and MPI layout. V1 keeps 512 particles
and its serial layout; the large MPI studies keep 64 ranks and 4,096 particles.
The extended rank budget keeps 16,384 particles and 20/50 MH proposals rather
than baseline 10/25. An MH step is one proposal per particle, not a variable
sweep. Adaptive stage counts and runtime may change: this is **not** an
equal-compute or speedup comparison.

Metadata primary reporting is fixed in advance at beta-one snapshot **0**,
before extra moves. All metadata fits additionally save snapshots after 250
and 1,000 final-target proposals per particle. These are named sensitivity
analyses, not candidate winners chosen by truth-based recovery. M1 primary
mean-PIP graphs use the original four evaluation seeds (0–3); all eight enter
the reproducibility analysis. The four-seed competitor mean must not be compared
to an unlabeled eight-seed SMC mean.

The score-schema compatibility check rejects ambiguous old provenance and any
potentially affected two-category categorical native/rank-native score table.
Such a target requires rebuilding and rerunning every shared-score method;
it must not silently enter this sampler-only workflow. Binary-logit children,
K>=3 categorical children and CLG-BIC scores are unchanged. Every accepted
table is validated during freeze and again on the compute-node preflight.

## Reuse and provenance

Unchanged exact references, Order MCMC, REV MCMC, Partition MCMC, and bnlearn
fits are reused by content hash on the identical target. They are not described
as newly run v0.7 competitors. Order- and structure-modular targets remain
distinct; compare each sampler to its own exact reference when available.
Point-estimate graph searches have no posterior-MCSE claim.

`plan.json` freezes every fit, original seed/configuration, immutable input
locator and hash, and the executing source snapshot. `inventory.json` gives
cohort counts. `code_snapshot/` is the executable source. No score rebuilding
or data regeneration occurs. The correction-pilot fit identity and candidate
hash are checked before reuse; its raw result hash is recorded at reporting.

## Scheduling

`scripts/submit_catchup_v07.sh` activates the project environment in submission
and worker shells. It freezes into `results/sampler-corrections/v07-catchup-v1`
unless a new `OUTPUT_ROOT` is explicitly supplied. `PREPARE_ONLY=1` freezes
without scheduling.

The first job runs the regression tests and input/target preflight. A controller
scheduled with `afterany` submits scientific arrays **only if preflight passed**.
Each small/medium task holds up to four sequential inference seeds; metadata
uses one fit per task. This keeps the pending task count below the account's
2,500-job submission limit. Each MPI cohort permits two simultaneous fits;
V1 permits 12 serial tasks. There is no MPI rank-scaling experiment.

Per-cohort reports run with `afterany`, explicitly enumerate missing/invalid
fits and refuse incomplete-cohort scientific summaries. The metadata report
also waits for the shared pilot. No `afterok` chain can leave a permanent
`DependencyNeverSatisfied` dependency. Queue `Dependency` before preflight or
fit completion is normal; it is not an experiment result.

`submission.json` records every accepted job immediately. Reinvoking submission
does not duplicate recorded jobs. If submission stopped part-way through the
array rollout, inspect that file and resume `catchup_submit launch` from the
frozen snapshot; it skips previously accepted arrays. A failed or timed-out fit
needs an explicit reviewed recovery submission, not an automatic unbounded
retry. Do not delete result records to force a rerun.

## Outputs and interpretation

For each cohort, `analysis/<cohort>/` will contain:

- `completion.json`: every planned fit accounted for, including failures.
- `per_run_metrics.csv`: corrected, historical and unchanged-competitor metrics.
- `mean_pip_graph_metrics.csv`: constrained Bayes-action DAG from evaluation-seed
  mean PIPs, separately from the average of individual-run graph metrics.
- `dataset_mean_metrics.csv` and `cell_summary.csv`: seeds averaged within data
  before datasets are summarized; experimental conditions remain separate.
- `seed_stability.csv`: directed/skeleton Jaccard, identical-graph indicator,
  complete-edge-inventory PIP RMSE, maximum edge-wise seed SD and SE of seed mean.
- `exact_rmse_flags.json`: mean <=0.012 and empirical p95 <=0.020 checks wherever
  a matching exact order reference exists. No matching reference means unavailable,
  not a pass. A completed job does not necessarily pass the accuracy check.
- `artifact_hashes.json` and `interpretation.json`: reporting provenance and caveats.

`analysis/completion.json` summarizes all cohorts after the reports finish.
Full posterior artifacts, histories and snapshot PIPs are retained under
`runs/<fit_id>/`. Sparse sampler PIPs are expanded to all d(d-1) directed edges
for reporting; an omitted edge is zero, not missing data.

Independent-seed SD describes run-to-run variation; SE describes variation of
their mean. Neither detects a bias common to all seeds or establishes
high-dimensional convergence. Genealogy-aware diagnostics must be considered
alongside independent seeds, and legacy within-run heuristic MCSE must not be
presented as a calibrated confidence interval. Existing publication figures
are not overwritten or automatically endorsed: inspect the corrected tables
and accuracy flags before updating manuscript figures or claims.
