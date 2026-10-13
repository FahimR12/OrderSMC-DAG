# V3 start status — updated 2026-09-06

## Completed correctness evidence

- V1 full exact validation is complete: 810/810 planned rows.
- Exact Order DP: 90/90; OrderSMC: 360/360; Order MCMC: 360/360.
- OrderSMC passes the frozen mean and empirical-95th-percentile PIP-RMSE gates
  in every primary mixed-composition cell and every sample-size supplement.

## V3 implementation

- Frozen topologies: Asia, Insurance, ALARM, and Hailfinder, with hashes of the
  source BIF archives recorded in `fixtures/known_topologies_v1.json`.
- Native mixed benchmark: independently simulated replicates from the published
  `bnlearn::clgaussian.test` DAG and fitted conditional-Gaussian mechanisms.
- Native external controls: bnlearn HC and Tabu with `bic-cg`, restricted to the
  conditional-Gaussian-compatible panel.
- Search controls: bnlearn HC and Tabu using the same serialized OrderSMC local
  mixed score on every V3 panel.
- Environment: R 4.4.3 and bnlearn 5.2.1 in the pinned
  `ordersmc-bnlearn` environment.
- OrderSMC uses within-run MPI only; exact DP, MCMC, and bnlearn are
  single-process runs.

## Current smoke cohort

Authoritative root:

`results/brain-study/v3/seed-20260903-smoke-r2`

- 14/14 datasets and candidate tables prepared.
- 95/95 planned rows complete with no failures.
- Exact DP: 5/5.
- OrderSMC, Order MCMC, Partition MCMC, and REV-MCMC: 14/14 each.
- Native bnlearn: 6/6; shared-score bnlearn: 28/28.

Smoke caps observations at `max(100, 5 * dimension)` while retaining the
declared paper sample size in `benchmark_n_train`. It is a plumbing test and its
accuracy values are not scientific results. In particular, minimum candidate
edge recall (0.879) and complete-family recall (0.857) are below the 0.95 paper
gate at this deliberately reduced sample size; this must be reassessed on the
pilot at the declared sample sizes.

The earlier `seed-20260903-smoke` root used unreduced sample sizes and is not a
reportable cohort. Partial Slurm chains created during the controller outage
must be cancelled/confirmed absent before MPI rows are resubmitted. Do not pool
either smoke cohort with pilot or confirmatory results.

## Gate to the pilot

1. Prepare all 42 three-replicate pilot datasets at the declared sample sizes.
2. Pass the immutable candidate-table audit: at least 0.95 true-parent and
   complete-family recall, and 1.00 recall in exact eight-node cells.
3. Run all 453 pilot rows, including 84 OrderSMC rows with 4,096 total
   particles over 64 MPI ranks and two method seeds per dataset.
4. Report seed-averaged PIPs/graphs, per-seed stability, threshold ambiguity,
   SMC edge MCSE, genealogical ESS, root ancestors, and unique orders.
5. Freeze candidate-pool width and compute resources using runtime and recall
   diagnostics only, then create a separate full confirmatory cohort.
