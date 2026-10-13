# V3 pilot launch status — 2026-09-06

> This first resource-selection cohort stopped at its candidate gate (39/42
> datasets met the original per-dataset threshold). It contains no scientific
> inference results and is superseded by `V3_PILOT_R2_STATUS_2026-09-06.md`.

Result root:

`results/brain-study/v3/seed-20260903-pilot`

Immutable plan SHA-256:

`80bdda0b4626b3d01d841380f6cd6de35f0723053faa2fa6bd6d49fda8274985`

The plan contains 42 independently generated datasets and 453 fits:

- Exact Order DP: 15
- OrderSMC: 84 (4,096 total particles per fit; 64 within-run MPI ranks)
- Order MCMC: 84
- Partition MCMC: 84
- REV-MCMC: 84
- bnlearn HC/Tabu with the shared mixed score: 84
- native bnlearn HC/Tabu with `bic-cg` on the compatible CLG panel: 18

There are three datasets per cell and two stochastic-method seeds per dataset.
Method seeds are repeated measurements; paper summaries average edge PIPs within
dataset before computing the representative DAG and graph-recovery metrics.

Slurm workflow:

- preflight: `10435675` (completed successfully)
- dataset preparation array: `10435676`
- candidate integrity/recall audit: `10435677`
- post-audit inference launcher: `10435704`

The launcher runs after the audit reaches a terminal state. It rechecks the
audit and submits no fits unless every candidate artifact is intact and both
true-parent and complete-family recall are at least 0.95 (1.00 for exact
eight-node cells). If the audit passes, it submits only missing immutable runs
and then aggregates the cohort.

Raw OrderSMC results persist per-edge MCSE, effective and genealogical ESS,
distinct root ancestors, unique final/visited orders, the executed sampling
configuration, and full protocol MPI RNG lineage. Analysis writes per-seed
stability, seed-averaged PIPs and representative DAGs, threshold ambiguity,
and seed-averaged DAGBagM-style metrics.

This is a diagnostic pilot used to freeze candidate and compute resources. Its
4,096-particle, 64-rank allocation is not an equal-compute or speedup claim and
must remain separate from the later confirmatory cohort.
