# V3 pilot-r2 status — 2026-09-06

Authoritative pilot root:

`results/brain-study/v3/seed-20260903-pilot-r2`

Immutable plan SHA-256:

`80bdda0b4626b3d01d841380f6cd6de35f0723053faa2fa6bd6d49fda8274985`

Execution configuration:

`configs/execution_v3_pilot_candidate_v1.json`

The first preparation-only cohort showed that capping both `screen_k` and
`parent_pool_max` at 10 prevented pair rescue from adding any parents. No
scientific fits were launched from that cohort. Resource-only sensitivities,
performed before inspecting scientific accuracy, selected `screen_k=10` and
`parent_pool_max=15` for max-indegree-four recognized networks. This preserves
five slots for pair rescue without the cost of brute-force screening to 20.

For non-exact panels, the predeclared complete-family recall threshold is
evaluated as a mean across the three independent datasets in each cell, matching
the experiment's statistical unit. Per-dataset minima and true-parent edge
recall remain reported descriptively. Every exact eight-node dataset must have
1.00 edge and complete-family recall. Artifact integrity is required for every
dataset.

The immutable plan contains 42 datasets and 453 fits: 15 Exact Order DP, 84
OrderSMC, 84 Order MCMC, 84 Partition MCMC, 84 REV-MCMC, 84 shared-score
bnlearn HC/Tabu, and 18 native `bic-cg` bnlearn HC/Tabu fits. There are two
stochastic-method seeds per dataset. OrderSMC uses 4,096 total particles over
64 within-run MPI ranks.

Slurm workflow:

- preflight: `10435759`
- preparation array: `10435760`
- candidate integrity/recall audit: `10435761`
- guarded inference launcher: `10435762`

The guarded launcher runs after the audit terminates, repeats the audit, and
submits the 453 fits only if it passes. The final aggregation writes raw
completion, per-run and seed-averaged DAGBagM-style metrics, exact-PIP gates,
seed stability, seed-averaged PIPs/representative DAGs, and threshold-ambiguity
tables. Raw OrderSMC results include per-edge MCSE, effective and genealogical
ESS, distinct root ancestors, unique orders, executed configuration, and MPI
RNG lineage.

This is a diagnostic resource-selection pilot, not the confirmatory cohort and
not evidence for a speedup or equal-compute claim.
