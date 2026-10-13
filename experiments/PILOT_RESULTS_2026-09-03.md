# Mixed-data validation results, 2026-09-03

## Evidence status

- V1 is the frozen confirmatory exact-posterior study: 90 datasets and 810
  planned rows. All 720 stochastic rows completed. Two of 90 exact-reference
  rows failed only in the supplementary `d=12, n/d=50` graph-size diagnostic;
  the six primary cells are complete and unaffected.
- V2 is a pilot: 48 datasets, 16 mixed-data cells, two seeds per stochastic
  method, and 384/384 completed rows. It is suitable for deciding what must be
  fixed before the V2 full cohort, not for a final confirmatory claim.
- All uncertainty intervals below resample independent datasets within cells;
  method seeds are averaged within dataset.

## V1 exact-posterior gate

Every primary cell passes the predeclared OrderSMC limits: mean directed-PIP
RMSE no larger than 0.012 and empirical dataset-level p95 no larger than 0.020.

| Dimension | Mixed composition | Mean PIP RMSE | Empirical p95 | OrderSMC s | Order MCMC s | Skeleton F1: SMC / MCMC | Cross-type directed F1: SMC / MCMC |
|---:|---|---:|---:|---:|---:|---:|---:|
| 8 | balanced | 0.00829 | 0.01258 | 60.4 | 275.8 | 0.826 / 0.817 | 0.412 / 0.423 |
| 8 | continuous-dominant | 0.00858 | 0.01234 | 61.8 | 275.8 | 0.888 / 0.892 | 0.588 / 0.588 |
| 8 | discrete-dominant | 0.00884 | 0.01452 | 62.7 | 275.8 | 0.811 / 0.811 | 0.460 / 0.468 |
| 12 | balanced | 0.00550 | 0.00696 | 183.4 | 278.7 | 0.892 / 0.892 | 0.493 / 0.493 |
| 12 | continuous-dominant | 0.00638 | 0.01009 | 187.1 | 278.5 | 0.906 / 0.906 | 0.699 / 0.699 |
| 12 | discrete-dominant | 0.00538 | 0.00781 | 181.5 | 278.6 | 0.847 / 0.847 | 0.613 / 0.615 |

Candidate true-parent and complete-family recall are 1.0 in all six primary
cells. OrderSMC is about 4.4--4.6 times faster at d=8 and 1.49--1.54 times
faster at d=12. Its worst cell-level lower paired 95% bound is -0.033 for
cross-type directed F1 and -0.0091 for skeleton F1, within the predeclared
-0.05 recovery-competitiveness margin.

Across four seeds, OrderSMC's mean pairwise PIP RMSE ranges from 0.00742 to
0.01263 across the six cells. Directed selected-graph Jaccard ranges from 0.960
to 1.000 and skeleton Jaccard from 0.982 to 1.000. Order MCMC has lower PIP
variation, but the OrderSMC selected graphs are highly reproducible in V1.

## V2 pilot macro-results

The table is an equal-cell macro-average. F1 excludes the mixed empty-DAG cell;
the other posterior metrics include all 16 cells.

| Method | Skeleton F1 | Cross-type directed F1 | Directed AUPRC | Sampling s | End-to-end s | Pairwise PIP RMSE | Directed Jaccard | Skeleton Jaccard |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| OrderSMC | 0.913 | 0.724 | 0.698 | 218.5 | 468.8 | 0.0720 | 0.794 | 0.926 |
| Order MCMC | 0.921 | 0.737 | 0.698 | 305.7 | 555.9 | 0.0435 | 0.899 | 0.968 |
| Partition MCMC | 0.665 | 0.363 | 0.214 | 301.0 | 551.2 | 0.2379 | 0.171 | 0.379 |
| REV-MCMC | 0.546 | 0.325 | 0.190 | 301.1 | 551.3 | 0.1861 | 0.124 | 0.235 |

For the same-target comparison, OrderSMC minus Order MCMC is -0.00817 in
skeleton F1 (95% CI -0.01030 to -0.00614) and -0.01309 in cross-type directed
F1 (95% CI -0.01991 to -0.00644). Both pass the predeclared lower-bound margin
of -0.05, although Order MCMC is modestly better at the selected operating
point. Directed AUPRC is essentially equal: difference 0.00018 (95% CI -0.00934
to 0.00961).

The equal-cell OrderSMC/Order-MCMC time ratio is 0.715 (95% CI 0.699--0.732)
for inference and 0.834 (0.824--0.844) end to end. This corresponds to about a
1.40x inference speedup and a 1.20x end-to-end speedup. Candidate construction
is therefore a substantial part of total cost and should be reported
separately.

On the mixed empty DAG, OrderSMC and Order MCMC each select two false edges on
average; Partition MCMC selects 10.7 and REV-MCMC 4.5. The weak performance and
very low seed agreement of Partition MCMC and REV-MCMC indicate inadequate
mixing within 300 seconds; these pilot results are not evidence of a general
formulation advantage.

## Gates that fail before V2 full

The screened candidate pool is adequate in the main dimension panel but fails
the >=0.95 complete-family-recall rule in three stress cells:

| Cell | Mean true-parent recall | Mean complete-family recall | Minimum complete-family recall |
|---|---:|---:|---:|
| d=50, n/d=5 | 0.875 | 0.773 | 0.700 |
| d=50, n/d=10 | 0.967 | 0.933 | 0.900 |
| d=50, weak signal | 0.955 | 0.907 | 0.860 |

These cells require the predeclared wider-pool sensitivity. Their current graph
recovery must be labelled candidate-limited.

OrderSMC also has worse large-domain seed stability than Order MCMC. The most
difficult OrderSMC cells are Student-t(3) noise (PIP RMSE 0.141, directed
Jaccard 0.585), centred Gamma noise (0.114, 0.668), and d=100 (0.0858, 0.656).
Before V2 full, use the pilot to freeze a higher-information SMC schedule or a
parallel ensemble of independent particle populations, then verify stability on
held-out pilot seeds. Do not tune on the V2 full cohort.

## Implementation findings

The distributed smoke run exposed an avoidable REV-MCMC cost. Replacing repeated
whole-DAG construction by an equivalent cached descendant test made all 16
smoke rows complete; exhaustive three-node checks prove the feasibility test is
equivalent.

The two supplementary V1 exact failures came from subtracting very large log
scores before graph-size normalization. Shifted compensated normalization now
returns 132 directed PIPs and total graph-size mass 1.0 for both preserved
failing tables. Raw failures remain in the original result tree. Aggregation was
also made failure-aware. The complete core and experiment test suites pass after
both changes.

## Result locations

- V1: `results/brain-study/v1/seed-20260904-full/analysis/`
- V2: `results/brain-study/v2/seed-20260903-pilot/analysis/`
- Amendment audit: `experiments/configs/budget_freeze_v1.json`

The evidence is promising but not submission-complete. V1 supports the central
posterior-approximation claim. V2 supports recovery competitiveness and a real
runtime advantage, but V2 full should not start until candidate recall and
large-domain seed stability are addressed. Accepted external mixed-data
comparators and recognized-network V3 results are also still required.
