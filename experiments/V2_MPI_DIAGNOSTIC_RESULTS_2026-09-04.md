# V2 high-particle MPI diagnostic results (2026-09-04)

## Status and scope

- Output: `results/brain-study/v2/seed-20260903-pilot-mpi-diagnostic-r64-r2`
- Completion: 96/96 OrderSMC rows after transport-only retries.
- Design: 48 independently generated mixed datasets, two inference seeds per dataset.
- Execution: 64 MPI ranks per OrderSMC run on one node.
- Inputs: all 48 regenerated dataset artifacts and candidate-table content hashes match the original V2 pilot.
- Interpretation: resource-unmatched diagnostic. Accuracy and reproducibility may be compared with the original V2 methods; runtime may not be used as an equal-compute claim.

The first pass completed 89/96 rows. Seven rows were lost when UCX attempted unreliable intra-node connections through the cluster fabric and aborted ranks. The seven rows completed normally after forcing `UCX_TLS=self,sm`. This was an infrastructure failure, not an OrderSMC numerical or statistical failure.

## Schedules

| Dimension | Particles per run | MPI ranks | Particles per rank | Independent seeds |
|---:|---:|---:|---:|---:|
| 20 | 4,096 | 64 | 64 | 2 |
| 50 | 2,048 | 64 | 32 | 2 |
| 100 | 1,024 | 64 | 16 | 2 |

## Dataset-level mean recovery

Inference seeds are averaged within a dataset before dataset means are calculated.

| d | Method | Datasets | Directed AUPRC | Directed Brier | Skeleton AUPRC | Skeleton F1 | Expected pair-state SHD |
|---:|---|---:|---:|---:|---:|---:|---:|
| 20 | MPI-OrderSMC | 3 | 0.5562 | 0.07839 | 0.9059 | 0.8700 | 23.45 |
| 20 | Original OrderSMC | 3 | 0.5680 | 0.07748 | 0.9041 | 0.8694 | 23.29 |
| 20 | Order MCMC | 3 | 0.5579 | 0.07822 | 0.9084 | 0.8718 | 23.43 |
| 50 | MPI-OrderSMC | 42 | 0.7284 | 0.01547 | 0.8820 | 0.8556 | 34.27 |
| 50 | Original OrderSMC | 42 | 0.6973 | 0.01698 | 0.8772 | 0.8488 | 36.75 |
| 50 | Order MCMC | 42 | 0.6945 | 0.01615 | 0.8787 | 0.8560 | 34.76 |
| 100 | MPI-OrderSMC | 3 | 0.9066 | 0.00436 | 0.9833 | 0.9611 | 37.24 |
| 100 | Original OrderSMC | 3 | 0.8346 | 0.00667 | 0.9731 | 0.9403 | 52.66 |
| 100 | Order MCMC | 3 | 0.8805 | 0.00447 | 0.9817 | 0.9628 | 37.79 |

Lower is better for Brier and SHD; higher is better for AUPRC and F1.

At `d=50`, the paired dataset-level MPI-OrderSMC minus Order MCMC differences were:

- directed AUPRC: +0.03382, 95% dataset-bootstrap interval [+0.02449, +0.04355];
- directed Brier: -0.00068, interval [-0.00122, -0.00021];
- skeleton AUPRC: +0.00332, interval [+0.00178, +0.00492];
- skeleton F1: -0.00038, interval [-0.00309, +0.00222];
- expected pair-state SHD: -0.49, interval [-1.32, +0.28].

MPI-OrderSMC had higher directed AUPRC than Order MCMC in every one of the 13 non-null `d=50` condition cells. The `d=20` and `d=100` comparisons contain only three datasets each and remain exploratory; their bootstrap intervals should not be treated as reliable inferential evidence.

## Effective sample size and elapsed time

Values are medians across individual method/seed rows, with minima and maxima in brackets.

| d | Method | ESS | Elapsed seconds |
|---:|---|---:|---:|
| 20 | MPI-OrderSMC | 3,662.9 [3,394.4, 3,771.6] | 188.9 [188.3, 190.8] |
| 20 | Original OrderSMC | 124.1 [114.4, 191.6] | 225.2 [192.0, 252.9] |
| 20 | Order MCMC | 61.4 [40.9, 140.5] | 300.0 [299.1, 300.7] |
| 50 | MPI-OrderSMC | 1,772.9 [1,034.1, 2,048.0] | 207.9 [153.8, 286.3] |
| 50 | Original OrderSMC | 92.4 [56.7, 112.0] | 215.1 [176.9, 298.4] |
| 50 | Order MCMC | 21.5 [1.0, 242.2] | 306.1 [297.1, 318.8] |
| 100 | MPI-OrderSMC | 850.2 [640.3, 1,024.0] | 149.5 [139.2, 196.2] |
| 100 | Original OrderSMC | 68.9 [44.7, 80.0] | 247.6 [224.2, 315.9] |
| 100 | Order MCMC | 3.6 [1.0, 5.0] | 305.3 [302.5, 305.5] |

These elapsed times establish that the diagnostic objective—fast individual high-particle results—was met. They are not a speedup claim because MPI-OrderSMC used 64 cores while each baseline run used one core.

## Inference-seed reproducibility

For each dataset, directed edge posterior probabilities from seed 0 and seed 1 are compared over every possible directed edge. The table reports the median dataset PIP RMSE (lower is better) and median Pearson correlation (higher is better).

| d | Method | Dataset pairs | PIP RMSE | PIP correlation |
|---:|---|---:|---:|---:|
| 20 | MPI-OrderSMC | 3 | 0.00323 | 0.99993 |
| 20 | Original OrderSMC | 3 | 0.01046 | 0.99928 |
| 20 | Order MCMC | 3 | 0.00689 | 0.99969 |
| 50 | MPI-OrderSMC | 42 | 0.04277 | 0.97527 |
| 50 | Original OrderSMC | 42 | 0.07627 | 0.91893 |
| 50 | Order MCMC | 42 | 0.03443 | 0.98210 |
| 100 | MPI-OrderSMC | 3 | 0.04760 | 0.94086 |
| 100 | Original OrderSMC | 3 | 0.07944 | 0.84338 |
| 100 | Order MCMC | 3 | 0.04075 | 0.95694 |

The larger MPI population substantially improves OrderSMC reproducibility at every dimension. It is better than Order MCMC at `d=20`, while Order MCMC retains slightly better median seed agreement at `d=50` and `d=100`. Only two seeds were used, so a publication-grade reproducibility claim still requires more independent inference seeds.

## Diagnostic conclusion

The underpowered original V2 SMC schedule was materially limiting performance. Increasing the particle population produces large ESS gains, substantially better seed agreement, improved recovery over the original SMC at `d=50` and `d=100`, and competitive or better recovery than the same-target Order MCMC. The strongest evidence is the heterogeneous `d=50` panel with 42 datasets. The result is promising enough to justify a confirmatory mixed-data experiment, but it does not yet support a compute-efficiency claim or a final cross-seed reproducibility claim.
