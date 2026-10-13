# OrderSMC-DAG: Lean Experimental Protocol, Version 3.0

Four confirmatory experiments and one mandatory validation gate for a reproducible mixed-data Bayesian structure-learning study.

| Field | Value |
|---|---|
| Protocol version | 3.0 |
| Protocol date | 14 August 2026 |
| Statistical unit | One independently generated graph--dataset pair |
| Proposed-method target | Candidate-restricted order-modular posterior |
| Status | Design and pilot settings frozen; every confirmatory launch requires the validation gate |

## 1. Purpose and permitted claims

The empirical programme answers four questions with the smallest defensible set of experiments.

| Experiment | Primary question | Core methods | Scale | Main output |
|---|---|---|---|---|
| E1. Exact posterior validation | Does OrderSMC approximate its declared posterior? | Exact Order DP, OrderSMC, Order MCMC | d in {8, 12} | PIP error versus exact inference |
| E2. Main mixed-data benchmark | Is OrderSMC competitive on mixed networks? | OrderSMC, Order MCMC, Partition MCMC, REV-MCMC | d in {20, 50, 100} | Accuracy, calibration, and runtime table |
| E3. Difficult-posterior time-to-accuracy | When is population SMC preferable to ordinary MCMC? | OrderSMC and Order MCMC primary; structure-target methods secondary | d in {50, 100} | Error and stability versus wall-clock time |
| E4. MPI strong scaling | Does the implementation exploit parallel hardware? | OrderSMC | d in {100, 200} | Runtime, speedup, and efficiency |

The programme intentionally excludes broad baseline sweeps, large tuning grids, and many auxiliary metrics. A mandatory preflight gate, G0, validates the score mathematics and sampler primitives but is not presented as a fifth scientific experiment.

The paper may claim empirical posterior accuracy relative to the declared target, competitive recovery on the simulated mixed-data regime, faster stabilization when supported by E3, and measured parallel scaling when supported by E4. It must not claim causal identification from observational data, universal superiority, or native mixed-data support in reimplemented competitors.

## 2. Targets and comparison classes

For node i, let C_i be its retained parent sets, ell_i(P) the local log score, and r_i(P) the structural log potential. OrderSMC and Order MCMC target

\[
\gamma_\beta(\sigma) = \prod_i \sum_{P \in C_i,\; P \subseteq \mathrm{Pred}_\sigma(i)} \exp\{r_i(P)+\beta\ell_i(P)\},\quad 0 \leq \beta \leq 1.
\]

Their scientific target is the normalized distribution at beta = 1. Exact Order DP evaluates this same candidate-restricted order-modular posterior.

Partition MCMC and REV-MCMC target a candidate-restricted structure-modular posterior over DAGs. Sharing local scores does not make the order- and structure-modular posteriors identical: marginalization over orders can weight a DAG according to its compatible orders. Comparisons are consequently classified as follows.

- **Same-target inference comparisons:** Exact Order DP, OrderSMC, and Order MCMC. PIP error to a shared exact or high-budget order-target reference is permitted.
- **End-to-end formulation comparisons:** OrderSMC against Partition MCMC or REV-MCMC. Recovery, calibration, runtime, and failure rates may be compared, but a PIP difference between targets is not called sampler error.

Within each target class, methods use the same candidate table, local scores, and target-calibrated edge-count prior. Prior calibration is performed without outcome data and frozen before confirmatory execution.

## 3. Mandatory validation gate G0

G0 must pass before E1--E4 begin. It is a software and mathematical acceptance gate, not a paper performance experiment.

1. BDeu and Gaussian Normal--Inverse-Gamma local scores agree with independent closed-form calculations.
2. Binary-logistic and categorical-softmax Laplace scores agree with independent quadrature or importance-sampling references on frozen small-family fixtures. The maximum absolute log-score errors are 0.10 for binary-logistic fixtures and 0.20 for the deliberately difficult small-n softmax fixture. These bounds validate and characterize the approximation; they do not assert that Laplace evidence is exact.
3. All 72 structural branch cells execute and return finite values. Independent formula or numerical-integration anchors cover every distinct score branch; repeated integrations of every sample-size and parent-count combination are not required.
4. Score invariance, derivative, degenerate-data, category-relabelling, and prior-boundary checks pass.
5. On enumerated tiny order spaces, SMC incremental weights and Order-MCMC acceptance probabilities agree with direct calculations.
6. Fixed data, configuration, seed, and execution layout replay to identical serialized outputs, or to a documented numerical tolerance when MPI reduction ordering prevents byte identity.

A failed cell blocks only the affected score family or execution path, but no affected confirmatory row may run. E1 alone cannot replace G0 because Exact Order DP and OrderSMC consume the same local-score table and could agree on an incorrectly computed table.

## 4. Common data-generating process

Each statistical replicate is an independently generated DAG and dataset. Node types are assigned as evenly as possible among continuous, binary, and categorical variables; categorical cardinality is sampled from {3, 4}. All parent--child type combinations are permitted.

- Graphs are generated after sampling a random topological order. E1 uses expected mean indegree 1.5 and maximum indegree 2. E2 uses expected mean indegree 2 and maximum indegree 3.
- Continuous children use standardized linear-Gaussian mechanisms, binary children use logistic mechanisms, and categorical children use baseline-invariant softmax mechanisms.
- The primary signal regime is moderate. Mechanism coefficients, intercept calibration, marginal-probability acceptance rules, and all rejected generator attempts are recorded in the dataset manifest.
- E3 uses a frozen difficult-instance generator with modular or hub topology, weak cross-module edges, weakly identified orientations, and several plausible order blocks. Dataset membership is determined only from generator metadata and data-free structural criteria. No method output is used to select the confirmatory cohort.
- Every dataset stores the adjacency matrix, variable types, mechanisms, train data, optional test data, generator seed hierarchy, rejection history, and content hashes.

The observational results describe recovery under this synthetic mechanism family. They do not establish causal identification without additional assumptions.

## 5. Candidate sets, local scores, and priors

E1 uses every parent set up to maximum indegree 2, so candidate screening cannot confound exact-target validation.

E2--E4 construct one candidate table per dataset and reuse its exact bytes across all compatible methods and seeds. The primary candidate policy is maximum indegree 3, screen_k = 20, parent_pool_max = 30, no family-count truncation after the selected parent pool, and the frozen pair-rescue rule in the implementation. Candidate construction uses the training data only.

The following are reported for every dataset:

- directed true-parent recall and complete-family recall;
- number of retained parents and parent sets per child;
- candidate construction time and peak memory;
- the candidate-table content hash.

On five predeclared E2 datasets per dimension, a wider screen_k = 30 and parent_pool_max = 40 sensitivity run assesses truncation. This is supplementary and does not alter the primary candidate set after results are observed.

The primary mixed score uses the implementation's native decomposable family scores: BDeu for wholly discrete families, Normal--Inverse-Gamma evidence for Gaussian regressions, and proper-prior Laplace evidence for logistic and softmax families. All score hyperparameters are frozen by G0 and a separate pilot. A DG-BIC analysis, if performed, is labelled a pseudo-posterior sensitivity analysis rather than exact finite-sample Bayesian evidence.

The structural potential is r_i(P) = -lambda |P|. Lambda is calibrated separately for the order- and structure-modular targets to a declared prior expected mean indegree. The same calibrated value is used by every method within a target class.

## 6. Budgets, tuning, and failures

Hyperparameters are chosen using E1 and separate pilot datasets that never enter confirmatory summaries. They are then frozen across E2 and E3. No dataset-specific retuning is permitted.

E2 and E3 use equal wall-clock budgets on the same hardware allocation. Preprocessing and candidate construction are reported separately and are also included in end-to-end runtime. The method-only timer starts from the identical serialized candidate table. CPU model, core count, numerical thread limits, memory limit, package versions, MPI implementation, and backend are recorded.

Every planned row remains visible. Completed, timeout, out-of-memory, numerical-failure, and unavailable-adapter rows are reported explicitly. Failed rows are never silently dropped or replaced. Accuracy summaries report both completed-run results and completion/failure rates.

## 7. Metrics and statistical analysis

For directed PIPs p_ij, the primary exact-target error is

\[
\mathrm{RMSE}_{\mathrm{PIP}} = \sqrt{\frac{1}{d(d-1)}\sum_{i\neq j}(\widehat p_{ij}-p^{\mathrm{ref}}_{ij})^2}.
\]

For an unordered pair {i,j}, skeleton probability is

\[
p^{\mathrm{skel}}_{ij}=p_{i\to j}+p_{j\to i}.
\]

Primary metrics are directed-edge AUPRC, directed-edge Brier score, skeleton AUPRC, and skeleton F1 at a threshold frozen by the pilot. SHD is secondary. To avoid producing a cyclic graph by independent directed-edge thresholding, point-DAG SHD uses the highest-posterior sampled DAG under a frozen deterministic tie rule. Posterior expected pair-state SHD is also reported directly from the three pair-state probabilities and does not require a representative-DAG optimizer.

The graph--dataset pair is the independent statistical unit. Seeds are repeated measurements nested within a dataset: seed-level metrics are averaged within dataset before confirmatory contrasts. All method contrasts are paired by dataset. Report the mean difference and a 95% stratified paired bootstrap confidence interval obtained by resampling whole datasets within dimension. Never bootstrap individual edges or treat seeds as independent datasets.

E1 PIP RMSE is the sole primary correctness endpoint. E2 directed AUPRC and directed Brier are co-primary; skeleton recovery and runtime are supporting endpoints. E3 time-to-reference is primary. Other quantities are descriptive, limiting multiplicity without a large correction family.

## 8. Reproducibility contract

Reproducibility has two distinct meanings.

1. **Replay reproducibility:** identical data, configuration, seed, backend, and MPI layout reproduce identical output bytes where deterministic reductions are used. Otherwise every scalar and PIP must satisfy a frozen absolute and relative tolerance.
2. **Across-seed stability:** different valid Monte Carlo seeds should yield similar posterior summaries, not identical sampled graphs.

All full E1--E3 cells use four independent method seeds. In addition, a reproducibility subset uses 20 seeds for OrderSMC and Order MCMC on five predeclared difficult datasets at each of d = 50 and d = 100. Report pairwise PIP RMSE, mean per-edge seed standard deviation, top-edge inclusion frequency, skeleton Jaccard, and the 95th percentile across datasets. Thresholded-graph instability is interpreted alongside continuous PIPs because edges close to the threshold can change the selected graph despite stable probabilities.

Master dataset seeds, method seeds, reference seeds, and timing-repeat seeds occupy disjoint deterministic namespaces and are committed before generation or inference.

## 9. E1: exact posterior validation

E1 uses d in {8,12}, n = 20d, ten datasets per dimension, and four independent runs per stochastic method and dataset. The complete maximum-indegree-2 candidate pool is shared by Exact Order DP, OrderSMC, and Order MCMC.

The primary table reports PIP RMSE to Exact Order DP, directed AUPRC, directed Brier score, and end-to-end runtime by dimension. Dataset-level means and 95% paired bootstrap intervals are reported. Exact Order DP has PIP RMSE zero by definition.

E1 supports the claim that OrderSMC empirically approximates the declared order-modular posterior on instances where exact summation is feasible. It does not prove the local-score mathematics, establish perfect graph recovery, or establish a scalability advantage.

## 10. E2: main mixed-data benchmark

E2 uses d in {20,50,100}, n = 20d, twenty datasets per dimension, and four runs per method and dataset. The graph family is sparse random-order with expected mean indegree 2 and maximum indegree 3.

The primary same-target panel compares OrderSMC and Order MCMC. The end-to-end panel adds Partition MCMC and REV-MCMC. These adapters implement the published state spaces and transitions under the common candidate-restricted mixed score; this does not claim that the authors' original software natively supports this mixed model.

The main table reports directed AUPRC, directed Brier score, skeleton F1, posterior expected pair-state SHD, completion rate, candidate construction time, sampling time, and end-to-end time. Candidate recall and wider-pool sensitivity appear in the supplement.

DAG-GFlowNet is an optional modern end-to-end comparator if a faithful, tested adapter is available before the pilot freeze. Bayesian Causal Discovery Networks for Linear Mixed Data is optional only on a separately labelled compatible linear-mixed subpanel. Neither method may be inserted after confirmatory outcomes are inspected, and unavailable methods remain visible in the method registry.

## 11. E3: difficult-posterior time-to-accuracy

E3 uses d in {50,100}, n = 20d, and ten precommitted difficult graph--dataset pairs per dimension. OrderSMC versus Order MCMC is the primary same-target comparison. Partition MCMC and REV-MCMC may be shown as secondary structure-target results but are never evaluated against the order-target reference.

Budget points occur at 30, 60, 180, and 300 seconds as separate same-seed runs under a frozen effort ladder. This makes every reported OrderSMC point a completed beta = 1 approximation; the observed runtime and any budget overrun are retained. The corresponding MCMC runs use the same random stream prefix and stop at each wall-clock budget. A partially annealed SMC population is never reported as a posterior estimate.

The order-target empirical reference is constructed from independent high-budget runs with seeds disjoint from evaluated runs. Multiple reference runs must satisfy frozen split-run and between-run agreement checks. A separate structure-target reference is constructed for structure methods. Evaluated runs are never included in their own reference, including through pooling.

For independent runs a and b, stability at time t is summarized by

\[
D_{ab}(t)=\sqrt{\frac{1}{d(d-1)}\sum_{i\neq j}\left[\widehat p^{(a)}_{ij}(t)-\widehat p^{(b)}_{ij}(t)\right]^2}.
\]

This is a disagreement measure, not an accuracy measure. Accuracy uses PIP RMSE to the independent target-specific reference. The primary outcome is time to cross a stability-and-reference threshold frozen by the pilot; timeouts are right-censored and reported. AUPRC trajectories, top-edge variance, and meaningful ESS-per-second diagnostics are secondary.

## 12. E4: MPI strong scaling

E4 uses d in {100,200}, n = 20d, five datasets per dimension, the pilot-frozen particle population N = 512, and ranks

\[
r in {1, 2, 4, 8, 16, 32, 64, 128, 256}.
\]

Each dataset--rank cell has three timing repeats. Data, candidate-table bytes, particles, temperature schedule or adaptation rule, resampling, rejuvenation work, and seed mapping are fixed; only rank count changes. Candidate construction is reported separately. Both inference-stage and end-to-end time are reported.

For median runtime T_r,

\[
S_r=\frac{T_1}{T_r},\qquad E_r=\frac{S_r}{r}.
\]

Ranks through 32 define the primary useful-scaling region because they retain at least sixteen particles per rank. Ranks 64--256 deliberately probe saturation and are labelled as such; near-linear scaling is not expected there. CPU and rank binding, node allocation, MPI version, communication time, and resampling time are recorded.

Changing rank count may change floating-point reduction order or stochastic scheduling. A rank-layout validation therefore compares PIPs, evidence-ratio estimates, and other summaries against frozen tolerances. E4 supports a systems scalability claim only; it does not establish accuracy.

## 13. Execution order and paper footprint

1. Run G0 and block affected paths on any failure.
2. Generate separate pilot data, freeze score, prior, sampler, threshold, reference, candidate, resource, and failure rules.
3. Run E1 and verify exact-target accuracy plus replay reproducibility.
4. Run E2 and decide whether the method is competitive enough to support the paper.
5. Run E3 on the already frozen difficult-instance cohort.
6. Run E4 on the final OrderSMC configuration and university HPC system.

The main paper requires one exact-validation table, one mixed-benchmark table, one time-to-accuracy figure, and one MPI scaling figure. Reproducibility, candidate recall, wider candidate pools, detailed failures, and optional modern comparators belong in compact supplementary tables.

Every released result must include the protocol version, configuration and candidate-table hashes, dataset and seed manifests, raw per-seed outputs, environment lock, method status, and the command used to regenerate its aggregate table or figure.
