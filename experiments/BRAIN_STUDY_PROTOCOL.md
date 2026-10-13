# OrderSMC validation and living-brain study protocol, Version 1.1

## Status and scope

This protocol defines the evidence required before OrderSMC is applied to the
private living-brain dataset. It is intentionally separate from the older
generic E1--E4 performance programme. The new study has four sequential phases:

1. **V1 -- exact mixed-data validation:** verify posterior approximation on
   heterogeneous data when the exact candidate-restricted order posterior is
   available.
2. **V2 -- brain-matched synthetic validation:** measure graph recovery,
   uncertainty calibration, scalability, and seed stability under conditions
   chosen from data-free metadata of the brain study.
3. **V3 -- recognized network benchmarks:** evaluate recovery on published
   Bayesian-network topologies and, after an intervention adapter is accepted,
   the Sachs signalling data.
4. **V4 -- living-brain application:** fit and interpret the biological data
   only after V1--V3 and the brain-data validity gate pass.

The phases answer different questions. V1 validates inference against the
declared posterior. V2 and V3 evaluate recovery under known graphs. V4 is an
application with unknown ground truth and must not be used to establish
algorithm correctness.

## Claims that are and are not permitted

The study may support claims about approximation of a candidate-restricted
order-modular posterior for heterogeneous continuous, binary, and multilevel
categorical variables; coverage of parent--child type combinations unavailable
to narrower model classes; recovery under declared simulated mechanisms;
performance on recognized network topologies; computational efficiency; and
stable posterior associations in the brain data.

It does not by itself support unrestricted causal identification, correctness
under latent confounding, or instantaneous causal direction from dependent
time-series samples. Logistic and softmax family evidence is evaluated using the
implementation's proper-prior Laplace approximation and must be described as
such.

## G0 -- mandatory software and mathematical gate

Every launch begins with the existing strict score and adapter preflight. It
must establish all of the following:

- BDeu and Gaussian Normal--Inverse-Gamma scores agree with independent formulas.
- Logistic and softmax Laplace scores remain within their frozen numerical
  reference tolerances.
- Exact Order DP agrees with direct order enumeration on tiny systems.
- Order-MCMC, Partition-MCMC, and REV-MCMC finite-state kernels satisfy their
  detailed-balance and stationarity acceptance checks.
- A fixed dataset, candidate table, seed, configuration, and local execution
  layout replay identically.
- The process uses one numerical thread and records its package/environment lock.

A failed G0 blocks every affected row. It cannot be replaced by agreement
between Exact Order DP and OrderSMC because both consume the same score table.

## G1 -- comparator capability and failure audit

Before accuracy comparisons, each external comparator is run through an explicit
capability matrix. Required mixed-family motifs are continuous-to-binary,
continuous-to-categorical, binary-to-continuous, categorical-to-continuous,
binary-to-categorical, categorical-to-binary, and mixed-parent families for
each child type. Tiny homogeneous controls are used only to establish that an
adapter installation works; they are not scientific experiment cells.

Every method--motif pair receives exactly one disposition:

- `supported_complete`: the declared model supports the cell and returns a valid
  acyclic result;
- `supported_runtime_failure`: the model supports it but execution fails;
- `supported_numerical_failure`: execution returns invalid or non-finite output;
- `unsupported_model_class`: the published model or loss does not represent the
  cell;
- `unavailable_adapter`: no independently accepted adapter was frozen.

Unsupported cells remain visible but are not assigned an accuracy of zero.
Coverage is reported as the fraction of required mixed-family motifs natively
supported. Accuracy is compared only on the intersection of supported cells,
alongside a separate all-motif result for OrderSMC.

DAGMA's official interface provides an `l2` loss for continuous data and a
`logistic` loss for binary data. It is therefore a homogeneous-control
comparator, not a primary method on heterogeneous tables. bnlearn supports
conditional linear Gaussian networks, but discrete children can only have
discrete parents; it is evaluated on that restricted mixed subpanel and marked
`unsupported_model_class` for prohibited directions. These are scope
differences, not claims that either implementation is generally broken.

## Common rules

### Frozen inference configuration

The current pilot freeze remains the starting configuration:

- OrderSMC: 512 particles, 100 temperature points, 10 ordinary rejuvenation
  steps, and 25 post-resampling steps.
- Order MCMC: thinning 50 and a 270-second sampler allocation within a
  300-second total budget.
- Maximum indegree 2 and a complete family pool in V1.
- Maximum indegree 3 and the frozen candidate-screening policy in V2--V4,
  unless the brain metadata require a smaller declared bound.

No method is tuned separately on a confirmatory dataset or benchmark result.
Any later change creates a new protocol version and new confirmatory cohort.

### Replication profiles

The **smoke** profile uses one dataset per cell and one method seed. It checks
plumbing only and is never reported scientifically.

The **pilot** profile uses three datasets per cell and two method seeds. It may
identify failures and estimate resource requirements, but it cannot be pooled
with confirmatory results.

The **full** profile uses ten independent generated datasets per V1 cell and
twenty independent generated datasets per V2/V3 cell, with four stochastic
method seeds nested within each dataset. Dataset generation seeds, method seeds,
benchmark-sampling seeds, and brain resampling seeds occupy disjoint namespaces.

### Statistical unit and uncertainty

The independent unit is a generated graph--dataset pair or an independently
sampled dataset from a fixed benchmark network. Method seeds are repeated
measurements within that unit. They are averaged before comparisons.

Contrasts are paired by dataset. Confidence intervals use 10,000 stratified
bootstrap samples of complete datasets, never individual edges or method seeds.
Report cell-level estimates and an equal-cell macro-average so large networks do
not dominate merely because they contain more possible edges.

### Common outputs and DAGBagM-aligned recovery metrics

Every row retains its status, runtime, peak memory, environment, data hash,
candidate-table hash, configuration hash, seed record, posterior directed-edge
probabilities, representative graph, and diagnostics. Missing, timeout,
out-of-memory, numerical-failure, and unavailable-adapter outcomes remain in the
planned matrix.

The headline graph-recovery metrics deliberately mirror the DAGBagM paper. Let
\(A\) denote either the true skeleton-edge set or the true directed-edge set of
interest and let \(\widehat A\) be its estimate. Define

\[
\mathrm{TPR}=\frac{\mathrm{TP}}{\mathrm{TP}+\mathrm{FN}},\qquad
\mathrm{FDR}=\frac{\mathrm{FP}}{\max(1,\mathrm{TP}+\mathrm{FP})},\qquad
F_1=\frac{2\mathrm{TP}}{2\mathrm{TP}+\mathrm{FP}+\mathrm{FN}}.
\]

For a skeleton comparison, each unordered adjacent pair is counted once. For a
directed comparison, only an exactly oriented arc is a true positive; a reversed
arc contributes one false positive and leaves the true arc as one false
negative. When the true graph is empty, TPR and F1 are undefined and the number
of selected false edges is reported directly, as in DAGBagM's null-graph
experiment.

These quantities are reported for both (a) all skeleton edges and (b) directed
cross-type edges. The latter set contains every true arc whose parent and child
have different declared types, extending DAGBagM's continuous--binary endpoint
to continuous, binary, and multilevel categorical variables. It is additionally
split into the six ordered cross-type directions. Same-type directed results and
the full nine parent-type/child-type table are supplementary.

Each posterior method is reduced to one graph by a DAG-constrained Bayes action
that minimizes posterior expected pair-state SHD. This is the closest analogue
of DAGBagM's SHD aggregation rule. A method that natively returns one graph is
scored on that graph. Any stability or selection threshold is frozen using only
the pilot cohort and is then unchanged for full data. Because different native
decision rules can imply different operating points, TPR--FDR curves and AUPRC
from edge confidences are also reported when a method provides meaningful edge
weights.

For each cell, the paper-style table gives the mean and standard deviation of
TPR, FDR, and F1 across independent datasets after averaging method seeds within
dataset. The paired dataset-level 95% confidence intervals specified above are
reported alongside them; method seeds are never counted as independent
replicates.

For a true mixed-data DAG, report:

- skeleton TPR, FDR, and F1;
- directed cross-type TPR, FDR, and F1;
- directed-edge and skeleton AUPRC as threshold-free secondary metrics;
- directed and skeleton Brier score as posterior-calibration metrics;
- posterior expected pair-state SHD;
- candidate true-parent and complete-family recall;
- sampling and end-to-end runtime, peak memory, and completion rate;
- recovery stratified by all nine parent-type/child-type combinations, plus
  macro-averages over child type and over cross-type direction.

For observational score-equivalent benchmark networks, skeleton and CPDAG-level
errors are primary. Directed error against one arbitrary member of an
equivalence class is secondary. Directed recovery may be primary only when
interventions, temporal ordering, or a declared identifiable mechanism justifies
it.

## V1 -- exact mixed-data posterior validation

### Question

Does frozen OrderSMC approximate the exact candidate-restricted order-modular
posterior across different compositions of genuinely mixed family tables?

### Confirmatory cells

Use dimensions 8 and 12, training size n = 20d, expected mean indegree 1.5,
maximum indegree 2, moderate signal, and ten independent datasets in each of the
following heterogeneous variable-type regimes, each of which contains all three
variable classes:

1. balanced mixed: approximately one third continuous, one third binary, and one
   third three/four-level categorical;
2. continuous-dominant mixed: approximately 60% continuous, 20% binary, and 20%
   categorical;
3. discrete-dominant mixed: approximately 20% continuous, 40% binary, and 40%
   categorical.

This gives 6 cells and 60 datasets. Every dataset uses the complete parent-set
pool through indegree 2. Run Exact Order DP once and OrderSMC plus Order MCMC at
four independent method seeds, for 540 total rows.

### Primary endpoint and gate

The sole primary endpoint is directed PIP RMSE to Exact Order DP. OrderSMC passes
V1 only if, in every one of the six cells:

- mean PIP RMSE is at most 0.012;
- the empirical 95th percentile of dataset-mean PIP RMSE is at most 0.020;
- all planned rows are accounted for and at least 95% complete.

These are the eligibility limits used by the frozen tuning pilot; they are not
estimated from the new confirmatory data. Report Order MCMC against the same
exact target, but do not require one stochastic method to beat the other.

### Supplementary sample-size check

For balanced-mixed d = 12 networks only, run n/d in {5, 10, 20, 50}, ten
datasets per ratio and four seeds. This is descriptive and shows where
finite-sample graph recovery, as distinct from posterior approximation,
improves.

## V2 -- brain-matched synthetic validation

### Metadata freeze

Before generation, complete `configs/brain_data_profile.template.json` using
metadata only: number and types of variables, independent sampling unit, sample
size per independent unit, subjects/sessions, temporal spacing, missingness,
interventions, and plausible sparsity. Do not inspect OrderSMC results when
choosing these values.

### Core design

The final V2 cells are generated automatically from the frozen profile. Until
that profile is available, the implementation pilot uses the following lean
reference grid:

- dimensions 20, 50, and 100 at n/d = 20, random sparse topology, moderate signal;
- at d = 50, n/d in {5, 10, 20, 50};
- at d = 50 and n/d = 20, random sparse, hub, and modular topology;
- at d = 50 and n/d = 20, weak, moderate, and strong signal.

Duplicated baseline cells are executed once. Each full cell contains twenty
independent graph--dataset pairs and four seeds per stochastic method.

The brain-matched type regime is used throughout this structural grid. Until its
metadata are frozen, balanced mixed is the pilot surrogate. Continuous-dominant
mixed and discrete-dominant mixed are added only at the d = 50, n/d = 20,
random-sparse, moderate-signal baseline. Thus composition is not crossed with
every stress factor.

To preserve the diagnostics used in the DAGBagM paper, the baseline also has
three non-Gaussian continuous-noise variants: Student t with 3 or 5 degrees of
freedom and mean-centred Gamma(shape 1, scale 2), in addition to Gaussian noise.
Binary and categorical children retain their Bernoulli-logistic and softmax
mechanisms, so every such dataset is still heterogeneous. One balanced-mixed
empty-DAG cell at d = 50 and n/d = 20 measures the selected false-edge count.
The resulting pilot design has 13 non-null structural/robustness cells, one
mixed null-graph cell, and two mixed composition-sensitivity cells. Every cell
remains heterogeneous. Cross-type parent--child directions are represented in
every non-null graph where size permits and are guaranteed across each generated
cohort.

The dimension panel supplies the runtime-scaling comparison. Sampling time and
end-to-end time are plotted against dimension on identical allocated hardware;
the number of completed score/search operations is recorded where comparable.
A fixed wall-time budget is the primary fairness constraint. A search-step cap
may be shown only as a separately labelled diagnostic because one step does not
have the same meaning across SMC, MCMC, score search, and flow-network methods.

### Comparisons

- **Same posterior target:** OrderSMC versus Order MCMC.
- **End-to-end posterior formulation:** OrderSMC, Partition MCMC, and REV-MCMC.
- **Restricted mixed comparator:** bnlearn conditional-Gaussian structure
  learning on the CLG-compatible subpanel.
- **Homogeneous adapter controls only:** DAGMA continuous and binary losses.
- **External mixed structure-learning context:** BCDNets and a mixed-score
  DAG-GFlowNet only after independent adapter acceptance; PC/FCI and GES only
  with tests or scores valid for the declared cell.

External methods are compared on recovery, runtime, and failure rate, not on PIP
distance to the order-modular target unless they demonstrably target that same
posterior.

### Decision rules

Candidate complete-family recall must be at least 0.95 in every primary cell;
otherwise the main result is explicitly candidate-limited and the predeclared
wider-pool sensitivity is reported.

OrderSMC is called recovery-competitive with same-target Order MCMC only when
the equal-cell macro-average paired interval satisfies both DAGBagM-aligned
conditions:

- lower 95% bound for skeleton F1(OrderSMC - Order MCMC) >= -0.05;
- lower 95% bound for directed cross-type F1(OrderSMC - Order MCMC) >= -0.05.

AUPRC and Brier comparisons remain secondary posterior-quality results. TPR and
FDR are always shown next to F1 so an apparent gain cannot conceal a material
loss of power or inflation of false discoveries.

It is called faster only if the upper 95% bound for the paired runtime ratio is
below 1. No universal-superiority claim is made from isolated cells.

## V3 -- recognized mixed network benchmarks

### Restricted native mixed benchmark

Use a versioned, checksum-verified definition of bnlearn's `clgaussian.test`.
It contains discrete, Gaussian, and conditional-Gaussian variables and therefore
provides a useful overlap cell in which both OrderSMC and bnlearn's restricted
mixed model are applicable. Sample twenty independent datasets at n in {250,
1,000, 5,000}. This benchmark does not test continuous parents of discrete
children.

Run native bnlearn hill climbing and Tabu with `bic-cg` on this panel. In
addition, run both searches with the serialized OrderSMC local mixed score on
all V3 panels. The latter condition is explicitly labelled **bnlearn search +
OrderSMC score**: it tests whether greedy/tabu point search finds a competitive
graph under the same local score, and is not presented as native bnlearn mixed
modelling or as posterior inference.

### Topology-preserving heterogeneous benchmarks

Use the published Asia, Insurance, ALARM, and Hailfinder DAG topologies. Replace
their native homogeneous conditional distributions with frozen mixed mechanisms
while retaining every directed edge. Type assignment is deterministic and uses
the balanced-mixed regime, with a constrained assignment ensuring that all
feasible cross-type directions occur at cohort level. These results are labelled
**topology-preserving semi-synthetic mixed benchmarks**, never native benchmark
data.

| Topology | Nodes | Sample sizes | Full replicates |
|---|---:|---|---:|
| Asia | 8 | 250, 1,000 | 20 |
| Insurance | 27 | 500, 2,000, 10,000 | 20 |
| ALARM | 37 | 500, 2,000, 10,000 | 20 |
| Hailfinder | 56 | 1,000, 5,000, 20,000 | 20 |

For Asia and `clgaussian.test`, include Exact Order DP when the complete
indegree-bounded candidate pool is feasible. Larger networks use the shared
frozen screened candidate table and report its true-family recall.

### Sachs intervention benchmark

Sachs is deferred until an intervention-aware score adapter passes a tiny-state
acceptance test. The adapter must exclude or modify the local likelihood for a
node in samples where that node is experimentally intervened upon. Pooling those
samples as ordinary observational rows is prohibited.

Primary Sachs evaluation uses skeleton and intervention-identifiable directed
edges relative to a versioned consensus graph. Alternative consensus graphs and
unknown edges are reported as sensitivity analyses, not silently scored as
false positives.

## V4 -- living-brain application

V4 remains blocked until the brain-data profile is complete. The profile decides
which of three valid routes applies:

1. **Independent cross-sectional/trial rows:** use the static DAG model with
   subject/session blocking and a held-out biological validation plan.
2. **Repeated measurements summarized into independent trials:** freeze the
   summary transformation without using network outcomes, then apply the static
   model to trial-level rows.
3. **Raw dependent time series:** implement and validate a lagged/dynamic model
   before causal interpretation. A contemporaneous static fit may be shown only
   as an exploratory conditional-dependence network.

Subjects or animals are biological replicates. Time points within a subject are
not treated as independent replicates. Primary analyses fit subject/session
networks separately or use a prespecified hierarchical aggregation;
indiscriminate pooling is prohibited.

The biological analysis must predeclare its primary contrast, node inclusion,
preprocessing, missing-data rule, temporal handling, forbidden/required edges,
covariates, subject exclusions, and validation evidence. Shuffled-data and
within-subject stability controls are mandatory. Connections are reported with
posterior probabilities and stability, not only a thresholded picture.

In direct analogy with DAGBagM's ovarian-cancer application, V4 reports the
parents and children of each prespecified biological outcome node, their edge
posterior probabilities, and their subject/session resampling frequencies.
Orthogonal validation is matched to the brain endpoint rather than copied from
the cancer study: AUROC/AUPRC and calibration for binary outcomes, macro-F1 and
multiclass log loss for categorical outcomes, RMSE and held-out log predictive
density for continuous outcomes, or hazard ratios only when the endpoint is
genuinely time-to-event. Effect estimates, confidence intervals, multiplicity-
adjusted p-values where tests are used, and held-out validation are reported.
Discovery count alone is not evidence that one method is better.

## Execution order

1. Implement variable-type regimes and verify them with smoke V1.
2. Run full V1 and stop if its posterior-accuracy gate fails.
3. Materialize checksum-locked public benchmark definitions and run V3 pilot.
4. Complete the brain metadata profile independently of algorithm results.
5. Run V2 pilot, confirm resources, then create a new immutable full cohort.
6. Freeze and run full V2 and V3 confirmatory cohorts.
7. Accept or reject the intervention and temporal adapters independently.
8. Freeze V4 and only then inspect the primary living-brain network results.

## Minimum paper result set

- G1 comparator capability/failure matrix over mixed-family motifs.
- V1 exact-posterior table by mixed-composition regime and dimension.
- V2 TPR--FDR and F1 plots for skeleton and directed cross-type recovery across
  dimension, sample size, topology, signal, and distributional robustness.
- Mixed empty-graph false-positive table and runtime-scaling plot.
- V3 benchmark table with native and semi-synthetic results kept separate.
- Across-seed stability and candidate-recall supplement.
- V4 living-brain posterior network, biological validation, negative controls,
  and subject/session sensitivity.

## Comparator scope references

- DAGBagM metric and experiment template:
  https://doi.org/10.1186/s12859-022-04864-y
- DAGMA documentation: https://dagma.readthedocs.io/en/latest/
- bnlearn hybrid-data model: https://www.bnlearn.com/examples/fit/
- bnlearn mixed test network:
  https://www.bnlearn.com/documentation/man/clgaussian-test.html
