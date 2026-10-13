# Section 2 — Metadata-calibrated synthetic validation

## Scope

Section 2 tests known-truth DAG recovery in statistical regimes calibrated to
the six privacy-preserving Living Brain profiles. It is distinct from Section
1 (generic synthetic correctness) and Section 3 (participant-level biological
application).

The frozen profile priority is protein pathway, RNA pathway, protein molecule,
RNA molecule, joint molecule, then joint pathway.

The generator imports only aggregate distributional targets. Every DAG is an
independent seeded draw. Participant values, empirical variable names, fitted
edges, directions, and Markov blankets are prohibited as sources of truth.

## Panels

- **M0 calibration fidelity:** target-versus-generated marginal, correlation,
  covariate-association, and spectrum summaries. Recovery is not used to tune
  the generator.
- **M1 matched recovery:** observed sample size and dimension, real-like
  marginals, modular truth, maximum indegree two.
- **M2 realistic-distribution penalty:** Gaussian and metadata-calibrated data
  share the graph, base coefficients, and random streams.
- **M3 information curve:** half, matched, and double observed sample size use
  the same graph and nested per-node random streams.
- **M4 CLG-compatible sensitivity:** use both a conditional-Gaussian-compatible
  orientation regime and affine, Gaussian continuous mechanisms. Contrast this
  with the unrestricted metadata-calibrated arm as an end-to-end sensitivity,
  not an isolated representation advantage. Directions and distributions both change.
- **M5 reproducibility and controls:** eight OrderSMC seeds on the same M1
  datasets, plus empty-graph and PD-isolated shuffled-null controls. The latter
  removes PD-incident edges before generation and then independently permutes
  PD; it is not a permutation of the original M1 cohort.

The first four M1 OrderSMC seeds are the evaluation set used for fair M1--M3
comparisons. Four additional seeds are used only for the eight-seed M5
stability analysis; they cannot improve the reported M1--M3 recovery estimate.

Unsupported algorithm/profile combinations are reported as structurally
unsupported, not as zero accuracy. Native bnlearn conditional-Gaussian runs
appear only in M4. Shared-score bnlearn searches use the OrderSMC local score
and are labelled accordingly.

The primary cross-sectional state template is age -> PD state -> molecular
measurements. Sex remains exogenous: its weak observed association with PD is a
calibration target rather than a forced direct edge. It is a device for
reproducing the reported associations and is not interpreted as a causal model
of disease onset. M4 reverses the age--disease ordering and disables nonlinear
quantile transforms. Changing directions alone does not establish a Gaussian
likelihood. Version 3 uses `clg-bic` for all shared-score methods on this M4
arm; unrestricted and metadata-transformed arms use `rank-native` scoring.
Rank-native transforms continuous columns using deterministic midrank normal
scores within each fitting sample, leaving categories unchanged. It defines a
plug-in generalized posterior, not an integrated rank likelihood or evidence
on the original measurement scale. Historical `native` results remain separate.
Native bnlearn retains its own BIC and structural-prior conventions and is
labelled an external baseline, not an identical-posterior comparison.

## Replication and inference

Full M1 would have 20 independent DAG/data pairs per profile; secondary panels
would have 10 per included profile. That phase remains locked. The protein
pilot has five M1 pairs and three per secondary condition: 23 datasets and
226 fits in total. Each has 196 variables, with 222 training observations
(111/444 for the M3 information curve) and an independent equally sized test
cohort. No participant-level observations are used.

Seeds are nested computational repeats, not independent datasets. The primary
posterior graph summary averages PIPs across evaluation seeds within a dataset
and constructs its Bayes-action representative graph once; its recovery is not
the average of the per-seed F1 values. Per-seed metrics are also retained.
Dataset-level paired bootstraps use independent graph/data pairs as units.
M5 separately reports directed, skeleton and fixed-threshold uncertainty-display
stability, without presenting uncertain directions as reproducibly identified.

The accepted resource policy is 4096 particles, 100 temperatures, 10 MH moves
normally and 25 after resampling, ESS fraction 0.5 and lambda 1. It uses 64 MPI
ranks per SMC fit, at most four simultaneous SMC fits. The 16384-particle,
20/50-move budget remains a high-precision reference rather than the primary
pilot setting. See `configs/rank_acceptance_freeze_v1.json` and the accepted
publication addendum. Order MCMC and shared-score bnlearn HC/Tabu provide the
unrestricted baselines; native bnlearn HC/Tabu is included only in M4.

Female prevalence is absent from the supplied summaries. The explicit 0.5
assumption remains, with metadata-fidelity sensitivity checks at 0.4 and 0.6.
These hold the selected scale fixed and do not select inference settings.
If a prevalence count is supplied after freezing, it requires a separately
labelled revision rather than silently changing a running study.

## Finite-sample metadata calibration

The earlier generator compared statistics from a 2000-observation construction
sample against metadata measured in 222 people. Version 3 keeps that construction
sample for the mechanisms, but selects the same one-dimensional scale grid using
the mean summaries of eight disjoint samples at the metadata cohort size (222).
The original metadata-only loss is unchanged. Calibration always uses that
cohort size, including the M3 half/double-n arms, so paired mechanisms stay fixed.
Selection samples are separate from training, testing and validation streams.

A metadata-only development check found a second mismatch: the old template
gave direct age/sex-to-protein edges generic molecular-effect magnitudes
(uniform 0.30--0.65), despite their much weaker reported associations, while
direct PD effects used magnitude 0.17. The new `metadata-ratio-v1` policy multiplies
the sampled age/sex-to-protein coefficients by
`(0.17 / 0.475) * (mean_abs_association_with_covariate / mean_abs_association_with_PD)`.
Here 0.475 is the old uniform mean. This retains the graph, signs and relative
coefficient variation; it does not change age--PD or molecular-to-molecular edges.
These marginal associations motivate a synthetic scale convention, not estimates
of causal coefficients. The unchanged global-scale grid is then calibrated.
The initial finite-sample-only attempt failed the sex-association tolerance on
two of five graphs; this development history must be retained, not described as
an untouched validation set. No graph-recovery result has been examined or used
to choose this policy.

For each of the five M1 graphs, a further 50 independent size-222 draws audit the
selected mechanisms. Tolerances are unchanged from `metadata_analysis.py`.
Both absolute and relative discrepancies and their Monte Carlo standard errors
are reported. A systematic miss is defined as
`max(0, abs(mean - target) - 1.96 * MCSE) / divisor > tolerance`, where `divisor`
is the absolute target for effective rank and the 80%-variance component count,
and one otherwise. This diagnoses discrepancies after simulation uncertainty;
it is not an equivalence test or an uncertainty interval for the original cohort.
The gate requires no systematic misses on any primary graph. It also reports
whether the raw mean is inside each tolerance. No graph is excluded or redrawn
to pass; no recovery metric enters calibration. A failed gate stops inference.
The ordinary M0 audit additionally reports the actual train/test cohort summaries.

## Staged execution

Smoke and protein pilot are exploratory. Full six-profile execution is locked
until the protein pilot passes integrity and candidate recall, produces the M0
audit, establishes seed stability, and freezes generator tolerances, candidate
settings, inference resources, and a fresh confirmatory seed.

Machine-readable design: `configs/section2_metadata_v3.json`; execution:
`configs/execution_section2_pilot_v3.json`. Versions 1 and 2 and all old results
remain historical artifacts and must not be pooled with this new score/calibration
edition. New submissions use `results/section2-v3/seed-20260923-pilot`.

Run `bash scripts/submit_section2_v3_pilot.sh`. It activates the environment,
archives the complete execution tree and aggregate metadata into the new output
root, and submits calibration and score/adapter preflight, preparation, then a
gated inference controller. Prepared M1 manifests must match the exact audited
mechanisms; every candidate table must pass integrity checks, and each cell's
mean directed-parent and complete-family recall must meet the existing 0.95
requirement. Individual lower recalls remain visible and are not excluded.
Workers run from the archived tree, not the mutable working directory. Dependency
completion uses `afterany` with explicit validation, so a failed prerequisite is
reported rather than leaving permanently unsatisfied dependencies.

`submission_pipeline.json` records the stage jobs; calibration reports are in
`calibration_validation/` and the gate in `validation/calibration-gate.json`.
Inference job IDs are recorded separately in `submission_inference.json`.
Logs live in `code_snapshot/logs/`; final panel summaries go to
`analysis/section2/`. The earlier score-transition, structure-target mixing,
rank-native and extended-budget audits are retained as development evidence.
This pilot is still exploratory, not untouched confirmatory validation.
