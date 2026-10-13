# Changelog

## Unreleased

## 0.6.0 - 2026-09-14

- Added an explicit hyperparameter-free restricted conditional-Gaussian BIC
  score (`score_method="clg-bic"`) with multinomial discrete conditionals and
  context-specific Gaussian regressions.
- Enforced the restricted-CLG model space: continuous-to-discrete edges are
  inadmissible, and non-estimable local Gaussian families are excluded with an
  auditable candidate-build count.
- Added a held-out V3 CLG corrective-validation plan and restart-safe Slurm
  launchers while preserving the legacy native mixed score and its artifacts.
- Added mathematical, provenance, candidate-support, plan, and configuration
  tests for the new score.

- Added executable, candidate-restricted Partition MCMC and REV-MCMC adapters,
  with exhaustive tiny-state normalizer, PIP, detailed-balance, and stationarity
  acceptance checks.
- Added a checked adapter-validation report and made full score plus adapter
  validation a mandatory launch dependency.
- Added resumable E1--E4 Slurm workflows, deterministic plans, atomic dataset
  preparation, raw result ledgers, dataset-level bootstrap summaries, explicit
  failure rows, E3 budget-point references, and E4 scaling summaries.
- Integrated the modular E3 topology generator with the mixed SEM generator.

## 0.5.0 - 2026-08-07

- Added explicit SMC estimates of the candidate-restricted order-target
  normalizer `log Z_1`, the beta-zero baseline `log Z_0`, and their evidence
  ratio `log(Z_1/Z_0)`, including the uniform-order `n!` proposal factor.
- Added adaptive conditional-ESS tempering, an explicit no-resampling mode,
  configurable symmetric order-proposal mixtures, and per-proposal acceptance
  diagnostics.
- Added posterior snapshots that are produced only at beta one, with optional
  additional final-target MH sweeps; intermediate tempered PIPs are not exposed
  as posterior snapshots.
- Added atomic, resumable local-SMC checkpoints containing particles, normalized
  and log weights, beta position, lineage, diagnostics, evidence accumulators,
  RNG state, best visited DAG, and final-target snapshots.
- Froze checkpoint gzip headers for byte-identical serialization of the same
  state, and added a timing-independent inferential replay digest covering
  particles, PIPs, counters, evidence estimates and candidate identity.
- Replaced new candidate-table pickle output with canonical JSON in deterministic
  gzip, including a row-order-independent content address, strict expected-hash
  loading, complete-pool acceptance validation and trusted legacy-v1 migration.
- Added result persistence for posterior snapshots and protocol-focused tests.
- Added a globally optimized acyclic, indegree-constrained marginal Bayes-action
  DAG under pair-state SHD (so a reversal costs one), plus labelled
  product-of-three-state-marginals and independent-edge sensitivity modes.
- Added deterministic DAG-to-CPDAG conversion using compelled/reversible edge
  labelling, validated against brute-force Markov-equivalence classes.
- Added a provenance-bound posterior-DAG pushforward from collapsed order
  particles and a Rao--Blackwellized posterior graph-size distribution.
- Added complete candidate-build, target-evaluation, resume-validation,
  best-DAG, posterior-summary and total local-score accounting for fair-resource
  experiment records.
- Made best-visited-DAG selection independent of candidate-table row order:
  exact local-family score ties use the lexicographically smallest canonical
  parent-edge tuple, and global score ties use a documented canonical key.

## 0.4.0 - 2026-08-03

- Added an explicit global Degenerate-Gaussian BIC score for mixed data using
  the score-equivalent construction of Andrews, Ramsey, and Cooper (2019).
- Added `score_method: dg`, ordinary-BIC `dg_penalty_discount: 1.0`, and an
  opt-in `dg_ridge` for singular local covariance matrices; the existing
  heterogeneous marginal scores remain available as `score_method: native`.
- Split the standardized logistic/softmax prior scale into `glm_tau` (default
  1.0) instead of reusing the Gaussian-regression `tau` (default 3.0).
- Embedded the categorical expansion, covariance convention, penalty, ridge,
  and score interpretation in candidate provenance.
- Added hand-calculation, covered-reversal, category-relabeling, singularity,
  deterministic-duplicate, and regularization tests for DG-BIC.

Candidate score schema 2 requires candidate tables from earlier releases to be
rebuilt.

## 0.3.0 - 2026-08-03

- Corrected serial and MPI SMC initialization for a uniform order proposal by
  weighting particles with the generally nonuniform prior target `gamma_0`.
- Replaced maximized Gaussian scores with proper Normal--Inverse-Gamma marginal
  likelihoods and made continuous-column standardization the default.
- Added proper intercept priors and robust convergence checks to binary and
  class-symmetric categorical Laplace scores.
- Made standard BDeu the default discrete score; BDs remains an explicit
  empirical-Bayes option.
- Corrected best-DAG selection to use the highest joint-score graph found and
  labelled particle edge MCSEs as genealogy-ignoring heuristics.
- Added resampling genealogy, order-diversity, ESS, and acceptance diagnostics.
- Added lossless string/boolean/category encoding, ambiguity checks for
  numeric-looking category labels, and explicit raw/encoded input modes.
- Embedded data, metadata, score, prior, and candidate-screening provenance in
  candidate tables and reject incompatible checkpoint reuse.
- Added exact small-network SMC tests, score invariance/stability tests, a
  two-rank MPI CI smoke test, and matched old-versus-corrected simulations.

Candidate tables built by 0.2 contain the old family scores and do not have
verifiable provenance. Rebuild them for 0.3 analyses; only use the legacy
checkpoint override for a deliberately preserved historical comparison.

## 0.2.0 - 2026-07-10

- Added the `ordersmc` command with inspect, fit, checkpoint, resume, summarize,
  and multi-seed workflows.
- Added JSON/YAML metadata and configuration files with quick, standard, and
  thorough presets.
- Added validated local/MPI APIs and early input checks.
- Added candidate-table persistence and reproducible result folders.
- Added directed, skeleton, orientation, entropy, Monte Carlo, and multi-seed
  edge-confidence diagnostics.
- Added ESS and MH acceptance diagnostics with actionable warnings.
- Added workflow, checkpoint, CLI, confidence, and MPI smoke tests.

## 0.1.0 - 2026-07-10

- Initial standalone extraction of pair-rescue Order-SMC and its MPI backend.
