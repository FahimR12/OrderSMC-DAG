# Direction-recovery audit (development diagnostic)

Entry point: `bash scripts/submit_orientation_audit.sh`.
Default output: `results/orientation-audit/v1`.

All three fresh eight-variable signal datasets from the completed null/prior
pilot are included under both priors. The data, candidate support and saved SMC
fits are immutable. Every case is processed; no best seed is selected.

The audit separates four questions:

1. **Sampler:** four Order-MCMC seeds (20,000 iterations, 5,000 burn-in, thinning
   five), compared with four saved SMC seeds and the exact order posterior.
   This is a target/recovery diagnostic, not an equal-compute competition.
2. **Score and prior:** exact structure-modular reference from the positive
   log-space partition recurrence; and an exact order sensitivity removing
   only the continuous marginal rank transform, with unchanged support/prior.
   Within the eight collider-free orientations of the known chain, record
   local likelihood, local prior, and linear-extension multiplicity separately.
   Those conditional weights are NOT a posterior over the unrestricted graph
   space, and the known skeleton is not supplied to fitted methods.
3. **Representative graph:** keep pair-state Hamming loss primary; show directed
   edge Hamming loss as a labelled sensitivity, not a replacement chosen by F1.
4. **Identifiability:** report skeleton, directed F1, reversals on recovered
   adjacencies, v-structures and CPDAG pair-state SHD. This chain has no compelled
   arrows under generic observational Markov equivalence. Parametric mixed
   families may distinguish directions, so this does not excuse confident
   incorrect orientation; record reverse PIPs >= 0.9 separately.

Comparators include bnlearn HC and Tabu with identical local scores/support,
but their point-score objective differs from the order posterior's Bayes action.
Native bnlearn BIC-CG is an explicitly restricted-support sensitivity: it cannot
include continuous-to-discrete edges present in this unrestricted generator.
Point-graph 0/1 values are never called posterior confidence. The historical
comparison uses all matched datasets and averages seeds within each dataset;
the five metadata M1 datasets are kept separate from negative controls.

Primary outputs: `analysis/summary.csv`, `edge_directions.csv`, and
`historical_comparison.csv`; complete per-case decompositions are in
`runs/<case>/result.json`. Completion requires all six cases. Source hashes,
source artifact hashes, configs, seeds and job IDs are retained. No default
score, prior, metric or sampling setting is changed by this audit.

Background: observational Markov equivalence is determined by skeleton and
unshielded colliders (Chickering, 1995,
https://maxchickering.com/publications/uai95.pdf). This is an equivalence-class
diagnostic, not a claim of identifiability under every restricted parametric
model. bnlearn's own mixed custom-score example documents the distinction
between unrestricted custom scores and conditional-Gaussian support:
https://www.bnlearn.com/examples/custom-score/.
