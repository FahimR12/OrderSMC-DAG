# Experimental rank-native local scoring

This opt-in robustness candidate does **not** replace the native or restricted
CLG scores. Use `OrderSMCConfig(score_method="rank-native")` or the CLI flag
`--score-method rank-native`. No prior or SMC hyperparameter has been retuned.

## Definition and target

For each continuous variable j, using only the n inference/training rows, set

    z_ij = Phi^{-1}((R_ij - 1/2) / n),

where R is the average rank for ties. By default z is then centered and divided
by its population standard deviation. No noise/jitter is added. Binary and
nominal categorical values retain their original encoding and likelihoods.
The transform is computed once per column, independently of every proposed
parent set, order and SMC seed. It is refitted for each sample-size prefix.

Apply the existing mixed local evidence functions to z and the unchanged
discrete data. The order target remains

    pi(order | z) proportional to
        product_j sum_{P subset predecessors(j)} exp(local_score(j,P;z) - |P|),

for the validation's lambda=1 prior and indegree cap=2. In general use the
configured existing family prior, unchanged. The implicit number-of-compatible-
orders factor in the DAG distribution remains. The exact structure reference
instead counts each DAG once. No order-prior correction is claimed.

These are **plug-in generalized-posterior weights**, not a fully Bayesian
copula rank likelihood and not marginal evidence for the original observations.
The empirical transform's uncertainty is not integrated. For a fixed smooth
invertible transformation, its Jacobian is graph-independent and cancels in
graph comparisons; that observation alone does not make this empirical rank
procedure an exact likelihood. Do not compare normalizing constants across
native and rank-native as original-data Bayes factors. Held-out predictive
density on the original scale is not implemented by this mode.

## What it does and does not guarantee

Strictly increasing marginal transformations preserving ties give identical
normal-score data, family scores and exact targets (up to input rounding).
This specifically prevents a monotone change of measurement scale from
changing the rank-native target. It is not evidence that that target is right.
Population invertible coordinate transforms preserve conditional independence;
empirical tied ranks can discard information and do not establish causality.

Normal marginal distributions do not imply Gaussian conditional regressions,
linear logits or correct nominal-category models. Marginal Gaussianization can
even damage a correctly specified Gaussian regression with mixed parents.
Nonlinear dependence, heavy-tailed conditional noise, confounding, missingness
and nonidentifiable directions are not solved. Report retained-arrow accuracy
alongside direction coverage; an uncertainty mask cannot fix wrong confident
arrows. The mode remains experimental unless independent validation supports it.

## Prespecified development comparison (2026-09-17)

See `experiments/configs/rank_validation_v1.json` and
`scripts/submit_rank_validation.sh`. Settings are fixed before reading recovery.

* Replay all 10 previous graphs under both scores, 2 measurement variants and
  n=250/1000/5000: 120 exact-reference conditions, no redundant SMC fits.
* Fresh pilot: 4 graphs for each of Gaussian and Student-t(df=3) conditional
  noise, each with native and monotone-transformed measurements. Both scores,
  the same three sample sizes and 8 paired inference seeds: 96 conditions and
  768 SMC fits. Noise variants are paired, not 8 wholly independent graph draws.
* Keep 4096 particles, 100 temperatures, 10/25 MH moves, lambda=1, d=8,
  max indegree=2, all candidate families. Use 64 MPI ranks per fresh fit;
  this is acceleration, not a scaling experiment. Replay uses one process.
* Exact order and exact structure references separate sampling, model and
  ordering-prior effects. Recovery summaries first average inference seeds
  within a dataset, then average independent graph replicates. Report all cells,
  including any worsening. Four fresh graphs are a development screen, not a
  publication-sized confirmatory cohort or a noninferiority test.
* Report all-seed identity, pairwise graph/PIP agreement and optional 0.90
  direction masking separately. Identical answers need not be correct.
* Implementation acceptance checks invariance, local-score agreement,
  provenance, MPI operation and probability validity. Recovery is not an
  inclusion gate. No automatic model promotion or metadata-study launch.

The approach is motivated by marginal-transform graphical modeling, not a
claim to reproduce the mixed-data copula posterior of an existing paper:
[Liu, Lafferty and Wasserman (2009)](https://www.jmlr.org/papers/v10/liu09a.html)
study nonparanormal undirected graphs;
[Castelletti (2024)](https://arxiv.org/abs/2312.13168) develops a different Bayesian
copula approach to mixed-data directed networks. Their theoretical guarantees
are not transferred to this plug-in extension.
