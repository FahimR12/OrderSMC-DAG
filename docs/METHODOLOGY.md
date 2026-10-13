# Statistical model and SMC target

This document defines the probability model targeted by OrderSMC-DAG. It is
part of the public interface: changing the equations for an existing named
score is a change in model, not merely an implementation detail.
Candidate-table provenance records the package version, score-schema version,
and selected score method. A release that changes an existing method must
increment the score schema; adding a separately named method leaves existing
candidate-table targets compatible and is distinguished by the method and
package-version fields.

## Candidate-restricted model space

For each node (i), candidate construction produces a finite collection
\(\mathcal C_i\) of parent sets. The collection may be restricted by
`screen_k`, `max_parents`, rescue settings, and `keep_per_child`. All reported
probabilities are conditional on these retained collections. An excluded
parent set has probability zero.

For scientific analyses, repeat the fit with wider candidate settings and
report the sensitivity of important edges. When the number of variables is
small, use `screen_k = number_of_variables - 1` and `keep_per_child = null` to
retain every parent set allowed by `max_parents`.

## Local family evidence

Each retained family has a log evidence \(\ell_i(P)\) and a structural log
potential

\[
r_i(P) = -\lambda k_i(P) - c\lvert P\rvert,
\]

with the optional tail-edge modification described by the configuration.
The dimension \(k_i(P)\) is determined by the complete model, including
unobserved discrete contexts; it is not estimated from the observed context
count.

The candidate-table attribute containing \(\ell_i(P)\) is still named
`loglik` for checkpoint compatibility. With `score_method: native` it is a
marginal-evidence term; with `score_method: dg` it is a DG-BIC local score;
with `score_method: clg-bic` it is a restricted conditional-Gaussian BIC
score. `logevidence` remains a readable compatibility alias.

### Restricted conditional-Gaussian BIC option

`score_method: clg-bic` is intended when the declared data-generating model is
a restricted conditional-Gaussian (CLG) Bayesian network. It is selected by
the statistical model class, not by a dataset name. A discrete child may have
only discrete parents. A continuous child may have both discrete and
continuous parents. Consequently, continuous-to-discrete edges are outside
this model space and receive no candidate family.

For a discrete child with (r) levels and (q) possible discrete-parent
configurations, let (n_{jk}) be the cell counts and
(n_j=\sum_k n_{jk}). Its maximized local log likelihood and parameter count
are

\[
\widehat\ell_i(P)=\sum_{j,k:n_{jk}>0}n_{jk}\log(n_{jk}/n_j),
\qquad k_i(P)=q(r-1).
\]

For a continuous child with (p) continuous parents, fit a separate Gaussian
linear regression (including an intercept) in every discrete-parent
configuration (j). If that context contains (n_j) observations and has
residual sum of squares (RSS_j),

\[
\widehat\ell_i(P)=-\frac12\sum_j n_j
\left[\log\!\left(2\pi\frac{RSS_j}{n_j}\right)+1\right],
\qquad k_i(P)=q(p+2).
\]

Both cases use

\[
s_i(P)=\widehat\ell_i(P)-\frac12 k_i(P)\log n.
\]

Thus the variance denominator is the context size (n_j), as required for
maximum likelihood, while the BIC penalty uses the total sample size (n).
The score has no fitted prior-scale or score-penalty hyperparameter. It is a
BIC generalized-posterior weight, not an exact marginal likelihood.
Continuous standardization contributes only a child-specific constant and
does not change graph rankings.

A continuous local family is omitted if any observed context has no residual
degrees of freedom, its design is rank deficient, or its residual variance is
numerically zero. This is an estimability restriction: the unregularized
Gaussian maximum likelihood, and hence ordinary BIC, is not finite for those
families. The candidate-build provenance records how many such family requests
were excluded.

### Score-equivalent Degenerate-Gaussian BIC option

Mixed native conditionals (BDeu, Gaussian regression, logistic regression,
and softmax regression) are not one likelihood-equivalent joint model. For a
score-equivalent mixed-data sensitivity analysis, `score_method: dg` implements
Andrews, Ramsey, and Cooper's Degenerate-Gaussian (DG) BIC construction.

Let \(d_j=1\) for a continuous variable and \(d_j=K_j-1\) for a discrete
variable with \(K_j\) categories. A discrete variable is expanded to \(K_j-1\)
indicators, dropping encoded category \(K_j-1\). From the population/MLE
covariance \(\widehat\Sigma\) of the expanded data, define

\[
L(Q)=-\frac{n}{2}\left[d_Q\log(2\pi e)+
\log|\widehat\Sigma_Q|\right], \qquad L(\varnothing)=0.
\]

For child \(j\), parents \(P\), and penalty discount \(c\), the local score is

\[
s_j(P)=L(\{j\}\cup P)-L(P)-\frac{c}{2}d_jd_P\log n.
\]

`dg_penalty_discount: 1.0` gives ordinary BIC. The implementation uses DG for
every local family in the run; mixing DG only for cross-type edges with native
same-type scores would lose score equivalence. For a covered reversal the two
joint-likelihood differences telescope and

\[
d_A d_P+d_B(d_P+d_A)=d_B d_P+d_A(d_P+d_B),
\]

so Markov-equivalent DAGs receive the same DG score. Continuous
standardization and category-reference changes do not affect graph rankings
when `dg_ridge: 0.0`. Exact singular local covariance matrices are rejected;
an explicitly positive `dg_ridge` adds one global diagonal ridge before
submatrices are taken. This retains the telescoping score equivalence but only
approximately retains scale and category-coding invariance.

DG treats expanded category indicators as jointly Gaussian and BIC as a
marginal-likelihood approximation. Resulting edge weights are therefore
pseudo-posterior weights. Report skeletons and compelled CPDAG orientations as
the primary observational result; an arbitrary member of an equivalence class
is not a discovered causal orientation. Version 0.5 provides deterministic
DAG-to-CPDAG conversion for graph summaries. CPDAG conversion does not make the
native mixed score score-equivalent and does not turn reversible orientations
into causal discoveries.

The equivalence statement applies to the local DG score and to the symmetric
parameter/edge penalties. Direction-dependent candidate screening,
`keep_per_child` truncation, or an order-modular prior can still give different
posterior mass to equivalent DAGs. Use complete parent pools when testing score
equivalence, and treat posterior orientation within a CPDAG as
algorithm/prior-sensitive unless external identification assumptions apply.

Reference: Bryan Andrews, Joseph Ramsey, and Gregory F. Cooper (2019),
“Learning High-dimensional Directed Acyclic Graphs with Mixed Data-types,”
Proceedings of Machine Learning Research 104, equations 5--8.

### Discrete children without continuous parents

The default `discrete_score: bdeu` uses the standard BDeu
Dirichlet-multinomial marginal likelihood with equivalent sample size
`alpha`. The optional `discrete_score: bds` reallocates `alpha` over observed
parent configurations. BDs is an empirical-Bayes score and is not generally
score-equivalent at finite sample sizes; results using it should be labelled
accordingly.

### Continuous children

Within every configuration of the discrete parents, a continuous child uses
Bayesian linear regression

\[
y = Zb + \epsilon,\qquad
b\mid\sigma^2 \sim N(0,\sigma^2\tau^2I),\qquad
\sigma^2\sim\operatorname{InvGamma}(a_0,b_0).
\]

The implementation integrates both (b) and \(\sigma^2\) analytically. The
proper prior keeps the evidence finite for singleton and rank-deficient
contexts. Continuous columns are standardized internally by default, making
the coefficient prior comparable across measurement units.

`clg_ig_shape` is \(a_0\) and `clg_ig_scale` is \(b_0\) in the density
proportional to
\((\sigma^2)^{-(a_0+1)}\exp(-b_0/\sigma^2)\). Standardizing a child contributes
the same data-scale Jacobian to every candidate family for that child, so it
does not change posterior comparisons among DAGs; it does make the stated
prior portable across measurement units.

### Discrete children with continuous parents

Binary and categorical children use logistic and softmax regressions with
proper zero-mean Normal priors on every coefficient, including intercepts.
The local evidence is a Laplace approximation at a converged posterior mode.
The categorical model uses a class-symmetric parameterization so changing the
integer labels of outcome classes does not change its evidence.

Their prior standard deviation is `glm_tau` (default 1.0), deliberately
separate from the Gaussian-regression `tau`. In the binary model it is the SD
of a log-odds coefficient. For K>=3, in the class-symmetric softmax it is the
SD in an orthonormal zero-sum contrast basis, so a pairwise class-log-odds
coefficient has SD \(\sqrt{2}\,\mathtt{glm\_tau}\). For K=2, v0.7 uses
contrast variance `glm_tau²/2`, giving exactly the binary-logit prior and
score regardless of the `bin` versus `cat` metadata label.
Reusing `tau: 3.0` creates an extremely
diffuse prior on either log-odds scale and materially changes cross-family
normalization constants.

These logistic and softmax evidences are approximations, not conjugate exact
marginals. The collection of mixed-family scores is not generally
score-equivalent across every reversal. In observational data, directed-edge
support must therefore be interpreted together with Markov equivalence,
domain assumptions, and sensitivity to score hyperparameters; it is not by
itself proof of causal direction.

## Distribution over orders

For an order (O), let \(\mathcal C_i(O)\) contain the retained parent sets
whose members precede child (i). At inverse temperature \(\beta\), the
unnormalized order target is

\[
\gamma_\beta(O)
=
\prod_i
\sum_{P\in\mathcal C_i(O)}
\exp\{r_i(P)+\beta\ell_i(P)\}.
\]

Conditional on the order, parent sets are independent across children and

\[
\Pr(P_i=P\mid O,D)
=
\frac{\exp\{r_i(P)+\ell_i(P)\}}
{\sum_{Q\in\mathcal C_i(O)}\exp\{r_i(Q)+\ell_i(Q)\}}.
\]

This conditional factorization permits exact Rao-Blackwellized edge
probabilities for each sampled order.

The formula above defines the legacy `collapsed_likelihood` path. For that
path the initial SMC proposal is uniform over orders. Because \(\gamma_0\) is not
generally uniform, initial importance weights are proportional to
\(\gamma_0/q_0\). Subsequent stages multiply by
\(\gamma_{\beta_t}/\gamma_{\beta_{t-1}}\), with resampling and invariant
Metropolis moves as configured.

As of v0.7, the high-level `OrderSMCConfig` default uses the **geometric** path
\(\widetilde\gamma_t(O)=\gamma_1(O)^t\), with adaptive temperatures. Thus
\(\widetilde\gamma_0(O)=1\): uniform proposed orders have equal initial weights,
even with asymmetric candidate screening. The final target and conditional
parent-set distribution above are unchanged. An increment has log weight
\((t'-t)\log\gamma_1(O)\). Both serial and MPI execution use this path.
This is an algorithmic bridge change, not a different scientific prior.
The low-level `order_smc` and `order_smc_mpi` functions retain their explicit
legacy fixed-path defaults; pass `tempering_path="geometric"` and
`adaptive_cess_target=0.9` there. Historical experiment JSONs without path
settings retain their legacy fixed path through the execution harness.

### Normalizing constants and evidence-ratio semantics

For \(N\) initial orders sampled from \(q_0(O)=1/d!\), version 0.5 records

\[
\widehat Z_0
=
\frac{1}{N}\sum_{m=1}^N\frac{\gamma_0(O_m)}{q_0(O_m)}
=
\frac{d!}{N}\sum_{m=1}^N\gamma_0(O_m).
\]

The factor \(d!\), which is irrelevant to self-normalized posterior weights,
is therefore restored for the absolute normalizing-constant estimate. At
tempering stage \(t\), with normalized pre-increment weights \(W_{t-1}^{(m)}\),
the recorded log increment is the logarithm of

\[
\widehat R_t
=
\sum_{m=1}^N W_{t-1}^{(m)}
\frac{\gamma_{\beta_t}(O_m)}
     {\gamma_{\beta_{t-1}}(O_m)}.
\]

On the legacy collapsed-likelihood path the reported values have the following
exact software meanings:

- `log_normalizing_constant_estimate` is
  \(\log\widehat Z_0+\sum_t\log\widehat R_t\), an SMC estimate of
  \(\log Z_1\).
- `log_prior_normalizing_constant_estimate` is \(\log\widehat Z_0\).
- `log_evidence_ratio_estimate` is
  \(\sum_t\log\widehat R_t\), estimating \(\log(Z_1/Z_0)\).

On the geometric path, the initial normalizer is exactly \(d!\), not the
structural-prior normalizer. `log_normalizing_constant_estimate` still estimates
\(\log Z_1\), but `log_prior_normalizing_constant_estimate` and
`log_evidence_ratio_estimate` are unavailable (NaN in Python, null in JSON).
`log_path_normalizer_ratio_estimate` records \(\log(Z_1/d!)\). It must not be
interpreted as prior-relative marginal evidence. MPI now records these
normalizer diagnostics too. Log estimates are not claimed to be unbiased.

These quantities refer to the candidate-restricted, order-modular target. A
candidate pool selected using the same observed data is part of the fitted
procedure, so this ratio must not be presented as an unrestricted marginal
likelihood or used to compare runs with different candidate spaces. Native
logistic/softmax families also use Laplace evidence approximations, and DG-BIC
produces a pseudo-evidence ratio rather than a finite-sample Bayesian marginal
likelihood.

### Tempering, proposals, and posterior snapshots

The high-level default is adaptive geometric tempering. It selects the
largest next beta whose conditional ESS fraction

\[
\frac{(\sum_m W_m u_m)^2}{\sum_m W_m u_m^2},
\qquad
u_m=\gamma_{\beta'}(O_m)/\gamma_\beta(O_m),
\]

meets `adaptive_cess_target`, subject to a progress guard and maximum stage
count. On the geometric path CESS is monotone in the increment; if the
CESS-preserving step is smaller than the progress guard, sampling fails rather
than forcing a collapsing step. MPI adaptive scheduling requires the geometric
path. The legacy local collapsed-path adaptive heuristic remains available
for development, without a global monotonicity guarantee.
Resampling can be disabled explicitly with `resampling: none` (or
the legacy-equivalent ESS threshold zero). Both local and MPI MH kernels mix global
swaps, adjacent swaps, and segment reversals; each is symmetric, and proposed
and accepted counts are recorded separately by move type.

`final_target_snapshot_steps` never records PIPs at an intermediate
temperature. Every snapshot has beta exactly one. Step zero means immediately
after the ordinary beta-one rejuvenation; positive values request that many
additional proposals **per particle**, not complete sweeps over all variables,
at the final target. Local and MPI backends both support these snapshots. Consequently these
snapshots can be used as posterior-accuracy checkpoints, whereas rows from the
annealing path cannot.

Local SMC can be checkpointed after completed tempering stages and requested
final-target snapshots. A resumable checkpoint contains the particle orders,
normalized and log weights, current targets and beta, complete lineage and
history, evidence accumulators, best visited state, and NumPy generator state.
The candidate-table content digest and sampling configuration are checked on
load. This stage-boundary checkpointing is currently local-only; candidate
tables remain separately reusable by both local and MPI samplers.
The gzip header is frozen, so serializing the same checkpoint state to two
paths yields identical bytes. Runtime measurements are still recorded, but a
separate `inferential_replay.json` excludes them and content-addresses the
orders, weights, PIPs, deterministic history and proposal counters, evidence
estimates, resource counters, and candidate-table identity.

## Order-modular interpretation

The target is intentionally **order modular**. Its joint distribution is

\[
\Pr(O,G\mid D)\propto
\mathbf 1\{O\text{ is topological for }G\}
\prod_i \exp\{r_i(P_i)+\ell_i(P_i)\}.
\]

Therefore the marginal mass of a DAG contains a factor equal to its number of
compatible topological orders (linear extensions). This is not the same as a
structure-modular posterior that counts every DAG once. Output metadata records
this semantic choice as `candidate_restricted_order_modular`.

## Output interpretation

- Edge probabilities are posterior inclusion probabilities under the retained
  candidate space and order-modular model.
- `best_edges` is the highest joint-score DAG found among the visited orders;
  it is a visited joint graph/order mode, **not** the marginal-posterior MAP DAG.
  Maximizing the latter would also require the graph's linear-extension factor.
  Exact
  local-family score ties choose the lexicographically smallest canonical
  `(parent_name, child_name)` edge tuple, so serialization row order cannot
  change this summary.
- A thresholded acyclic summary graph is a deterministic marginal-probability
  summary, not a posterior draw.
- `marginal_utility_dag()` globally minimizes posterior expected pair-state
  SHD under acyclicity and the requested indegree bound. Its three states per
  unordered pair are no edge and the two orientations, so a reversal costs
  one. `loss="pair_state_log_probability"` is a labelled product-marginal MAP
  sensitivity and is not the SHD Bayes action.
- Weight ESS measures weight imbalance. It does not measure resampling
  genealogy or all Markov-chain dependence.
- Per-run particle standard errors are heuristic. Independent SMC seeds and
  candidate-width sensitivity runs are the primary uncertainty checks.
  `edge_confidence.csv` labels the MCSE uncalibrated and leaves the legacy
  `MC95Lower/Upper` columns blank. `seed_consensus_table` reports between-run
  SD and the SE of the seed-averaged estimator, with an approximate Student-t
  interval. This is not a posterior credible interval and cannot detect bias
  shared by all runs. No ESS-based convergence certificate is emitted.
- Annealing-path diagnostics with beta below one are not posterior PIP
  estimates. Only the final result and explicitly labelled beta-one snapshots
  have posterior semantics.

## Method references

- Friedman, N. and Koller, D. (2003), *Being Bayesian About Network
  Structure*.
- Del Moral, P., Doucet, A. and Jasra, A. (2006), *Sequential Monte Carlo
  Samplers*.
- Scutari, M. (2016), *An Empirical-Bayes Score for Discrete Bayesian
  Networks*.
- Geiger, D. and Heckerman, D. (1994), *Learning Gaussian Networks*.
