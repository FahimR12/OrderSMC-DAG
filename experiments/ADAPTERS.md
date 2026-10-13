# Algorithm adapters and acceptance contract

The active experiment harness contains four executable posterior-inference adapters and two executable `bnlearn` score-search controls. The posterior samplers consume the same serialized candidate support and the same local mixed-data log-evidence values. Order-target and structure-target priors may have different frozen edge penalties; retargeting changes only the structural log potential, never the local likelihood score.

| Adapter | State | Declared target | Acceptance evidence |
|---|---|---|---|
| Exact Order DP | subsets/orders | candidate-restricted order-modular | log normalizer agrees with direct order enumeration |
| Order MCMC | total orders | candidate-restricted order-modular | enumerated proposal symmetry, detailed balance, and stationarity |
| Partition MCMC | ordered source partitions | candidate-restricted structure-modular | partition normalizer and PIPs equal Exact Structure; split/merge MH satisfies detailed balance |
| REV-MCMC | DAGs | candidate-restricted structure-modular | ordinary add/delete/reverse kernel is invariant; orphan-and-resample reversal has conjugate detailed balance |
| bnlearn HC/Tabu, native | DAGs | BIC conditional-Gaussian score | accepted only on the CLG-compatible `clgaussian.test` panel; R 4.4.3 and bnlearn 5.2.1 are pinned |
| bnlearn HC/Tabu, shared score | DAGs | point estimate maximizing the serialized OrderSMC local mixed score | tiny mixed table completes through the R subprocess adapter; unsupported candidate families receive negative-infinite local score |

Partition MCMC uses the unique iterative-source ordered partition of each DAG. For a node in block `k>0`, allowed parent sets are subsets of earlier blocks containing at least one parent in block `k-1`; block-zero nodes have the empty parent set. Rao--Blackwellization over allowed families produces edge PIPs.

REV-MCMC mixes ordinary add/delete/reverse proposals with the published reversal move: select an existing edge, orphan both endpoints, sample the reversed-edge endpoint family, then sample the other endpoint family. Its Hastings ratio includes both sequential family normalizers and the forward/reverse numbers of selectable edges. The default reversal probability is `1/15`, matching the published experimental setting.

Run the executable acceptance report before inference:

```bash
ordersmc-experiments validate-adapters --strict --output adapter-validation.json
```

The report exhaustively checks a complete three-node candidate space (25 DAGs and 13 positive-mass ordered partitions). The test suite also runs both stochastic adapters and checks finite, normalized PIPs.

References:

- Kuipers and Moffa (2017), *Partition MCMC for Inference on Acyclic Digraphs*, DOI `10.1080/01621459.2015.1133426`.
- Grzegorczyk and Husmeier (2008), *Improving the structure MCMC sampler for Bayesian networks by introducing a new edge reversal move*, DOI `10.1007/s10994-008-5057-7`.

DAG-GFlowNet and BCDNets remain optional, unavailable adapters. They are not silently substituted. Native bnlearn results are never reported outside its conditional-Gaussian model class, in which discrete children cannot have continuous parents. The shared-score condition is labelled `bnlearn search + OrderSMC score`; it is a search control, not native bnlearn mixed modelling.

The V3 external-adapter preflight is executable with:

```bash
python -m ordersmc_experiments.bnlearn_adapter --strict --output bnlearn-validation.json
```
