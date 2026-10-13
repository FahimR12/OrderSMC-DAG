"""Audit null-score arithmetic, frozen likelihoods, prior and decisions."""
from __future__ import annotations
import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

from ordersmc import OrderSMCConfig
from ordersmc.api import make_scorer
from ordersmc.persistence import load_candidate_table
from ordersmc.graphs import optimal_bayes_action_dag
from . import catchup, reporting_repair
from .execution import _atomic_json
from .readiness import validated_probabilities
from .readiness_audits import pips
from .sampler_corrections import checked_input


def full_support_prior_edges(d,penalty,k=2):
    """Analytic order-modular prior, full parent support; NOT the screened prior."""
    total=0.
    for predecessors in range(d):
        w=[math.comb(predecessors,j)*math.exp(-penalty*j) for j in range(min(predecessors,k)+1)]
        total+=sum(j*x for j,x in enumerate(w))/sum(w)
    return total


def audit(output):
    source=reporting_repair.SOURCE;plan=catchup.checked_source(source/"plan.json")
    result=dict(source_plan_sha256=plan["plan_sha256"],datasets=[],default_changed=False,
        null_metrics_policy="false-edge counts, not F1/AUPRC when no positives exist",
        full_support_prior_illustration=[dict(d=d,penalty=p,mean_edges=full_support_prior_edges(d,p))
            for d in (8,20,196) for p in (1.,math.log(d-1))],
        prior_caveat="illustration assumes complete support; actual screened order prior differs and is data-dependent")
    for d in [d for d in plan["datasets"] if d["cohort"]=="metadata" and d["spec"]["negative_control"]=="empty"]:
        if d["truth_edges"]:raise RuntimeError("empty control has nonempty declared truth")
        table=load_candidate_table(checked_input(d["candidate"]),expected_sha256=d["candidate_sha256"])
        data_path=next(checked_input(a) for a in d["inputs"] if a["path"].endswith("/train.csv"))
        data=pd.read_csv(data_path,float_precision="round_trip")
        metadata=table.provenance["metadata"]["variables"]
        types={node:m["type"] for node,m in metadata.items()};cards={node:m["cardinality"] or 0 for node,m in metadata.items()}
        cfg=OrderSMCConfig.from_mapping(table.provenance["candidate_build_config"]);scorer=make_scorer(data,types,cards,cfg)
        continuous=list(scorer.Xf);corr=np.corrcoef(np.array([scorer.Xf[n] for n in continuous]))
        fit=next(f for f in plan["fits"] if f["dataset_key"]==d["key"] and f["seed_index"]==0)
        record,_=catchup.fit_record(source,plan,fit);maps=[validated_probabilities(pips(s)) for s in record["snapshots"]]
        selected={tuple(e) for s in record["snapshots"] for e in s["representative_edges"]}
        checked=0;error=0.;families=[]
        for child_idx,child in enumerate(table.nodes):
            count=np.count_nonzero(table.parent_sets[child_idx]>=0,axis=1)
            if not np.allclose(table.logprior[child_idx],-cfg.edge_penalty*count,rtol=0.,atol=1e-12):raise RuntimeError("unexpected prior formula")
            for index,parents in enumerate(table.parent_sets[child_idx]):
                names=tuple(table.nodes[int(i)] for i in parents if i>=0)
                # All root families and every retained family containing a
                # selected false edge, plus each child's best-scoring family.
                if names and not any((n,child) in selected for n in names) and index!=int(np.argmax(table.loglik[child_idx]+table.logprior[child_idx])):continue
                score=scorer.score(child,names);diff=abs(score.log_marginal-float(table.loglik[child_idx][index]));error=max(error,diff);checked+=1
                if len(names)==1 and (names[0],child) in selected:
                    null=scorer.score(child,()).log_marginal
                    families.append(dict(child=child,parent=names[0],family=score.family,
                        log_bayes_factor=score.log_marginal-null,penalized_gain=score.log_marginal-null-cfg.edge_penalty))
        if error>1e-7:raise RuntimeError("stored local likelihoods failed recomputation")
        decisions=[]
        for s,m in zip(record["snapshots"],maps):
            directed=optimal_bayes_action_dag(m,d["nodes"],max_indegree=cfg.max_parents,loss="directed_edge_hamming")
            decisions.append(dict(steps=s["final_target_steps"],pair_state_false_edges=len(s["representative_edges"]),
                directed_hamming_false_edges=directed.number_of_edges(),pips_above_point9=sum(v>=.9 for v in m.values()),
                posterior_expected_edges=sum(m.values())))
        result["datasets"].append(dict(dataset=d["dataset_id"],n=len(data),d=len(table.nodes),
            declared_true_edges=0,score_method=cfg.score_method,prior_edge_penalty=cfg.edge_penalty,
            checked_families=checked,maximum_score_recomputation_error=error,
            maximum_abs_rank_normal_correlation=float(np.max(np.abs(corr-np.eye(len(corr))))),
            false_single_parent_evidence=families,decisions=decisions))
    _atomic_json(output/"null_score_audit.json",result)
    print([(d["dataset"],d["checked_families"],d["maximum_score_recomputation_error"]) for d in result["datasets"]],flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--output",required=True,type=Path)
    audit(p.parse_args().output)
