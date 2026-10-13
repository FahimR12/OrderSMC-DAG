"""Read-only scientific inputs; new diagnostic artifacts for reporting/null checks."""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import networkx as nx
import numpy as np

from . import catchup
from .execution import _atomic_json
from .readiness import summarize_posterior, validated_probabilities
from .readiness_audits import pips


def reproduce(root, output):
    plan = catchup.checked_source(root / "plan.json")
    for d in [d for d in plan["datasets"] if d["cohort"] == "v3_mixed"]:
        groups = defaultdict(list)
        for f in [f for f in plan["fits"] if f["dataset_key"] == d["key"]]:
            record, _ = catchup.fit_record(root, plan, f)
            groups["ordersmc-v07"].append(pips(record["snapshots"][0]))
            groups["ordersmc-original"].append(pips(catchup.located(f["original"])))
        for c in [c for c in plan["comparisons"] if c["dataset_key"] == d["key"]]:
            groups[c["method"]].append(pips(catchup.located(c["original"])))
        for method, maps in groups.items():
            keys = sorted(set().union(*maps))
            mean = validated_probabilities({e: float(np.mean([m.get(e, 0.) for m in maps])) for e in keys})
            truth = nx.DiGraph(); truth.add_nodes_from(d["nodes"]); truth.add_edges_from(d["truth_edges"])
            try:
                summarize_posterior(mean, truth, d["nodes"], d["max_parents"])
            except RuntimeError as error:
                positive = nx.DiGraph(); positive.add_nodes_from(d["nodes"])
                positive.add_edges_from((u,v) for (u,v),v0 in mean.items()
                    if v0 > max(0.,1.-mean.get((min(u,v),max(u,v)),0.)-mean.get((max(u,v),min(u,v)),0.)))
                record = dict(source_plan_sha256=plan["plan_sha256"], dataset_key=d["key"], method=method,
                    nodes=d["nodes"], max_indegree=d["max_parents"], truth_edges=d["truth_edges"],
                    edge_probabilities=catchup._all_edges(d["nodes"],mean), error=str(error),
                    component_sizes=sorted(map(len,nx.strongly_connected_components(positive)),reverse=True))
                destination=output/"failing_mean_pip_case.json"
                if destination.exists():
                    raise FileExistsError(destination)
                _atomic_json(destination,record)
                print("REPRODUCED",d["key"],method,record["component_sizes"],str(error),flush=True)
                return
        print("checked",d["dataset_id"],flush=True)
    raise RuntimeError("original failure not reproduced")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    args=p.parse_args();reproduce(args.source,args.output)


if __name__=="__main__":main()
