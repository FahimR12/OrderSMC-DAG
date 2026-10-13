"""Protocol exact-PIP gate: average seed RMSE within data BEFORE quantiles."""
from __future__ import annotations

import numpy as np
import pandas as pd


def dataset_rmse_gate(frame, limits, *, method="ordersmc-v07",
                      group_columns=("cell_id", "budget", "final_target_steps")):
    """Do not substitute RMSE of mean PIPs or quantiles over individual seeds.

    r_ds = sqrt(mean_e((p_hat_dse - p_exact_de)**2));
    r_d = mean_s(r_ds); gate uses mean_d(r_d) and Q_.95({r_d}).
    The per-seed quantile is retained as a separate, non-gating diagnostic.
    Complete expected seed coverage must be checked by the calling workflow.
    """
    report = dict(schema="ordersmc-dataset-exact-rmse-gate-v2", method=method,
        statistical_unit="dataset", seed_aggregation="arithmetic-mean-of-per-seed-RMSE",
        quantile_method="linear", mean_limit=float(limits["mean"]), p95_limit=float(limits["p95"]),
        minimum_datasets=3, flags=[], all_passed=None)
    if frame.empty:
        return report
    selected = frame[frame.method == method].copy()
    if "role" in selected:
        selected = selected[selected.role != "reproducibility-extension"]
    selected = selected[selected.own_target_pip_rmse.notna()]
    if selected.empty:
        return report
    values = selected.own_target_pip_rmse.to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("exact-reference RMSE must be finite and nonnegative")
    identity = [*group_columns, "dataset_key", "seed_index"]
    if selected.duplicated(identity).any():
        raise ValueError("duplicate seed identity in accuracy gate")
    for key, group in selected.groupby(list(group_columns), dropna=False, sort=True):
        key = key if isinstance(key, tuple) else (key,)
        per_dataset = group.groupby("dataset_key", sort=True).own_target_pip_rmse.mean()
        errors = per_dataset.to_numpy(dtype=float)
        mean, percentile = float(errors.mean()), float(np.quantile(errors, .95, method="linear"))
        evaluable = len(errors) >= report["minimum_datasets"]
        passed = bool(mean <= limits["mean"] and percentile <= limits["p95"]) if evaluable else None
        report["flags"].append(dict(zip(group_columns, key), n_datasets=len(errors), n_runs=len(group),
            mean=mean, p95=percentile, maximum=float(errors.max()), passed=passed,
            status="pass" if passed is True else "fail" if passed is False else "insufficient-datasets",
            per_seed_p95_diagnostic=float(np.quantile(group.own_target_pip_rmse, .95, method="linear")),
            dataset_mean_rmse={str(k): float(v) for k, v in per_dataset.items()}))
    passed_values = [f["passed"] for f in report["flags"]]
    report["all_passed"] = (False if False in passed_values else
                            None if not passed_values or None in passed_values else True)
    return report
