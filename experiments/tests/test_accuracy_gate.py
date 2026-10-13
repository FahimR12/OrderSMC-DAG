import numpy as np
import pandas as pd
import pytest

from ordersmc_experiments.accuracy_gate import dataset_rmse_gate


def rows():
    return pd.DataFrame([dict(dataset_key=f"d{d}", seed_index=s, cell_id="mixed", budget="baseline",
        final_target_steps=0, method="ordersmc-v07", role="evaluation", own_target_pip_rmse=e)
        for d, errors in enumerate([[0., .024], [.01, .01], [.01, .01]]) for s, e in enumerate(errors)])


def test_quantile_is_over_dataset_mean_rmse_not_seeds():
    report = dataset_rmse_gate(rows(), dict(mean=.012, p95=.020))
    flag = report["flags"][0]
    assert flag["n_datasets"] == 3 and flag["n_runs"] == 6
    assert flag["mean"] == pytest.approx(np.mean([.012, .01, .01]))
    assert flag["p95"] == pytest.approx(np.quantile([.012, .01, .01], .95))
    assert flag["per_seed_p95_diagnostic"] > .020
    assert report["all_passed"] is True


def test_extension_seeds_do_not_change_primary_gate():
    frame = rows()
    extension = frame.iloc[[0]].assign(seed_index=8, role="reproducibility-extension", own_target_pip_rmse=1.)
    assert dataset_rmse_gate(pd.concat([frame, extension]), dict(mean=.012,p95=.02)) == dataset_rmse_gate(frame, dict(mean=.012,p95=.02))


def test_gate_no_reference_or_insufficient_datasets_is_not_a_pass():
    assert dataset_rmse_gate(rows().assign(own_target_pip_rmse=np.nan), dict(mean=.012,p95=.02))["all_passed"] is None
    report = dataset_rmse_gate(rows().iloc[:2], dict(mean=.012,p95=.02))
    assert report["all_passed"] is None
    assert report["flags"][0]["status"] == "insufficient-datasets"


def test_gate_rejects_duplicates_and_infinities():
    frame = rows()
    with pytest.raises(ValueError, match="duplicate"):
        dataset_rmse_gate(pd.concat([frame, frame.iloc[:1]]), dict(mean=.012,p95=.02))
    with pytest.raises(ValueError, match="finite"):
        dataset_rmse_gate(frame.assign(own_target_pip_rmse=np.inf), dict(mean=.012,p95=.02))
