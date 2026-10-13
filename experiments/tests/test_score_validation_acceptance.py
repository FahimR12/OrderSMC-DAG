from __future__ import annotations

import json

import ordersmc_experiments.cli as cli_module
import ordersmc_experiments.score_validation_v2 as validation_module
from ordersmc_experiments.score_validation import (
    FIXTURE_MATRIX_SHA256,
    audit_fixture_matrix,
    run_branch_boundary_audit,
    run_e0a_full_validation,
    run_e0a_score_validation,
    run_error_policy_checks,
    run_invariance_checks,
    run_structural_fixture_matrix,
)


def test_static_fixture_matrix_is_canonical_hashed_and_covers_all_axes() -> None:
    audit = audit_fixture_matrix()
    assert audit.matrix_sha256 == FIXTURE_MATRIX_SHA256
    assert audit.complete
    assert audit.expanded_fixture_count == 72
    assert len(audit.resolved_fixture_hashes) == 72
    assert set(audit.sample_sizes) == {8, 20, 100}
    assert set(audit.child_types) == {
        "continuous",
        "binary",
        "categorical3",
        "categorical4",
    }


def test_independent_references_pass_their_own_numerical_error_gates() -> None:
    records = {record.family: record for record in run_e0a_score_validation()}
    assert records["bdeu"].absolute_error < 1e-10
    assert records["bdeu_cat3_two_parent"].absolute_error < 1e-10
    assert records["bdeu_cat4_two_parent"].absolute_error < 1e-10
    assert records["clg_nig"].absolute_error < 1e-10
    assert records["clg_nig_no_parent"].absolute_error < 1e-10
    assert records["clg_nig_mixed_contexts"].absolute_error < 1e-10
    assert records["clg_nig_scalar_quadrature"].absolute_error < 1e-10

    logit = records["binary_logit_laplace"]
    assert logit.diagnostics["quadrature_status"] == "converged"
    assert logit.diagnostics["reported_absolute_log_error"] < 1e-8
    assert logit.passed
    assert logit.absolute_error < logit.tolerance == 0.10

    logit_three = records["binary_logit_3coef_laplace"]
    assert logit_three.diagnostics["free_coefficients"] == 3
    assert logit_three.diagnostics["quadrature_status"] == "converged"
    assert logit_three.diagnostics["reported_absolute_log_error"] < 1e-8
    assert logit_three.passed
    assert logit_three.absolute_error < logit_three.tolerance == 0.10

    softmax = records["multinomial_softmax_laplace"]
    agreement = records["softmax_reference_agreement"]
    assert softmax.diagnostics["quadrature_status"] == "converged"
    assert softmax.diagnostics["reported_absolute_log_error"] < 1e-8
    # Protocol 7.1 says 10^7 effective draws OR MCSE < 0.01.  This run uses
    # the latter branch and must not pretend to have 10^7 effective draws.
    assert softmax.diagnostics["importance_effective_sample_size"] < 10_000_000
    assert softmax.diagnostics["importance_log_mcse"] < 0.01
    assert agreement.passed
    assert agreement.absolute_error < 0.03
    assert softmax.passed
    assert softmax.absolute_error < softmax.tolerance == 0.20


def test_all_static_branch_cells_dispatch_and_return_finite_scores() -> None:
    records = run_structural_fixture_matrix()
    assert len(records) == 72
    assert all(record.passed for record in records)
    assert len({record.diagnostics["data_sha256"] for record in records}) == 72


def test_derivatives_modes_hessians_and_invariances_pass() -> None:
    report = run_e0a_full_validation()
    assert all(record["passed"] for record in report["derivative_audits"])
    assert all(record.passed for record in run_invariance_checks())


def test_explicit_pathology_policy_never_converts_errors_to_scores() -> None:
    records = run_error_policy_checks()
    assert len(records) == 8
    assert all(record.passed for record in records)
    errors = {record.pathology: record for record in records}
    assert errors["constant_predictor"].observed_policy == "ValueError"
    assert errors["singular_dg_covariance"].observed_policy == "ValueError"
    assert errors["nonfinite_input"].observed_policy == "ValueError"


def test_branch_boundary_prior_predictive_audit_is_deterministic_and_finite() -> None:
    first = run_branch_boundary_audit()
    second = run_branch_boundary_audit()
    assert first == second
    assert first.replicates == 128
    assert first.finite_rate == 1.0
    assert not first.duplicated_external_penalty
    assert first.passed


def test_full_g0_gate_passes_branch_coverage_and_declared_approximation_bounds() -> None:
    report = run_e0a_full_validation()
    assert report["status"] == "pass"
    assert report["all_passed"]
    assert all(record["status"] == "pass" for record in report["score_accuracy_records"])
    coverage = report["reference_coverage"]
    assert coverage["status"] == "pass"
    assert coverage["complete"]
    assert coverage["blocker_code"] is None
    assert coverage["independently_validated_fixture_count"] == 9
    assert coverage["expanded_structural_cells"] == 72
    assert coverage["all_expanded_cells_executed"]


def test_missing_mathematical_reference_branch_blocks_coverage(monkeypatch) -> None:
    matrix = validation_module.load_fixture_matrix()
    matrix["reference_anchors"] = [
        row
        for row in matrix["reference_anchors"]
        if row["fixture_id"] != "softmax-cat3-one-continuous-n8"
    ]
    monkeypatch.setattr(validation_module, "load_fixture_matrix", lambda path=None: matrix)
    coverage = validation_module._reference_coverage_report(72)
    assert coverage["status"] == "fail"
    assert not coverage["complete"]
    assert "softmax-cat3-one-continuous" in coverage["blocker"]


def test_e0a_cli_normalizes_tuple_diagnostics_for_atomic_json_output(
    tmp_path, monkeypatch
) -> None:
    report = {
        "schema": "ordersmc-e0a-validation-report-v1",
        "status": "fail",
        "all_passed": False,
        "fixture_matrix": {"sample_sizes": (8, 20, 100)},
        "diagnostics": {"quadrature_history": ((8, 0.1), (16, 0.01))},
    }
    monkeypatch.setattr(cli_module, "run_e0a_full_validation", lambda: report)
    destination = tmp_path / "e0a.json"

    assert cli_module.main(["e0a", "--full", "--output", str(destination)]) == 0

    persisted = json.loads(destination.read_text(encoding="utf-8"))
    assert persisted["fixture_matrix"]["sample_sizes"] == [8, 20, 100]
    assert persisted["diagnostics"]["quadrature_history"] == [
        [8, 0.1],
        [16, 0.01],
    ]
