"""Honest method registry for the lean paper protocol."""

from __future__ import annotations


METHODS = (
    {
        "method": "exact-order-dp",
        "target": "order-modular",
        "role": "E1 reference",
        "implementation": "included",
    },
    {
        "method": "ordersmc",
        "target": "order-modular",
        "role": "proposed method",
        "implementation": "included",
    },
    {
        "method": "order-mcmc",
        "target": "order-modular",
        "role": "same-target baseline",
        "implementation": "included",
    },
    {
        "method": "partition-mcmc",
        "target": "structure-modular",
        "role": "end-to-end baseline",
        "implementation": "included-validated",
        "validation": "exact normalizer/PIP parity and detailed balance",
    },
    {
        "method": "rev-mcmc",
        "target": "structure-modular",
        "role": "end-to-end baseline",
        "implementation": "included-validated",
        "validation": "standard-kernel invariance and conjugate REV detailed balance",
    },
    {
        "method": "dag-gflownet",
        "target": "method-native",
        "role": "optional modern end-to-end baseline",
        "implementation": "optional-adapter-required",
    },
    {
        "method": "bcdnets-linear-mixed",
        "target": "method-native",
        "role": "optional compatible-subpanel baseline",
        "implementation": "optional-adapter-required",
    },
)


def method_registry() -> dict[str, object]:
    return {
        "schema": "ordersmc-method-registry-v3",
        "methods": [dict(row) for row in METHODS],
        "note": (
            "Registry membership is not an implementation claim. Confirmatory rows "
            "remain visibly unavailable until their adapters pass acceptance tests."
        ),
    }
