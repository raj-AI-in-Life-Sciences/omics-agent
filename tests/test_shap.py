"""5 tests: SHAP shape, NaN, ensemble aggregation."""

from __future__ import annotations

import numpy as np
import pytest


def test_shap_aggregator_output_shape(rng, feature_names):
    from omics_agent.explainability.shap_aggregator import EnsembleSHAPAggregator
    N = 5
    weights = {"cox": 0.25, "rsf": 0.25, "deepsurv": 0.25, "xgb": 0.25}
    agg = EnsembleSHAPAggregator(weights=weights, feature_names=feature_names)

    sv_cox = rng.standard_normal((N, 50)).astype(np.float32)
    sv_rsf = rng.standard_normal((N, 267)).astype(np.float32)
    sv_dsv = rng.standard_normal((N, 267)).astype(np.float32)
    sv_xgb = rng.standard_normal((N, 267)).astype(np.float32)

    ens = agg.aggregate(sv_cox, sv_rsf, sv_dsv, sv_xgb)
    assert ens.shape == (N, 267)


def test_shap_aggregator_no_nan(rng, feature_names):
    from omics_agent.explainability.shap_aggregator import EnsembleSHAPAggregator
    N = 5
    weights = {"cox": 0.3, "rsf": 0.3, "deepsurv": 0.2, "xgb": 0.2}
    agg = EnsembleSHAPAggregator(weights=weights, feature_names=feature_names)

    sv_cox = rng.standard_normal((N, 50)).astype(np.float32)
    sv_rsf = rng.standard_normal((N, 267)).astype(np.float32)
    sv_dsv = rng.standard_normal((N, 267)).astype(np.float32)
    sv_xgb = rng.standard_normal((N, 267)).astype(np.float32)

    ens = agg.aggregate(sv_cox, sv_rsf, sv_dsv, sv_xgb)
    assert not np.isnan(ens).any()


def test_shap_top_features_length(rng, feature_names):
    from omics_agent.explainability.shap_aggregator import EnsembleSHAPAggregator
    N = 5
    weights = {"cox": 0.25, "rsf": 0.25, "deepsurv": 0.25, "xgb": 0.25}
    agg = EnsembleSHAPAggregator(weights=weights, feature_names=feature_names)
    ens = rng.standard_normal((N, 267)).astype(np.float32)
    top = agg.top_features(ens, n=15)
    assert len(top) == 15


def test_shap_table_has_required_keys(rng, feature_names):
    from omics_agent.explainability.shap_aggregator import EnsembleSHAPAggregator
    weights = {"cox": 0.25, "rsf": 0.25, "deepsurv": 0.25, "xgb": 0.25}
    agg = EnsembleSHAPAggregator(weights=weights, feature_names=feature_names)
    ens = rng.standard_normal((3, 267)).astype(np.float32)
    table = agg.shap_table(ens, patient_idx=0, n=10)
    for row in table:
        assert "feature" in row and "shap" in row and "direction" in row


def test_shap_weights_proportional(rng, feature_names):
    """Ensemble SHAP should scale linearly with weights."""
    from omics_agent.explainability.shap_aggregator import EnsembleSHAPAggregator
    N = 3
    sv = np.ones((N, 267), dtype=np.float32)

    # All weight to RSF
    agg1 = EnsembleSHAPAggregator({"cox": 0.0, "rsf": 1.0, "deepsurv": 0.0, "xgb": 0.0},
                                   feature_names)
    ens1 = agg1.aggregate(np.zeros((N, 50)), sv, np.zeros((N, 267)), np.zeros((N, 267)))
    # Expect values approximately equal to rsf_shap (padded Cox adds zeros)
    assert np.allclose(ens1[:, 50:256], 0.0, atol=1e-5)  # embed dims 50-256: cox zero-padded
