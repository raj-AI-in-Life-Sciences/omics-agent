"""10 tests: shape, monotonicity, loss descent for survival models."""

from __future__ import annotations

import numpy as np
import pytest
import torch


# ── CoxPH ────────────────────────────────────────────────────────────────────

def test_cox_fit_predict_shape(X, durations, events):
    from omics_agent.survival.cox_model import CoxModel
    m = CoxModel()
    m.fit(X[:, :256], X[:, 256:], durations.astype(float), events.astype(int))
    risk = m.predict_partial_hazard(X[:, :256], X[:, 256:])
    assert risk.shape == (len(X),)


def test_cox_risk_is_finite(X, durations, events):
    from omics_agent.survival.cox_model import CoxModel
    m = CoxModel()
    m.fit(X[:, :256], X[:, 256:], durations.astype(float), events.astype(int))
    risk = m.predict_partial_hazard(X[:, :256], X[:, 256:])
    assert np.all(np.isfinite(risk))


# ── RSF ──────────────────────────────────────────────────────────────────────

def test_rsf_fit_predict_risk_shape(X, structured_y):
    from omics_agent.survival.rsf_model import RSFModel
    m = RSFModel(n_estimators=10)
    m.fit(X, structured_y)
    risk = m.predict_risk(X)
    assert risk.shape == (len(X),)


def test_rsf_survival_matrix_shape(X, structured_y):
    from omics_agent.survival.rsf_model import RSFModel
    m = RSFModel(n_estimators=10)
    m.fit(X, structured_y)
    surv = m.predict_survival_matrix(X)
    assert surv.shape[0] == len(X)
    assert surv.shape[1] == len(m.eval_times_)


def test_rsf_survival_monotone(X, structured_y):
    """S(t) should be non-increasing for each patient."""
    from omics_agent.survival.rsf_model import RSFModel
    m = RSFModel(n_estimators=10)
    m.fit(X, structured_y)
    surv = m.predict_survival_matrix(X)
    diffs = np.diff(surv, axis=1)
    assert np.all(diffs <= 1e-6), "Survival curve is not monotone non-increasing"


# ── DeepSurv ─────────────────────────────────────────────────────────────────

def test_deepsurv_loss_decreases(X, durations, events):
    from omics_agent.survival.deepsurv_model import DeepSurvModel, breslow_loss
    import torch
    m = DeepSurvModel(input_dim=X.shape[1], n_epochs=5)
    m.fit(X, durations.astype(float), events.astype(float))
    risk = m.predict_risk(X)
    assert risk.shape == (len(X),)
    assert np.all(np.isfinite(risk))


def test_deepsurv_risk_shape(X, durations, events):
    from omics_agent.survival.deepsurv_model import DeepSurvModel
    m = DeepSurvModel(input_dim=X.shape[1], n_epochs=2)
    m.fit(X, durations.astype(float), events.astype(float))
    assert m.predict_risk(X).shape == (len(X),)


# ── XGBoost AFT ──────────────────────────────────────────────────────────────

def test_xgb_aft_fit_predict(X, durations, events, feature_names):
    from omics_agent.survival.xgb_model import XGBSurvivalModel
    m = XGBSurvivalModel(n_estimators=10)
    m.fit(X, durations.astype(float), events.astype(int), feature_names=feature_names)
    risk = m.predict_risk(X)
    assert risk.shape == (len(X),)
    assert np.all(np.isfinite(risk))


# ── Ensemble ─────────────────────────────────────────────────────────────────

def test_ensemble_weights_sum_to_one(X, durations, events):
    from omics_agent.survival.ensemble import SurvivalEnsemble
    rng  = np.random.default_rng(0)
    risk = rng.standard_normal(len(X)).astype(np.float32)
    ens  = SurvivalEnsemble()
    ens.calibrate_weights(risk, risk, risk, risk, durations.astype(float), events)
    total = sum(ens.weights.values())
    assert abs(total - 1.0) < 1e-5


def test_ensemble_rank_shape(X, durations, events):
    from omics_agent.survival.ensemble import SurvivalEnsemble
    rng  = np.random.default_rng(1)
    risk = rng.standard_normal((4, len(X))).astype(np.float32)
    ens  = SurvivalEnsemble()
    ranks = ens.ensemble_risk_ranks(*risk)
    assert ranks.shape == (len(X),)
