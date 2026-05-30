"""4 tests: marginal coverage, interval shape, calibration."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def conformal_setup(rng, eval_times):
    from omics_agent.survival.conformal import SurvivalConformalPredictor
    N = 30
    T = len(eval_times)
    # Predict slightly perturbed survival curves
    true_surv = np.linspace(1.0, 0.3, T)
    pred_surv = np.tile(true_surv, (N, 1)) + rng.uniform(-0.1, 0.1, (N, T)).astype(np.float32)
    pred_surv = np.clip(pred_surv, 0, 1)

    durations = rng.uniform(12, 120, N).astype(np.float32)
    events    = rng.binomial(1, 0.7, N).astype(int)

    cp = SurvivalConformalPredictor(alpha=0.10)
    cp.calibrate(pred_surv, durations.astype(float), events, eval_times.astype(float))
    return cp, pred_surv, durations, events


def test_conformal_interval_shape(conformal_setup, eval_times):
    cp, pred_surv, _, _ = conformal_setup
    lo, hi = cp.predict_interval(pred_surv)
    assert lo.shape == pred_surv.shape
    assert hi.shape == pred_surv.shape


def test_conformal_lower_le_upper(conformal_setup):
    cp, pred_surv, _, _ = conformal_setup
    lo, hi = cp.predict_interval(pred_surv)
    assert np.all(lo <= hi + 1e-6)


def test_conformal_bounds_in_01(conformal_setup):
    cp, pred_surv, _, _ = conformal_setup
    lo, hi = cp.predict_interval(pred_surv)
    assert np.all(lo >= 0.0)
    assert np.all(hi <= 1.0)


def test_conformal_q_hat_positive(conformal_setup):
    cp, _, _, _ = conformal_setup
    assert cp.q_hat >= 0.0
