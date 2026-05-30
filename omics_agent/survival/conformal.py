"""
Split conformal prediction for survival curves.

Provides guaranteed P(y ∈ interval) ≥ 1−α under exchangeability,
calibrated on the dedicated calibration split (never seen by survival models).
"""

from __future__ import annotations

import math
import numpy as np
from lifelines import KaplanMeierFitter
from typing import Tuple


class SurvivalConformalPredictor:
    """
    Split conformal predictor for survival curve S(t).

    Nonconformity score: max_t |S_hat(t) − KM(t)| across evaluation timepoints.
    Calibration: fit quantile on calibration set.
    Inference: return lower/upper bounds per timepoint.
    """

    def __init__(self, alpha: float = 0.10):
        self.alpha = alpha
        self.q_hat: float = 0.0
        self.eval_times_: np.ndarray = np.array([])

    def calibrate(
        self,
        predicted_surv:  np.ndarray,    # (N_cal, T) predictions on calibration set
        true_durations:  np.ndarray,    # (N_cal,)
        true_events:     np.ndarray,    # (N_cal,)
        eval_times:      np.ndarray,    # (T,)
    ) -> None:
        self.eval_times_ = eval_times

        # KM estimate on calibration set
        km = KaplanMeierFitter()
        km.fit(true_durations, event_observed=true_events)
        km_surv = np.array([
            km.survival_function_at_times([t]).values[0, 0]
            for t in eval_times
        ], dtype=np.float32)

        # Per-patient max absolute deviation
        scores = np.abs(predicted_surv - km_surv).max(axis=1)  # (N_cal,)
        n = len(scores)
        q_level = math.ceil((1 - self.alpha) * (n + 1)) / n
        self.q_hat = float(np.quantile(scores, min(q_level, 1.0)))

    def predict_interval(
        self,
        predicted_surv: np.ndarray,   # (N, T) or (T,) for single patient
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return (lower, upper) survival curve bounds. Shape: (N, T) or (T,)."""
        lower = np.clip(predicted_surv - self.q_hat, 0.0, 1.0)
        upper = np.clip(predicted_surv + self.q_hat, 0.0, 1.0)
        return lower, upper

    def empirical_coverage(
        self,
        predicted_surv: np.ndarray,   # (N, T)
        true_durations: np.ndarray,
        true_events:    np.ndarray,
    ) -> float:
        """Fraction of patients whose true KM curve falls within the band."""
        km = KaplanMeierFitter()
        km.fit(true_durations, event_observed=true_events)
        km_surv = np.array([
            km.survival_function_at_times([t]).values[0, 0]
            for t in self.eval_times_
        ])

        lower, upper = self.predict_interval(predicted_surv)
        covered = np.all((km_surv >= lower) & (km_surv <= upper), axis=1)
        return float(covered.mean())
