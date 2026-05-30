"""SHAP LinearExplainer for CoxPH model (lifelines pipeline)."""

from __future__ import annotations

import numpy as np
import shap
from typing import List, Optional


class CoxSHAPExplainer:
    """
    Wraps shap.LinearExplainer around the CoxModel's internal sklearn Pipeline.

    The CoxPHFitter is not a sklearn estimator, so we wrap the linear
    combination (log-partial-hazard = Xβ) as a callable for LinearExplainer.
    """

    def __init__(self, cox_model, feature_names: List[str]):
        self.cox_model     = cox_model          # fitted CoxModel instance
        self.feature_names = feature_names
        self._explainer: Optional[shap.LinearExplainer] = None

    def fit(self, X_background: np.ndarray) -> "CoxSHAPExplainer":
        """
        Fit LinearExplainer using background dataset.
        X_background: (N_bg, n_features) — the reduced feature matrix
                      fed into CoxModel (PCA-50 + 11 clinical = 61 dims).
        """
        coefs = self.cox_model.get_coefficients()   # (n_features,)
        # Wrap as sklearn-compatible linear model
        linear_model = _LinearWrapper(coefs)
        self._explainer = shap.LinearExplainer(linear_model, X_background)
        return self

    def shap_values(self, X: np.ndarray) -> np.ndarray:
        """Return SHAP values (N, n_features)."""
        if self._explainer is None:
            raise RuntimeError("Call fit() before shap_values()")
        return self._explainer.shap_values(X)

    def top_features(self, X: np.ndarray, n: int = 15) -> List[dict]:
        """Return top-n features by mean |SHAP| with direction."""
        sv = self.shap_values(X)
        mean_abs = np.abs(sv).mean(axis=0)
        idx = np.argsort(mean_abs)[::-1][:n]
        return [
            {
                "feature":    self.feature_names[i],
                "mean_shap":  round(float(sv[:, i].mean()), 5),
                "abs_shap":   round(float(mean_abs[i]), 5),
                "direction":  "risk+" if sv[:, i].mean() > 0 else "risk-",
            }
            for i in idx
        ]


class _LinearWrapper:
    """Minimal sklearn-compatible wrapper so LinearExplainer accepts β vector."""

    def __init__(self, coefs: np.ndarray):
        self.coef_      = coefs
        self.intercept_ = np.array([0.0])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return X @ self.coef_
