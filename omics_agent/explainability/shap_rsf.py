"""SHAP TreeExplainer for RandomSurvivalForest."""

from __future__ import annotations

import numpy as np
import shap
from typing import List, Optional


class RSFSHAPExplainer:
    """
    TreeExplainer for scikit-survival RandomSurvivalForest.

    RSF is a tree ensemble → shap.TreeExplainer works natively.
    We explain the cumulative hazard function (predict_cumulative_hazard_function
    is not directly SHAP-able; instead we explain predict_risk which calls
    the ensemble's oob_prediction_ or the cumulative hazard integrated to a scalar).
    """

    def __init__(self, rsf_model, feature_names: List[str]):
        self.rsf_model     = rsf_model        # fitted RSFModel instance
        self.feature_names = feature_names
        self._explainer: Optional[shap.TreeExplainer] = None

    def fit(self, X_background: np.ndarray) -> "RSFSHAPExplainer":
        """
        Build TreeExplainer. X_background is used as reference distribution.
        """
        self._explainer = shap.TreeExplainer(
            self.rsf_model.model,
            data=shap.sample(X_background, min(100, len(X_background))),
            feature_perturbation="interventional",
        )
        return self

    def shap_values(self, X: np.ndarray) -> np.ndarray:
        """Return SHAP values (N, n_features). Shape matches X."""
        if self._explainer is None:
            raise RuntimeError("Call fit() before shap_values()")
        sv = self._explainer.shap_values(X)
        # sksurv RSF TreeExplainer may return list — take first element
        if isinstance(sv, list):
            sv = sv[0]
        return sv

    def top_features(self, X: np.ndarray, n: int = 15) -> List[dict]:
        sv = self.shap_values(X)
        mean_abs = np.abs(sv).mean(axis=0)
        idx = np.argsort(mean_abs)[::-1][:n]
        return [
            {
                "feature":   self.feature_names[i],
                "mean_shap": round(float(sv[:, i].mean()), 5),
                "abs_shap":  round(float(mean_abs[i]), 5),
                "direction": "risk+" if sv[:, i].mean() > 0 else "risk-",
            }
            for i in idx
        ]
