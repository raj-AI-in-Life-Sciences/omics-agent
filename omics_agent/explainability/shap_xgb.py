"""SHAP TreeExplainer for XGBoost AFT survival model."""

from __future__ import annotations

import numpy as np
import shap
import xgboost as xgb
from typing import List, Optional


class XGBSHAPExplainer:
    """
    TreeExplainer for XGBSurvivalModel (xgb.Booster under the hood).

    XGBoost natively supports exact SHAP values via TreeExplainer.
    """

    def __init__(self, xgb_model, feature_names: List[str]):
        self.xgb_model     = xgb_model        # fitted XGBSurvivalModel
        self.feature_names = feature_names
        self._explainer: Optional[shap.TreeExplainer] = None

    def fit(self, X_background: np.ndarray) -> "XGBSHAPExplainer":
        self._explainer = shap.TreeExplainer(
            self.xgb_model.model,
            feature_perturbation="tree_path_dependent",
        )
        return self

    def shap_values(self, X: np.ndarray) -> np.ndarray:
        """Return SHAP values (N, n_features). Positive = higher predicted time = lower risk.
        We negate so positive SHAP = higher risk (consistent with other models)."""
        if self._explainer is None:
            raise RuntimeError("Call fit() before shap_values()")
        dmatrix = xgb.DMatrix(X, feature_names=self.xgb_model.feature_names)
        sv = self._explainer.shap_values(dmatrix)
        if isinstance(sv, list):
            sv = sv[0]
        # XGBoost AFT predicts time (higher=longer survival); negate for risk direction
        return -np.array(sv)

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
