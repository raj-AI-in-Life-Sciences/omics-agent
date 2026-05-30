"""SHAP GradientExplainer for DeepSurv (PyTorch)."""

from __future__ import annotations

import numpy as np
import shap
import torch
from typing import List, Optional


class DeepSurvSHAPExplainer:
    """
    GradientExplainer for the DeepSurvNet PyTorch model.

    GradientExplainer uses back-propagation through the network to compute
    expected gradients (SHAP-Gradient approximation).
    """

    def __init__(self, deepsurv_model, feature_names: List[str]):
        self.deepsurv_model = deepsurv_model   # fitted DeepSurvModel
        self.feature_names  = feature_names
        self._explainer: Optional[shap.GradientExplainer] = None

    def fit(self, X_background: np.ndarray) -> "DeepSurvSHAPExplainer":
        """
        Fit GradientExplainer with background tensor.
        X_background: (N_bg, n_features) float32 ndarray
        """
        self.deepsurv_model.net.eval()
        bg_tensor = torch.tensor(X_background, dtype=torch.float32)
        self._explainer = shap.GradientExplainer(
            self.deepsurv_model.net, bg_tensor
        )
        return self

    def shap_values(self, X: np.ndarray) -> np.ndarray:
        """Return SHAP values (N, n_features)."""
        if self._explainer is None:
            raise RuntimeError("Call fit() before shap_values()")
        x_tensor = torch.tensor(X, dtype=torch.float32)
        sv = self._explainer.shap_values(x_tensor)
        if isinstance(sv, list):
            sv = sv[0]
        return np.array(sv)

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
