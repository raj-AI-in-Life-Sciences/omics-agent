"""
C-index-weighted ensemble SHAP aggregator.

Combines per-model SHAP values into a single ensemble explanation,
weighting each model by its normalised C-index (same weights used by
SurvivalEnsemble for risk aggregation).
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional


class EnsembleSHAPAggregator:
    """
    Aggregate SHAP values from CoxPH, RSF, DeepSurv, XGBoost AFT.

    All models must share the same feature set. CoxPH uses PCA-reduced
    features (50 dims) + clinical (11), so we align everything to the
    full 61-dim feature space with PCA components expanded back.

    For portfolio simplicity we align to the full 267-dim space that
    RSF/DeepSurv/XGBoost use.  CoxPH SHAP values are padded with zeros
    for the embedding dims beyond PCA-50.
    """

    def __init__(
        self,
        weights: Dict[str, float],      # {"cox": w1, "rsf": w2, "deepsurv": w3, "xgb": w4}
        feature_names: List[str],       # 267-dim feature names
        n_embed_dims: int = 256,        # number of embedding dimensions (before clinical)
        n_pca_dims:   int = 50,         # PCA dims used by CoxPH
    ):
        self.weights       = weights
        self.feature_names = feature_names
        self.n_embed_dims  = n_embed_dims
        self.n_pca_dims    = n_pca_dims
        self.n_features    = len(feature_names)

    def aggregate(
        self,
        cox_shap:      np.ndarray,    # (N, 61)  PCA-50 + 11 clinical
        rsf_shap:      np.ndarray,    # (N, 267)
        deepsurv_shap: np.ndarray,    # (N, 267)
        xgb_shap:      np.ndarray,    # (N, 267)
    ) -> np.ndarray:
        """
        Return weighted ensemble SHAP matrix (N, 267).
        Cox SHAP values are zero-padded for embed dims 50..256.
        """
        N = rsf_shap.shape[0]
        cox_full = np.zeros((N, self.n_features), dtype=np.float32)
        # PCA dims → first n_pca_dims of embedding block
        n_cox_feats = cox_shap.shape[1]
        pca_cols  = min(self.n_pca_dims, n_cox_feats)
        clin_cols = n_cox_feats - pca_cols
        cox_full[:, :pca_cols] = cox_shap[:, :pca_cols]
        if clin_cols > 0:
            cox_full[:, self.n_embed_dims:self.n_embed_dims + clin_cols] = \
                cox_shap[:, pca_cols:]

        w = self.weights
        ensemble = (
            w.get("cox",      0.25) * cox_full
            + w.get("rsf",    0.25) * rsf_shap.astype(np.float32)
            + w.get("deepsurv", 0.25) * deepsurv_shap.astype(np.float32)
            + w.get("xgb",    0.25) * xgb_shap.astype(np.float32)
        )
        return ensemble

    def top_features(
        self,
        ensemble_shap: np.ndarray,   # (N, 267) or (267,)
        n: int = 15,
    ) -> List[dict]:
        """Return top-n features by mean |SHAP| from ensemble matrix."""
        sv = ensemble_shap
        if sv.ndim == 1:
            sv = sv[np.newaxis, :]
        mean_abs = np.abs(sv).mean(axis=0)
        idx = np.argsort(mean_abs)[::-1][:n]
        return [
            {
                "feature":    self.feature_names[i],
                "mean_shap":  round(float(sv[:, i].mean()), 5),
                "abs_shap":   round(float(mean_abs[i]), 5),
                "direction":  "risk+" if sv[:, i].mean() > 0 else "risk-",
                "model":      "ensemble",
            }
            for i in idx
        ]

    def shap_table(
        self,
        ensemble_shap: np.ndarray,
        patient_idx:   int = 0,
        n:             int = 15,
    ) -> List[dict]:
        """
        Return per-patient top-n SHAP table suitable for the LLM prompt.
        """
        sv = ensemble_shap
        if sv.ndim == 2:
            sv = sv[patient_idx]
        abs_sv = np.abs(sv)
        top_idx = np.argsort(abs_sv)[::-1][:n]
        return [
            {
                "feature":   self.feature_names[i],
                "shap":      round(float(sv[i]), 5),
                "direction": "↑risk" if sv[i] > 0 else "↓risk",
            }
            for i in top_idx
        ]
