"""
SHAP node: per-model SHAP → C-index-weighted ensemble aggregation.
"""

from __future__ import annotations

import pickle
from typing import Any, Dict, List

import numpy as np

from ..graph.state import OmicsState
from ..core.config import get_config

_EXPLAINERS: Dict[str, Any] = {}


def _load_explainer(key: str, path: str) -> Any:
    if key not in _EXPLAINERS:
        with open(path, "rb") as f:
            _EXPLAINERS[key] = pickle.load(f)
    return _EXPLAINERS[key]


def shap_node(state: OmicsState) -> OmicsState:
    """
    Compute SHAP values for each model and aggregate into ensemble SHAP.
    Returns top-15 features as a list of dicts in state["shap_values_ensemble"].
    """
    cfg    = get_config()
    errors = list(state.get("errors") or [])

    shap_cox: dict      = {}
    shap_rsf: dict      = {}
    shap_dsv: dict      = {}
    shap_xgb: dict      = {}
    shap_ens: list      = []

    try:
        from ..data.feature_engineer import CLINICAL_FEATURE_NAMES
        emb   = np.array(state.get("geneformer_embedding") or [0.0] * 256, dtype=np.float32)
        clin  = state.get("clinical_features") or {}
        clin_vec = np.array(
            [float(clin.get(k, 0.0)) for k in CLINICAL_FEATURE_NAMES],
            dtype=np.float32,
        )
        X_full = np.concatenate([emb, clin_vec])[np.newaxis, :]  # (1, 267)

        # CoxPH operates on PCA-50 space — build reduced X
        X_cox = X_full[:, :50]  # first 50 dims as proxy for PCA

        # ── Per-model SHAP ────────────────────────────────────────────────
        try:
            ex_cox = _load_explainer("shap_cox", cfg.shap_cox_path)
            sv_cox = ex_cox.shap_values(X_cox)         # (1, 50)
            shap_cox = {
                f"PC{i+1}": round(float(sv_cox[0, i]), 5)
                for i in range(sv_cox.shape[1])
            }
        except Exception as e:
            errors.append(f"shap_cox: {e}")
            sv_cox = np.zeros((1, 50))

        try:
            ex_rsf = _load_explainer("shap_rsf", cfg.shap_rsf_path)
            sv_rsf = ex_rsf.shap_values(X_full)
            feat_names = cfg.feature_names_267
            shap_rsf = {feat_names[i]: round(float(sv_rsf[0, i]), 5)
                        for i in range(sv_rsf.shape[1])}
        except Exception as e:
            errors.append(f"shap_rsf: {e}")
            sv_rsf = np.zeros((1, 267))

        try:
            ex_dsv = _load_explainer("shap_deepsurv", cfg.shap_deepsurv_path)
            sv_dsv = ex_dsv.shap_values(X_full)
            feat_names = cfg.feature_names_267
            shap_dsv = {feat_names[i]: round(float(sv_dsv[0, i]), 5)
                        for i in range(sv_dsv.shape[1])}
        except Exception as e:
            errors.append(f"shap_deepsurv: {e}")
            sv_dsv = np.zeros((1, 267))

        try:
            ex_xgb = _load_explainer("shap_xgb", cfg.shap_xgb_path)
            sv_xgb = ex_xgb.shap_values(X_full)
            feat_names = cfg.feature_names_267
            shap_xgb = {feat_names[i]: round(float(sv_xgb[0, i]), 5)
                        for i in range(sv_xgb.shape[1])}
        except Exception as e:
            errors.append(f"shap_xgb: {e}")
            sv_xgb = np.zeros((1, 267))

        # ── Ensemble aggregation ──────────────────────────────────────────
        try:
            from ..explainability.shap_aggregator import EnsembleSHAPAggregator
            ens_model = _load_explainer.__self__ if False else None  # not applicable
            # Load ensemble weights from the saved ensemble model
            with open(cfg.ensemble_path, "rb") as f:
                ens_m = pickle.load(f)
            weights = ens_m.weights

            agg = EnsembleSHAPAggregator(
                weights       = weights,
                feature_names = cfg.feature_names_267,
            )
            ens_shap = agg.aggregate(sv_cox, sv_rsf, sv_dsv, sv_xgb)  # (1, 267)
            shap_ens = agg.shap_table(ens_shap, patient_idx=0, n=15)
        except Exception as e:
            errors.append(f"shap_ensemble: {e}")
            shap_ens = []

    except Exception as exc:
        errors.append(f"shap_node: {exc}")

    return {
        **state,
        "shap_values_cox":      shap_cox,
        "shap_values_rsf":      shap_rsf,
        "shap_values_deepsurv": shap_dsv,
        "shap_values_xgb":      shap_xgb,
        "shap_values_ensemble": shap_ens,
        "errors": errors,
    }
