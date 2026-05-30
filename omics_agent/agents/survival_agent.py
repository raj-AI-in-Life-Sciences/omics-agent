"""
Survival subgraph nodes: cox, rsf, deepsurv, xgb, ensemble, conformal.

All nodes load pre-fitted model objects from config paths.
The models are loaded lazily (cached in module globals) to avoid
reloading on every graph invocation.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict

import numpy as np

from ..graph.state import OmicsState
from ..core.config import get_config

# ── Module-level model cache ─────────────────────────────────────────────────
_MODELS: Dict[str, Any] = {}


def _load_model(key: str, path: str) -> Any:
    if key not in _MODELS:
        with open(path, "rb") as f:
            _MODELS[key] = pickle.load(f)
    return _MODELS[key]


def _build_feature_matrix(state: OmicsState) -> np.ndarray:
    """
    Concatenate Geneformer embedding (256) + clinical (11) → (267,) array.
    """
    emb  = np.array(state.get("geneformer_embedding") or [0.0] * 256, dtype=np.float32)
    clin = state.get("clinical_features") or {}
    from ..data.feature_engineer import CLINICAL_FEATURE_NAMES
    clin_vec = np.array(
        [float(clin.get(k, 0.0)) for k in CLINICAL_FEATURE_NAMES],
        dtype=np.float32,
    )
    return np.concatenate([emb, clin_vec])[np.newaxis, :]  # (1, 267)


# ── Node functions ────────────────────────────────────────────────────────────

def cox_node(state: OmicsState) -> OmicsState:
    cfg    = get_config()
    errors = list(state.get("errors") or [])
    risk   = 0.0
    try:
        model = _load_model("cox", cfg.cox_model_path)
        X     = _build_feature_matrix(state)
        risk  = float(model.predict_partial_hazard(
            X[:, :50],                # PCA-50 block
            X[:, 256:]                # clinical block
        )[0])
    except Exception as exc:
        errors.append(f"cox_node: {exc}")
    return {**state, "cox_risk_score": risk, "errors": errors}


def rsf_node(state: OmicsState) -> OmicsState:
    cfg    = get_config()
    errors = list(state.get("errors") or [])
    risk   = 0.0
    surv_fn: dict = {}
    try:
        model  = _load_model("rsf", cfg.rsf_model_path)
        X      = _build_feature_matrix(state)
        risk   = float(model.predict_risk(X)[0])
        surv_m = model.predict_survival_matrix(X)       # (1, T)
        surv_fn = {
            int(t): round(float(s), 4)
            for t, s in zip(model.eval_times_, surv_m[0])
        }
    except Exception as exc:
        errors.append(f"rsf_node: {exc}")
    return {
        **state,
        "rsf_survival_function": surv_fn,
        "errors": errors,
        # rsf risk stored temporarily — ensemble needs it
        "_rsf_risk": risk,
    }


def deepsurv_node(state: OmicsState) -> OmicsState:
    cfg    = get_config()
    errors = list(state.get("errors") or [])
    risk   = 0.0
    try:
        model = _load_model("deepsurv", cfg.deepsurv_model_path)
        X     = _build_feature_matrix(state)
        risk  = float(model.predict_risk(X)[0])
    except Exception as exc:
        errors.append(f"deepsurv_node: {exc}")
    return {**state, "deepsurv_risk_score": risk, "errors": errors}


def xgb_node(state: OmicsState) -> OmicsState:
    cfg    = get_config()
    errors = list(state.get("errors") or [])
    risk   = 0.0
    try:
        model = _load_model("xgb", cfg.xgb_model_path)
        X     = _build_feature_matrix(state)
        risk  = float(model.predict_risk(X)[0])
    except Exception as exc:
        errors.append(f"xgb_node: {exc}")
    return {**state, "xgb_risk_score": risk, "errors": errors}


def ensemble_node(state: OmicsState) -> OmicsState:
    cfg    = get_config()
    errors = list(state.get("errors") or [])
    try:
        ens = _load_model("ensemble", cfg.ensemble_path)

        cox_r = np.array([state.get("cox_risk_score", 0.0)])
        rsf_r = np.array([state.get("_rsf_risk", 0.0)])
        dsv_r = np.array([state.get("deepsurv_risk_score", 0.0)])
        xgb_r = np.array([state.get("xgb_risk_score", 0.0)])

        ensemble_risk = float(
            ens.ensemble_risk_ranks(cox_r, rsf_r, dsv_r, xgb_r)[0]
        )

        # Build RSF survival matrix for single patient
        surv_fn = state.get("rsf_survival_function", {})
        eval_times = sorted(surv_fn.keys())
        surv_vals  = np.array([surv_fn[t] for t in eval_times], dtype=np.float32)
        surv_matrix = surv_vals[np.newaxis, :]  # (1, T)

        median_surv = float(
            ens.predict_median_survival(surv_matrix, np.array(eval_times))[0]
        )
        ensemble_curve = {int(t): round(float(s), 4)
                          for t, s in zip(eval_times, surv_vals)}
    except Exception as exc:
        errors.append(f"ensemble_node: {exc}")
        ensemble_risk  = 0.0
        median_surv    = 0.0
        ensemble_curve = {}
        eval_times     = []
        surv_matrix    = np.zeros((1, 1))

    return {
        **state,
        "ensemble_risk_score":     ensemble_risk,
        "ensemble_survival_curve": ensemble_curve,
        "predicted_median_survival": median_surv,
        "_surv_matrix": surv_matrix.tolist(),
        "_eval_times":  list(eval_times),
        "errors": errors,
    }


def conformal_node(state: OmicsState) -> OmicsState:
    cfg    = get_config()
    errors = list(state.get("errors") or [])
    lower: dict = {}
    upper: dict = {}
    try:
        conformal = _load_model("conformal", cfg.conformal_path)
        eval_times = state.get("_eval_times", [])
        surv_m = np.array(state.get("_surv_matrix", [[0.5]]), dtype=np.float32)

        lo, hi = conformal.predict_interval(surv_m)  # (1, T)
        lower = {int(t): round(float(lo[0, i]), 4) for i, t in enumerate(eval_times)}
        upper = {int(t): round(float(hi[0, i]), 4) for i, t in enumerate(eval_times)}
    except Exception as exc:
        errors.append(f"conformal_node: {exc}")
    return {**state, "conformal_lower": lower, "conformal_upper": upper, "errors": errors}
