"""
Finalize node: assemble the final clinical report dict.
"""

from __future__ import annotations

import datetime

from ..graph.state import OmicsState


def finalize_node(state: OmicsState) -> OmicsState:
    """Compose the final_report dict from all state fields."""
    report = {
        "patient_id":               state.get("patient_id"),
        "run_id":                   state.get("run_id"),
        "timestamp":                datetime.datetime.utcnow().isoformat() + "Z",
        "predicted_median_survival_months": state.get("predicted_median_survival"),
        "conformal_lower":          state.get("conformal_lower"),
        "conformal_upper":          state.get("conformal_upper"),
        "ensemble_survival_curve":  state.get("ensemble_survival_curve"),
        "ensemble_risk_score":      state.get("ensemble_risk_score"),
        "cox_risk_score":           state.get("cox_risk_score"),
        "rsf_survival_function":    state.get("rsf_survival_function"),
        "deepsurv_risk_score":      state.get("deepsurv_risk_score"),
        "xgb_risk_score":           state.get("xgb_risk_score"),
        "similar_patients":         state.get("similar_patients"),
        "cohort_median_survival":   state.get("cohort_median_survival"),
        "cohort_event_rate":        state.get("cohort_event_rate"),
        "shap_top_features":        state.get("shap_values_ensemble"),
        "llm_explanation":          state.get("llm_explanation"),
        "judge_scores":             state.get("judge_scores"),
        "judge_verdict":            state.get("judge_verdict"),
        "hitl_status":              state.get("hitl_status"),
        "clinician_note":           state.get("clinician_note"),
        "errors":                   state.get("errors"),
        "trace_url":                state.get("trace_url"),
    }
    return {**state, "final_report": report}
