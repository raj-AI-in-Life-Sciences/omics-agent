"""
OmicsState — single TypedDict contract shared across all LangGraph nodes.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from typing_extensions import TypedDict


class OmicsState(TypedDict, total=False):
    # ── Input ────────────────────────────────────────────────────────────
    patient_id:        str
    rna_seq_path:      str
    clinical_features: Dict[str, Any]

    # ── Geneformer ───────────────────────────────────────────────────────
    gene_tokens:          List[int]    # rank-ordered, len=2048
    geneformer_embedding: List[float]  # [CLS] dim=256

    # ── Survival ensemble ────────────────────────────────────────────────
    cox_risk_score:          float
    rsf_survival_function:   Dict[int, float]   # {timepoint_months: S(t)}
    deepsurv_risk_score:     float
    xgb_risk_score:          float
    ensemble_risk_score:     float
    ensemble_survival_curve: Dict[int, float]
    conformal_lower:         Dict[int, float]
    conformal_upper:         Dict[int, float]
    predicted_median_survival: float            # months

    # ── Retrieval ────────────────────────────────────────────────────────
    similar_patients:       List[Dict[str, Any]]   # [{patient_id, similarity, ...}]
    cohort_median_survival: float
    cohort_event_rate:      float

    # ── SHAP ─────────────────────────────────────────────────────────────
    shap_values_cox:      Dict[str, float]
    shap_values_rsf:      Dict[str, float]
    shap_values_deepsurv: Dict[str, float]
    shap_values_xgb:      Dict[str, float]
    shap_values_ensemble: List[Dict[str, Any]]  # top-15 [{feature, shap, direction}]

    # ── Explanation + Judge ──────────────────────────────────────────────
    llm_explanation:         str
    judge_scores:            Dict[str, float]    # {criterion: 1-5}
    judge_verdict:           str                 # PASS | REVISE | FAIL
    judge_feedback:          str
    _explanation_retries:    int

    # ── HITL ─────────────────────────────────────────────────────────────
    hitl_status:    str              # PENDING | APPROVED | OVERRIDDEN | REJECTED
    clinician_note: Optional[str]
    final_report:   Dict[str, Any]

    # ── Meta ─────────────────────────────────────────────────────────────
    run_id:    str
    trace_url: Optional[str]
    errors:    List[str]
