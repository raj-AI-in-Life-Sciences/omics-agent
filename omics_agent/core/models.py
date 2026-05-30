"""
Pydantic domain models shared across the OmicsAgent pipeline.
"""

from __future__ import annotations

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class PatientRecord(BaseModel):
    patient_id: str
    rna_seq_path: str                   # path to row-vector CSV
    clinical_features: Dict[str, float | str | int]


class SurvivalPrediction(BaseModel):
    model_name: str
    risk_score: float
    survival_curve: Dict[int, float]    # {month: S(t)}
    c_index: Optional[float] = None


class EnsemblePrediction(BaseModel):
    survival_curve: Dict[int, float]
    predicted_median_months: float
    conformal_lower: Dict[int, float]
    conformal_upper: Dict[int, float]
    model_weights: Dict[str, float]


class PatientMatch(BaseModel):
    patient_id: str
    faiss_idx: int
    similarity_score: float
    hybrid_score: float
    survival_months: float
    event: int
    stage: str
    subtype: Optional[str] = None


class SHAPResult(BaseModel):
    model_name: str
    feature_names: List[str]
    shap_values: List[float]            # aligned with feature_names


class EnsembleSHAP(BaseModel):
    feature_names: List[str]
    shap_values: List[float]
    top_features: List[Dict]            # [{name, shap, direction, models}]


class JudgeResult(BaseModel):
    evidence_grounding: int = Field(ge=1, le=5)
    uncertainty_acknowledgment: int = Field(ge=1, le=5)
    clinical_actionability: int = Field(ge=1, le=5)
    appropriate_hedging: int = Field(ge=1, le=5)
    factual_consistency: int = Field(ge=1, le=5)
    overall: float
    verdict: str                        # PASS | REVISE | FAIL
    feedback: str

    @property
    def passes(self) -> bool:
        return (
            self.overall >= 3.5
            and self.uncertainty_acknowledgment >= 4
            and self.factual_consistency >= 4
        )


class ClinicalReport(BaseModel):
    patient_id: str
    predicted_median_months: float
    conformal_interval: Dict[str, Dict[int, float]]   # lower/upper → {month: S(t)}
    top_shap_features: List[Dict]
    cohort_summary: Dict
    llm_explanation: str
    judge_scores: Dict[str, int | float | str]
    hitl_status: str
    clinician_note: Optional[str] = None
    run_id: str
    trace_url: Optional[str] = None
