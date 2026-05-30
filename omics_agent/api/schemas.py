"""Pydantic request/response schemas for the OmicsAgent API."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    patient_id:        str
    rna_seq_path:      str = Field(default="")
    clinical_features: Dict[str, Any] = Field(default_factory=dict)


class ResumeRequest(BaseModel):
    run_id:        str
    status:        str = Field(
        description="APPROVED | OVERRIDDEN | REJECTED | RE_EXPLAIN"
    )
    clinician_note: Optional[str] = None


class SurvivalCurvePoint(BaseModel):
    time_months:  int
    survival_prob: float


class PredictResponse(BaseModel):
    run_id:    str
    status:    str   # "running" | "awaiting_hitl" | "complete" | "error"
    message:   str
    trace_url: Optional[str] = None


class StatusResponse(BaseModel):
    run_id:              str
    status:              str
    predicted_median_survival_months: Optional[float] = None
    judge_verdict:       Optional[str] = None
    hitl_status:         Optional[str] = None
    llm_explanation:     Optional[str] = None
    final_report:        Optional[Dict[str, Any]] = None
    errors:              List[str] = Field(default_factory=list)


class ResumeResponse(BaseModel):
    run_id:  str
    status:  str
    message: str
