"""
Conditional edge routing functions for the OmicsAgent LangGraph.
"""

from __future__ import annotations

from .state import OmicsState

_MAX_RETRIES = 2


def route_after_judge(state: OmicsState) -> str:
    """
    After judge_agent runs:
      - PASS  → "hitl"
      - REVISE (retries remaining) → "explain"
      - REVISE (max retries reached) → "hitl"  (fallback — surface anyway)
      - FAIL   → "hitl"  (surface for clinician override)
    """
    verdict  = state.get("judge_verdict", "PASS")
    retries  = state.get("_explanation_retries", 0)

    if verdict == "PASS":
        return "hitl"
    if verdict == "REVISE" and retries < _MAX_RETRIES:
        return "explain"
    # REVISE exhausted or FAIL → surface to clinician
    return "hitl"


def route_after_hitl(state: OmicsState) -> str:
    """
    After hitl_agent receives clinician decision:
      - APPROVED / OVERRIDDEN → "finalize"
      - REJECTED              → END  (represented as "__end__")
      - re-explain            → "explain"
    """
    status = state.get("hitl_status", "APPROVED")
    if status in ("APPROVED", "OVERRIDDEN"):
        return "finalize"
    if status == "REJECTED":
        return "__end__"
    if status == "RE_EXPLAIN":
        return "explain"
    return "finalize"
