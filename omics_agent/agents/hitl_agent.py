"""
HITL node: surfaces draft report for clinician review via interrupt().

The graph is compiled with interrupt_before=["hitl"], so execution
pauses before this node runs.  The FastAPI /resume endpoint resumes
the graph by calling graph.invoke(Command(resume=clinician_decision),
config={"configurable": {"thread_id": run_id}}).

hitl_status values:
    PENDING    — initial state (never actually stored; interrupt prevents it)
    APPROVED   — clinician approved the report
    OVERRIDDEN — clinician approved with edits (clinician_note set)
    REJECTED   — clinician rejected (no report issued)
    RE_EXPLAIN — clinician requests a new explanation with their note
"""

from __future__ import annotations

from langgraph.types import interrupt, Command

from ..graph.state import OmicsState


def hitl_node(state: OmicsState) -> OmicsState:
    """
    Called after the graph is resumed from interrupt.
    The resume value from Command is expected to be a dict:
        {"status": "APPROVED" | "OVERRIDDEN" | "REJECTED" | "RE_EXPLAIN",
         "clinician_note": "<optional text>"}
    """
    # interrupt() suspends here on the FIRST entry (interrupt_before fires
    # before this node, so this body only runs on resume).
    decision = interrupt(
        {
            "patient_id":            state.get("patient_id"),
            "predicted_median_survival": state.get("predicted_median_survival"),
            "llm_explanation":       state.get("llm_explanation"),
            "judge_verdict":         state.get("judge_verdict"),
            "judge_scores":          state.get("judge_scores"),
        }
    )

    status         = decision.get("status", "APPROVED")
    clinician_note = decision.get("clinician_note")

    return {
        **state,
        "hitl_status":    status,
        "clinician_note": clinician_note,
    }
