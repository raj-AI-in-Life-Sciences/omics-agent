"""
OmicsAgent FastAPI application.

Endpoints:
  POST /predict  — start a new prediction run (async, returns run_id)
  POST /resume   — resume a paused HITL run with clinician decision
  GET  /status/{run_id} — poll run status + final report
"""

from __future__ import annotations

import uuid
from typing import Any, Dict

from fastapi import FastAPI, HTTPException, BackgroundTasks
from langgraph.types import Command

from .schemas import (
    PredictRequest, PredictResponse,
    ResumeRequest,  ResumeResponse,
    StatusResponse,
)
from ..graph.graph import build_graph
from ..core.config import get_config

app = FastAPI(
    title="OmicsAgent",
    description="Agentic cancer survival prognosis with LLM explanation",
    version="1.0.0",
)

# ── Graph singleton ──────────────────────────────────────────────────────────
_graph = None


def get_graph():
    global _graph
    if _graph is None:
        cfg    = get_config()
        _graph = build_graph(db_path=cfg.checkpoint_db_path)
    return _graph


# ── In-memory run-status store (for demo; replace with Redis in production) ──
_run_status: Dict[str, Dict[str, Any]] = {}


def _run_prediction(run_id: str, initial_state: dict) -> None:
    """Background task: invoke graph until interrupt or END."""
    graph  = get_graph()
    config = {"configurable": {"thread_id": run_id}}
    try:
        for event in graph.stream(initial_state, config=config):
            pass  # events are persisted in SqliteSaver
        # If we reach here without interrupt the graph finished
        state = graph.get_state(config).values
        _run_status[run_id] = {
            "status":   "complete" if not state.get("errors") else "error",
            "state":    state,
        }
    except Exception as exc:
        _run_status[run_id] = {"status": "error", "error": str(exc), "state": {}}


# ── Routes ───────────────────────────────────────────────────────────────────

@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest, background_tasks: BackgroundTasks):
    run_id = str(uuid.uuid4())
    initial_state = {
        "patient_id":        req.patient_id,
        "rna_seq_path":      req.rna_seq_path,
        "clinical_features": req.clinical_features,
        "run_id":            run_id,
        "errors":            [],
        "_explanation_retries": 0,
    }
    _run_status[run_id] = {"status": "running", "state": {}}
    background_tasks.add_task(_run_prediction, run_id, initial_state)

    cfg = get_config()
    return PredictResponse(
        run_id   = run_id,
        status   = "running",
        message  = "Prediction started. Poll /status/{run_id} for updates.",
        trace_url= f"https://smith.langchain.com/" if cfg.langchain_tracing else None,
    )


@app.post("/resume", response_model=ResumeResponse)
async def resume(req: ResumeRequest, background_tasks: BackgroundTasks):
    graph  = get_graph()
    config = {"configurable": {"thread_id": req.run_id}}

    try:
        graph_state = graph.get_state(config)
    except Exception:
        raise HTTPException(status_code=404, detail=f"run_id {req.run_id} not found")

    if not graph_state.next:
        raise HTTPException(status_code=400, detail="Run is not awaiting HITL input")

    decision = {
        "status":        req.status,
        "clinician_note": req.clinician_note,
    }

    def _resume_run():
        try:
            for event in graph.stream(
                Command(resume=decision), config=config
            ):
                pass
            state = graph.get_state(config).values
            _run_status[req.run_id] = {
                "status": "complete" if req.status != "REJECTED" else "rejected",
                "state":  state,
            }
        except Exception as exc:
            _run_status[req.run_id] = {"status": "error", "error": str(exc), "state": {}}

    background_tasks.add_task(_resume_run)
    return ResumeResponse(
        run_id  = req.run_id,
        status  = "resuming",
        message = f"Run {req.run_id} resumed with status={req.status}",
    )


@app.get("/status/{run_id}", response_model=StatusResponse)
async def status(run_id: str):
    info = _run_status.get(run_id)
    if info is None:
        # Try loading from checkpointer
        graph  = get_graph()
        config = {"configurable": {"thread_id": run_id}}
        try:
            gs    = graph.get_state(config)
            state = gs.values
            _next = gs.next
            if _next:
                run_status = "awaiting_hitl"
            elif state:
                run_status = "complete"
            else:
                raise HTTPException(status_code=404, detail=f"run_id {run_id} not found")
            info = {"status": run_status, "state": state}
        except HTTPException:
            raise
        except Exception:
            raise HTTPException(status_code=404, detail=f"run_id {run_id} not found")

    state = info.get("state", {})
    return StatusResponse(
        run_id          = run_id,
        status          = info.get("status", "unknown"),
        predicted_median_survival_months = state.get("predicted_median_survival"),
        judge_verdict   = state.get("judge_verdict"),
        hitl_status     = state.get("hitl_status"),
        llm_explanation = state.get("llm_explanation"),
        final_report    = state.get("final_report"),
        errors          = state.get("errors") or [],
    )


@app.get("/health")
async def health():
    return {"status": "ok"}
