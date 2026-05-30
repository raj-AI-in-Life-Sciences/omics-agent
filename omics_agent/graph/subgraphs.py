"""
Survival subgraph: cox → rsf → deepsurv → xgb → ensemble → conformal.

Compiled as a separate StateGraph and added to the main graph via
add_node("survival", build_survival_subgraph().compile()).
"""

from __future__ import annotations

from langgraph.graph import StateGraph, END

from .state import OmicsState


def build_survival_subgraph() -> StateGraph:
    """
    Returns an uncompiled StateGraph for the survival pipeline.
    All nodes are imported lazily to avoid circular imports and to keep
    the subgraph self-contained.
    """
    from ..agents.survival_agent import (
        cox_node,
        rsf_node,
        deepsurv_node,
        xgb_node,
        ensemble_node,
        conformal_node,
    )

    sg = StateGraph(OmicsState)
    sg.add_node("cox",       cox_node)
    sg.add_node("rsf",       rsf_node)
    sg.add_node("deepsurv",  deepsurv_node)
    sg.add_node("xgb",       xgb_node)
    sg.add_node("ensemble",  ensemble_node)
    sg.add_node("conformal", conformal_node)

    sg.set_entry_point("cox")
    sg.add_edge("cox",      "rsf")
    sg.add_edge("rsf",      "deepsurv")
    sg.add_edge("deepsurv", "xgb")
    sg.add_edge("xgb",      "ensemble")
    sg.add_edge("ensemble", "conformal")
    sg.add_edge("conformal", END)

    return sg
