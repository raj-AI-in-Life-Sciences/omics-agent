"""
Main OmicsAgent LangGraph StateGraph.

Topology:
    ingest → embed → survival_subgraph → retrieve → shap
         → explain → judge ──PASS──→ hitl → finalize → END
                          └─REVISE─→ explain (≤2 retries)
                          └─FAIL───→ hitl
    hitl: APPROVED/OVERRIDDEN → finalize
          REJECTED            → END
          RE_EXPLAIN          → explain
"""

from __future__ import annotations

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver

from .state import OmicsState
from .routing import route_after_judge, route_after_hitl
from .subgraphs import build_survival_subgraph


def build_graph(db_path: str = "omics_checkpoints.db") -> "CompiledGraph":  # type: ignore[name-defined]
    """
    Build and compile the full OmicsAgent graph.

    Parameters
    ----------
    db_path : SQLite path for SqliteSaver checkpointer.

    Returns
    -------
    Compiled LangGraph graph with interrupt_before=["hitl"].
    """
    from ..agents.ingestion_agent   import ingestion_node
    from ..agents.embedding_agent   import embedding_node
    from ..agents.retrieval_agent   import retrieval_node
    from ..agents.shap_agent        import shap_node
    from ..agents.explanation_agent import explanation_node
    from ..agents.judge_agent       import judge_node
    from ..agents.hitl_agent        import hitl_node
    from ..agents.finalize_agent    import finalize_node

    survival_subgraph = build_survival_subgraph().compile()

    graph = StateGraph(OmicsState)

    # ── Nodes ────────────────────────────────────────────────────────────
    graph.add_node("ingest",   ingestion_node)
    graph.add_node("embed",    embedding_node)
    graph.add_node("survival", survival_subgraph)
    graph.add_node("retrieve", retrieval_node)
    graph.add_node("shap",     shap_node)
    graph.add_node("explain",  explanation_node)
    graph.add_node("judge",    judge_node)
    graph.add_node("hitl",     hitl_node)
    graph.add_node("finalize", finalize_node)

    # ── Linear backbone ──────────────────────────────────────────────────
    graph.set_entry_point("ingest")
    graph.add_edge("ingest",   "embed")
    graph.add_edge("embed",    "survival")
    graph.add_edge("survival", "retrieve")
    graph.add_edge("retrieve", "shap")
    graph.add_edge("shap",     "explain")
    graph.add_edge("explain",  "judge")
    graph.add_edge("finalize", END)

    # ── Conditional edges ────────────────────────────────────────────────
    graph.add_conditional_edges(
        "judge",
        route_after_judge,
        {"hitl": "hitl", "explain": "explain"},
    )
    graph.add_conditional_edges(
        "hitl",
        route_after_hitl,
        {"finalize": "finalize", "__end__": END, "explain": "explain"},
    )

    # ── Checkpointer + HITL interrupt ────────────────────────────────────
    checkpointer = SqliteSaver.from_conn_string(db_path)
    compiled = graph.compile(
        checkpointer=checkpointer,
        interrupt_before=["hitl"],
    )
    return compiled
