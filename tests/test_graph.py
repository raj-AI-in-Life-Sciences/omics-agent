"""6 tests: graph compilation, node output shapes, routing logic."""

from __future__ import annotations

import pytest


def test_graph_compiles():
    """Graph should compile without errors (no API keys needed)."""
    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db = f.name
    try:
        from omics_agent.graph.graph import build_graph
        g = build_graph(db_path=db)
        assert g is not None
    finally:
        os.unlink(db)


def test_graph_has_correct_nodes():
    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db = f.name
    try:
        from omics_agent.graph.graph import build_graph
        g = build_graph(db_path=db)
        # Compiled graph exposes _graph attribute with nodes dict
        node_names = set(g.get_graph().nodes.keys())
        for expected in ["ingest", "embed", "survival", "retrieve",
                         "shap", "explain", "judge", "hitl", "finalize"]:
            assert expected in node_names, f"Node '{expected}' not found"
    finally:
        os.unlink(db)


def test_route_after_judge_pass():
    from omics_agent.graph.routing import route_after_judge
    state = {"judge_verdict": "PASS", "_explanation_retries": 0}
    assert route_after_judge(state) == "hitl"


def test_route_after_judge_revise_under_limit():
    from omics_agent.graph.routing import route_after_judge
    state = {"judge_verdict": "REVISE", "_explanation_retries": 1}
    assert route_after_judge(state) == "explain"


def test_route_after_judge_revise_at_limit():
    from omics_agent.graph.routing import route_after_judge
    state = {"judge_verdict": "REVISE", "_explanation_retries": 2}
    assert route_after_judge(state) == "hitl"


def test_route_after_hitl_approved():
    from omics_agent.graph.routing import route_after_hitl
    state = {"hitl_status": "APPROVED"}
    assert route_after_hitl(state) == "finalize"


def test_route_after_hitl_rejected():
    from omics_agent.graph.routing import route_after_hitl
    state = {"hitl_status": "REJECTED"}
    assert route_after_hitl(state) == "__end__"
