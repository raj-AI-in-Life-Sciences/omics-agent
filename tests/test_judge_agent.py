"""4 tests: JSON parse, score range, pass threshold, fail-safe."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest


def _mock_judge_response(scores: dict, feedback: str = ""):
    resp = MagicMock()
    resp.content = json.dumps({**scores, "feedback": feedback})
    return resp


_GOOD_SCORES = {
    "evidence_grounding": 4,
    "uncertainty_acknowledgment": 4,
    "clinical_actionability": 4,
    "appropriate_hedging": 4,
    "factual_consistency": 4,
}

_WEAK_SCORES = {
    "evidence_grounding": 2,
    "uncertainty_acknowledgment": 2,
    "clinical_actionability": 2,
    "appropriate_hedging": 2,
    "factual_consistency": 2,
}


def test_judge_pass_on_good_scores():
    from omics_agent.agents.judge_agent import judge_node
    state = {
        "llm_explanation": "Para 1 with 90% CI. Para 2 SHAP=0.3. Para 3 cohort. Para 4 consult.",
        "predicted_median_survival": 36.0,
        "conformal_lower": {24: 0.6},
        "conformal_upper": {24: 0.8},
        "errors": [],
    }
    with patch("omics_agent.agents.judge_agent.get_config") as mock_cfg:
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = _mock_judge_response(_GOOD_SCORES, "All good.")
        mock_cfg.return_value.get_llm_client.return_value = mock_llm
        mock_cfg.return_value.llm_provider = "openai"
        result = judge_node(state)

    assert result["judge_verdict"] == "PASS"
    assert result["judge_scores"]["uncertainty_acknowledgment"] == 4


def test_judge_revise_on_weak_scores():
    from omics_agent.agents.judge_agent import judge_node
    state = {
        "llm_explanation": "Patient may survive.",
        "predicted_median_survival": 24.0,
        "conformal_lower": {},
        "conformal_upper": {},
        "errors": [],
    }
    with patch("omics_agent.agents.judge_agent.get_config") as mock_cfg:
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = _mock_judge_response(_WEAK_SCORES, "Needs more citations.")
        mock_cfg.return_value.get_llm_client.return_value = mock_llm
        mock_cfg.return_value.llm_provider = "openai"
        result = judge_node(state)

    assert result["judge_verdict"] == "REVISE"


def test_judge_failsafe_on_bad_json():
    """Malformed JSON should default to REVISE without crashing."""
    from omics_agent.agents.judge_agent import judge_node
    state = {
        "llm_explanation": "Some text.",
        "predicted_median_survival": 12.0,
        "conformal_lower": {},
        "conformal_upper": {},
        "errors": [],
    }
    bad_resp = MagicMock()
    bad_resp.content = "NOT VALID JSON {{{"
    with patch("omics_agent.agents.judge_agent.get_config") as mock_cfg:
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = bad_resp
        mock_cfg.return_value.get_llm_client.return_value = mock_llm
        mock_cfg.return_value.llm_provider = "openai"
        result = judge_node(state)

    assert result["judge_verdict"] == "REVISE"
    assert len(result["errors"]) >= 1


def test_judge_score_range():
    """All scores should be integers in [1, 5]."""
    from omics_agent.agents.judge_agent import judge_node
    scores = {
        "evidence_grounding": 5,
        "uncertainty_acknowledgment": 5,
        "clinical_actionability": 3,
        "appropriate_hedging": 4,
        "factual_consistency": 5,
    }
    state = {
        "llm_explanation": "Full explanation with all criteria met.",
        "predicted_median_survival": 48.0,
        "conformal_lower": {},
        "conformal_upper": {},
        "errors": [],
    }
    with patch("omics_agent.agents.judge_agent.get_config") as mock_cfg:
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = _mock_judge_response(scores)
        mock_cfg.return_value.get_llm_client.return_value = mock_llm
        mock_cfg.return_value.llm_provider = "openai"
        result = judge_node(state)

    for k, v in result["judge_scores"].items():
        assert 1 <= v <= 5, f"Score {k}={v} out of [1,5]"
