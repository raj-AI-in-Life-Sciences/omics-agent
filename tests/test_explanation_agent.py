"""4 tests: prompt construction, LLM mock parsing."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


def _mock_llm_response(text: str):
    """Return a mock LangChain response object."""
    resp = MagicMock()
    resp.content = text
    return resp


def test_explanation_node_returns_string():
    """explanation_node should always return a non-empty string."""
    from omics_agent.agents.explanation_agent import explanation_node
    state = {
        "patient_id": "TEST-001",
        "predicted_median_survival": 36.0,
        "conformal_lower": {24: 0.55, 60: 0.30},
        "conformal_upper": {24: 0.75, 60: 0.50},
        "shap_values_ensemble": [
            {"feature": "embed_0", "shap": 0.23, "direction": "↑risk"},
        ],
        "similar_patients": [{"survival_months": 30}] * 5,
        "cohort_median_survival": 28.0,
        "cohort_event_rate": 0.65,
        "clinical_features": {"age_at_diagnosis": 55, "stage": 2, "subtype": "LumA"},
        "_explanation_retries": 0,
        "errors": [],
    }

    mock_text = (
        "Para 1: survival is approximately 36 months. "
        "Para 2: embed_0 (SHAP=0.23) drives risk. "
        "Para 3: cohort of 5 has median 28 months. "
        "Para 4: consider oncology consultation."
    )

    with patch("omics_agent.agents.explanation_agent.get_config") as mock_cfg:
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = _mock_llm_response(mock_text)
        mock_cfg.return_value.get_llm_client.return_value = mock_llm
        result = explanation_node(state)

    assert isinstance(result["llm_explanation"], str)
    assert len(result["llm_explanation"]) > 10


def test_explanation_node_increments_retries():
    from omics_agent.agents.explanation_agent import explanation_node
    state = {
        "predicted_median_survival": 24.0,
        "conformal_lower": {},
        "conformal_upper": {},
        "shap_values_ensemble": [],
        "similar_patients": [],
        "cohort_median_survival": 0,
        "cohort_event_rate": 0,
        "clinical_features": {},
        "_explanation_retries": 1,
        "errors": [],
    }
    with patch("omics_agent.agents.explanation_agent.get_config") as mock_cfg:
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = _mock_llm_response("Explanation text.")
        mock_cfg.return_value.get_llm_client.return_value = mock_llm
        result = explanation_node(state)

    assert result["_explanation_retries"] == 2


def test_explanation_node_fallback_on_llm_error():
    """Falls back to minimal explanation if LLM raises."""
    from omics_agent.agents.explanation_agent import explanation_node
    state = {
        "predicted_median_survival": 18.0,
        "conformal_lower": {},
        "conformal_upper": {},
        "shap_values_ensemble": [],
        "similar_patients": [],
        "cohort_median_survival": 0,
        "cohort_event_rate": 0,
        "clinical_features": {},
        "_explanation_retries": 0,
        "errors": [],
    }
    with patch("omics_agent.agents.explanation_agent.get_config") as mock_cfg:
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = RuntimeError("LLM unavailable")
        mock_cfg.return_value.get_llm_client.return_value = mock_llm
        result = explanation_node(state)

    assert "18.0" in result["llm_explanation"]
    assert len(result["errors"]) >= 1


def test_explanation_includes_revision_block():
    """When retries > 0 and judge_feedback is set, revision block is in prompt."""
    from omics_agent.agents import explanation_agent as ea

    captured_prompts = []

    def _capture_invoke(messages, **kwargs):
        for m in messages:
            captured_prompts.append(m.content)
        return _mock_llm_response("Revised explanation.")

    state = {
        "predicted_median_survival": 30.0,
        "conformal_lower": {},
        "conformal_upper": {},
        "shap_values_ensemble": [],
        "similar_patients": [],
        "cohort_median_survival": 0,
        "cohort_event_rate": 0,
        "clinical_features": {},
        "_explanation_retries": 1,
        "judge_feedback": "Please cite SHAP values explicitly.",
        "errors": [],
    }
    with patch("omics_agent.agents.explanation_agent.get_config") as mock_cfg:
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = _capture_invoke
        mock_cfg.return_value.get_llm_client.return_value = mock_llm
        ea.explanation_node(state)

    full_prompt = " ".join(captured_prompts)
    assert "Revision request" in full_prompt or "revision" in full_prompt.lower()
