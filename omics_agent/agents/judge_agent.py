"""
Judge node: 5-criterion LLM-as-judge rubric with JSON output.

Criteria (1-5 scale):
  evidence_grounding        — SHAP/cohort citation for every survival claim
  uncertainty_acknowledgment — conformal interval explicitly mentioned
  clinical_actionability    — closing paragraph has specific recommendations
  appropriate_hedging       — no deterministic survival statements
  factual_consistency       — numbers in text match the context provided

Pass: mean ≥ 3.5 AND uncertainty_acknowledgment ≥ 4 AND factual_consistency ≥ 4
Fail-safe: malformed JSON → verdict defaults to REVISE.
"""

from __future__ import annotations

import json

from ..graph.state import OmicsState
from ..core.config import get_config

_PASS_MEAN_THRESHOLD  = 3.5
_PASS_UNCERTAINTY_MIN = 4
_PASS_FACTUAL_MIN     = 4

_JUDGE_SYSTEM = """\
You are a senior oncology clinical AI evaluator. Score the following prognosis explanation.
Return ONLY valid JSON with this exact structure:
{
  "evidence_grounding": <1-5>,
  "uncertainty_acknowledgment": <1-5>,
  "clinical_actionability": <1-5>,
  "appropriate_hedging": <1-5>,
  "factual_consistency": <1-5>,
  "feedback": "<one-sentence explanation of the lowest score>"
}
Scoring guide:
  5 = exemplary, 4 = good with minor gap, 3 = adequate, 2 = notable gap, 1 = absent/wrong
"""


def _compute_verdict(scores: dict) -> str:
    if not scores:
        return "REVISE"
    mean = sum(scores[k] for k in [
        "evidence_grounding", "uncertainty_acknowledgment",
        "clinical_actionability", "appropriate_hedging", "factual_consistency"
    ]) / 5.0
    if (mean >= _PASS_MEAN_THRESHOLD
            and scores.get("uncertainty_acknowledgment", 0) >= _PASS_UNCERTAINTY_MIN
            and scores.get("factual_consistency", 0) >= _PASS_FACTUAL_MIN):
        return "PASS"
    if mean < 2.0:
        return "FAIL"
    return "REVISE"


def judge_node(state: OmicsState) -> OmicsState:
    cfg    = get_config()
    errors = list(state.get("errors") or [])
    scores: dict = {}
    verdict  = "REVISE"
    feedback = ""

    explanation = state.get("llm_explanation", "")
    if not explanation:
        errors.append("judge_node: no explanation to evaluate")
        return {**state, "judge_scores": scores, "judge_verdict": verdict,
                "judge_feedback": feedback, "errors": errors}

    median   = state.get("predicted_median_survival", 0.0)
    conf_lo  = state.get("conformal_lower", {})
    conf_hi  = state.get("conformal_upper", {})

    user_prompt = f"""\
## Context provided to the explanation system
Predicted median survival: {median:.1f} months
Conformal interval keys present: {bool(conf_lo and conf_hi)}
## Explanation to evaluate
{explanation}"""

    try:
        llm = cfg.get_llm_client()
        # Request JSON output
        from langchain_core.messages import HumanMessage, SystemMessage
        response = llm.invoke(
            [SystemMessage(content=_JUDGE_SYSTEM), HumanMessage(content=user_prompt)],
            response_format={"type": "json_object"} if cfg.llm_provider == "openai" else None,
        )
        raw = response.content.strip()
        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        parsed = json.loads(raw)
        scores = {
            k: int(parsed.get(k, 3))
            for k in [
                "evidence_grounding", "uncertainty_acknowledgment",
                "clinical_actionability", "appropriate_hedging", "factual_consistency",
            ]
        }
        feedback = str(parsed.get("feedback", ""))
        verdict  = _compute_verdict(scores)

    except Exception as exc:
        errors.append(f"judge_node: {exc} — defaulting to REVISE")
        verdict  = "REVISE"
        feedback = f"Judge failed to parse response: {exc}"

    return {
        **state,
        "judge_scores":   scores,
        "judge_verdict":  verdict,
        "judge_feedback": feedback,
        "errors": errors,
    }
