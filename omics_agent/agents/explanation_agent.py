"""
Explanation node: structured 4-paragraph LLM prognosis narration.

Prompt design:
  - System: oncology CDS AI, 250-400 word output, cite SHAP/cohort, hedge
  - User:   median survival, conformal interval, SHAP table, cohort stats,
            clinical context, optional clinician note
  - Output: 4 paragraphs — (1) summary+uncertainty, (2) genomic drivers,
            (3) cohort context, (4) clinical considerations
"""

from __future__ import annotations

from ..graph.state import OmicsState
from ..core.config import get_config


_SYSTEM_PROMPT = """\
You are an oncology clinical decision support AI assistant.
Generate a 4-paragraph prognosis explanation (250-400 words total).
Rules:
- Every survival claim MUST cite a SHAP value, a feature name, or a cohort statistic.
- Use probabilistic language: "suggests", "is associated with", "may indicate".
- Never make deterministic survival statements (avoid "will survive X months").
- Explicitly mention the 90% conformal prediction interval.
- Paragraph order: (1) Summary & uncertainty, (2) Genomic drivers,
  (3) Cohort context, (4) Clinical considerations.
"""


def _build_shap_table(shap_entries: list) -> str:
    if not shap_entries:
        return "No SHAP data available."
    header = "| Feature | SHAP | Direction |\n|---|---|---|"
    rows = "\n".join(
        f"| {e.get('feature','')} | {e.get('shap', 0):.4f} | {e.get('direction','')} |"
        for e in shap_entries[:15]
    )
    return f"{header}\n{rows}"


def _build_conformal_summary(state: OmicsState) -> str:
    lower = state.get("conformal_lower", {})
    upper = state.get("conformal_upper", {})
    if not lower or not upper:
        return "Conformal prediction interval: not available."
    times = sorted(lower.keys())
    t_24  = min(lower.keys(), key=lambda t: abs(t - 24), default=None)
    t_60  = min(lower.keys(), key=lambda t: abs(t - 60), default=None)
    parts = []
    if t_24 is not None:
        parts.append(f"24-month S(t) 90% CI: [{lower[t_24]:.2f}, {upper[t_24]:.2f}]")
    if t_60 is not None:
        parts.append(f"60-month S(t) 90% CI: [{lower[t_60]:.2f}, {upper[t_60]:.2f}]")
    return "; ".join(parts) if parts else "Conformal interval: not available."


def explanation_node(state: OmicsState) -> OmicsState:
    cfg    = get_config()
    errors = list(state.get("errors") or [])
    retries = int(state.get("_explanation_retries") or 0)

    clin   = state.get("clinical_features") or {}
    median = state.get("predicted_median_survival", 0.0)
    cohort_n      = len(state.get("similar_patients") or [])
    cohort_median = state.get("cohort_median_survival", 0.0)
    cohort_evrate = state.get("cohort_event_rate", 0.0)
    shap_table    = _build_shap_table(state.get("shap_values_ensemble") or [])
    conf_summary  = _build_conformal_summary(state)
    judge_fb      = state.get("judge_feedback", "")
    clinician_note = state.get("clinician_note", "")

    revision_block = ""
    if retries > 0 and judge_fb:
        revision_block = f"\n## Revision request\n{judge_fb}\nPlease address the above in the revised explanation.\n"

    clinician_block = ""
    if clinician_note:
        clinician_block = f"\n## Clinician note\n{clinician_note}\n"

    user_prompt = f"""\
## Predicted median survival: {median:.1f} months
## {conf_summary}
## SHAP top features
{shap_table}
## Cohort: {cohort_n} similar patients | median OS {cohort_median:.1f} months | {cohort_evrate*100:.1f}% 2-year event rate
## Clinical: age {clin.get('age_at_diagnosis', 'N/A')}, stage {clin.get('stage', 'N/A')}, subtype {clin.get('subtype', 'N/A')}
{clinician_block}{revision_block}
Write 4 paragraphs as instructed."""

    explanation = ""
    try:
        llm = cfg.get_llm_client()
        from langchain_core.messages import HumanMessage, SystemMessage
        response = llm.invoke([
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ])
        explanation = response.content.strip()
    except Exception as exc:
        errors.append(f"explanation_node: {exc}")
        explanation = (
            f"Predicted median survival: {median:.1f} months. "
            f"90% conformal interval available. "
            f"Top genomic driver: {(state.get('shap_values_ensemble') or [{}])[0].get('feature', 'N/A')}."
        )

    return {
        **state,
        "llm_explanation":      explanation,
        "_explanation_retries": retries + 1,
        "errors": errors,
    }
