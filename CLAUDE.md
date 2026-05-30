# OmicsAgent — CLAUDE.md

## Project summary
LangGraph multi-agent system for cancer survival prognosis (TCGA-BRCA + LGG).
Pipeline: ingest → Geneformer embed → survival ensemble → FAISS retrieval → SHAP → LLM explain → LLM-as-judge → HITL → finalize.

## Key design decisions
- **LLM provider**: both OpenAI and Anthropic, configured via `LLM_PROVIDER=openai|anthropic` in .env
- **Geneformer**: pre-cached HDF5 embeddings; `generate_results.py` uses synthetic 256-dim Gaussians — no download needed
- **Survival models**: CoxPH (lifelines), RSF (scikit-survival), DeepSurv (PyTorch), XGBoost AFT
- **Ensemble**: C-index-weighted rank aggregation (not average of scores)
- **Conformal**: split conformal on 10% cal holdout; nonconformity = max_t |S_hat − KM|
- **FAISS**: IndexFlatIP + L2-norm = cosine; BM25 hybrid at 0.7/0.3; stage hard filter ±1
- **SHAP**: LinearExplainer(Cox), TreeExplainer(RSF, XGBoost), GradientExplainer(DeepSurv)
- **LangGraph HITL**: `interrupt_before=["hitl"]`; resume via `Command(resume=decision)`
- **Tests**: 47 CPU-only tests, no API keys, no TCGA data; all in `tests/`

## Run commands
```bash
pytest tests/ -v --tb=short          # 47 tests
python generate_results.py           # 6 figures, results/summary.json
uvicorn omics_agent.api.main:app     # FastAPI dev server
docker compose up                    # api + pgvector
```

## File ownership
- State contract: `omics_agent/graph/state.py` (OmicsState TypedDict)
- Graph wiring: `omics_agent/graph/graph.py`
- Routing logic: `omics_agent/graph/routing.py`
- LLM prompt: `omics_agent/agents/explanation_agent.py` (_SYSTEM_PROMPT, user_prompt)
- Judge rubric: `omics_agent/agents/judge_agent.py` (_JUDGE_SYSTEM, _compute_verdict)
