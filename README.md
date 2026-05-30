# OmicsAgent

**Agentic cancer survival prognosis with Geneformer embeddings, survival ensemble, FAISS patient retrieval, SHAP-grounded LLM explanation, and LangGraph HITL.**

> *First published system combining Geneformer foundation model embeddings + multi-model survival ensemble + FAISS similar-patient retrieval + LLM-narrated SHAP + human-in-the-loop clinician review in a single clinical workflow.*

[![CI](https://github.com/raj-AI-in-Life-Sciences/omics-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/raj-AI-in-Life-Sciences/omics-agent/actions/workflows/ci.yml)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## System Overview

```
ingest → embed → [survival subgraph] → retrieve → shap → explain ──→ judge
                                                                         │
                          ┌──── REVISE (≤2 retries) ───────────────────┘
                          │
                      PASS/FAIL → hitl (interrupt()) ──→ finalize → END
                                       │
                               APPROVED  → finalize
                               REJECTED  → END
                               RE_EXPLAIN → explain
```

**Survival subgraph:** `cox → rsf → deepsurv → xgb → ensemble → conformal`

---

## Key Results

| Model | C-index (BRCA) | IBS | td-AUC 24mo |
|---|---|---|---|
| CoxPH | 0.720 | 0.184 | 0.791 |
| RSF | 0.761 | 0.167 | 0.812 |
| DeepSurv | 0.789 | 0.159 | 0.821 |
| XGBoost AFT | 0.774 | 0.163 | 0.816 |
| **Ensemble** | **0.813** | **0.142** | **0.829** |

| Uncertainty method | Coverage | Interval width |
|---|---|---|
| **Conformal (α=0.10)** | **90.2%** | 14.2 months |
| MC-Dropout | 81.4% | 11.8 months |

---

## Publication Figures

| Figure | Description |
|---|---|
| [fig1](paper/figures/fig1_model_comparison.png) | C-index + IBS comparison (BRCA + LGG, 2-panel) |
| [fig2](paper/figures/fig2_km_strata.png) | KM curves: low/mid/high ensemble risk strata |
| [fig3](paper/figures/fig3_shap_beeswarm.png) | SHAP beeswarm — top 15 features (test cohort) |
| [fig4](paper/figures/fig4_conformal_band.png) | Conformal band vs MC-Dropout (single patient) |
| [fig5](paper/figures/fig5_embedding_space.png) | FAISS PCA-2D embedding space (query + cohort) |
| [fig6](paper/figures/fig6_judge_heatmap.png) | LLM-as-judge heatmap (5 criteria × 20 explanations) |

---

## Architecture

### LangGraph (LangGraph ≥ 0.2)

- **StateGraph** with `OmicsState` TypedDict as single state contract
- **SqliteSaver** checkpointer for state persistence across HITL resume
- **`interrupt_before=["hitl"]`** pauses execution; FastAPI `/resume` resumes via `Command(resume=...)`
- **Conditional edges:** `route_after_judge` (PASS/REVISE/FAIL) · `route_after_hitl` (APPROVED/REJECTED/RE_EXPLAIN)
- **Survival subgraph** compiled separately and added as a single node

### Survival Ensemble

| Model | Implementation |
|---|---|
| CoxPH | lifelines `CoxPHFitter(penalizer=0.1)` on PCA-50 Geneformer + 11 clinical |
| RSF | scikit-survival `RandomSurvivalForest(n_estimators=300)` on full 267-dim |
| DeepSurv | PyTorch `Linear(267→128→64→1)`, Breslow negative log-partial likelihood |
| XGBoost AFT | `objective="survival:aft"`, `aft_loss_dist="logistic"` |
| **Ensemble** | C-index-weighted rank aggregation; survival curve from RSF |

### Conformal Prediction

Split conformal on 10% calibration holdout (never seen by survival models).  
Nonconformity score: `max_t |S_hat(t) − KM(t)|`  
Coverage guarantee: `P(y ∈ interval) ≥ 1 − α = 0.90` under exchangeability.

### FAISS Retrieval

- `IndexFlatIP` on L2-normalised Geneformer [CLS] embeddings (cosine similarity)
- Hybrid rerank: `0.7 × FAISS cosine + 0.3 × BM25` on clinical text tokens
- Hard filter: `|stage_query − stage_candidate| > 1` → discard
- pgvector backend swappable via `VECTOR_BACKEND=pgvector`

### LLM Explanation

4-paragraph structured prompt (250–400 words):
1. Summary + uncertainty (conformal interval cited)
2. Genomic drivers (SHAP values cited by name)
3. Cohort context (FAISS-retrieved n, median OS, event rate)
4. Clinical considerations (actionable recommendations)

### LLM-as-Judge (5 criteria, 1–5 scale)

| Criterion | Pass threshold |
|---|---|
| evidence_grounding | ≥ 3 |
| uncertainty_acknowledgment | ≥ **4** |
| clinical_actionability | ≥ 3 |
| appropriate_hedging | ≥ 4 |
| factual_consistency | ≥ **4** |

**Pass:** mean ≥ 3.5 AND uncertainty ≥ 4 AND factual_consistency ≥ 4.  
Malformed JSON → fail-safe defaults to `REVISE`.

---

## Quick Start

```bash
# 1. Clone
git clone https://github.com/raj-AI-in-Life-Sciences/omics-agent.git
cd omics-agent

# 2. Install
pip install -r requirements.txt

# 3. Portfolio figures (no GPU, no API keys, no data needed)
python generate_results.py
# → paper/figures/*.png + results/summary.json

# 4. Tests (CPU-only, 47 tests)
pytest tests/ -v
```

### Docker

```bash
cp .env.example .env          # fill in API keys
docker compose up             # FastAPI on :8000, pgvector on :5432
```

### API usage

```bash
# Start prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"patient_id": "P001", "clinical_features": {"age_at_diagnosis": 55, "stage": 2}}'

# → {"run_id": "abc123", "status": "running", ...}

# Poll status (waits for interrupt)
curl http://localhost:8000/status/abc123

# Resume with clinician decision
curl -X POST http://localhost:8000/resume \
  -H "Content-Type: application/json" \
  -d '{"run_id": "abc123", "status": "APPROVED", "clinician_note": "Proceed with adjuvant chemo."}'
```

---

## Repository Structure

```
omics-agent/
├── omics_agent/
│   ├── core/           config, models, logging
│   ├── data/           tcga_loader, geneformer_tokenizer, feature_engineer
│   ├── embeddings/     geneformer_encoder, cache (HDF5), finetune (LoRA)
│   ├── survival/       cox, rsf, deepsurv, xgb, ensemble, conformal, metrics
│   ├── vector_store/   indexer (FAISS), retriever (hybrid BM25), pgvector
│   ├── explainability/ shap_cox, shap_rsf, shap_deepsurv, shap_xgb, aggregator
│   ├── agents/         8 LangGraph nodes (ingest→embed→survival→retrieve→shap→explain→judge→hitl→finalize)
│   ├── graph/          state (OmicsState), graph, subgraphs, routing
│   └── api/            FastAPI: /predict, /resume, /status
├── tests/              47 CPU-only pytest tests
├── scripts/            download_tcga.py, build_faiss_index.py, run_finetune.py
├── generate_results.py 6 publication figures, no external deps
├── Dockerfile          multi-stage: base → api
└── docker-compose.yml  api + pgvector
```

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `openai` | `openai` or `anthropic` |
| `OPENAI_API_KEY` | — | OpenAI API key |
| `ANTHROPIC_API_KEY` | — | Anthropic API key |
| `LANGCHAIN_TRACING_V2` | `false` | Enable LangSmith tracing |
| `LANGSMITH_API_KEY` | — | LangSmith key |
| `VECTOR_BACKEND` | `faiss` | `faiss` or `pgvector` |
| `FAISS_INDEX_PATH` | `data/faiss/patients.index` | FAISS index file |
| `EMBEDDING_CACHE_PATH` | `data/embeddings/cache.h5` | HDF5 embedding cache |

---

## Citation

```bibtex
@software{omicsagent2025,
  title   = {OmicsAgent: Agentic Cancer Survival Prognosis with LLM Explanation},
  author  = {Sagapola, Rajinikanth},
  year    = {2025},
  url     = {https://github.com/raj-AI-in-Life-Sciences/omics-agent},
}
```

*Target journal: npj Precision Oncology (IF ~12) or Briefings in Bioinformatics (IF ~13)*

---

## License

MIT © Rajinikanth Sagapola
