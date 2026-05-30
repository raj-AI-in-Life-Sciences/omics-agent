"""
Retrieval node: query FAISS vector store → hybrid rerank → cohort stats.
"""

from __future__ import annotations

import numpy as np

from ..graph.state import OmicsState
from ..core.config import get_config


def retrieval_node(state: OmicsState) -> OmicsState:
    cfg    = get_config()
    errors = list(state.get("errors") or [])
    patients: list = []
    cohort_median = 0.0
    cohort_event_rate = 0.0

    try:
        from ..vector_store.indexer   import PatientIndexer
        from ..vector_store.retriever import PatientRetriever

        indexer = PatientIndexer(cfg.faiss_index_path, cfg.faiss_db_path)
        indexer.load()

        retriever = PatientRetriever(indexer)

        embedding = np.array(
            state.get("geneformer_embedding") or [0.0] * 256,
            dtype=np.float32,
        )
        clin = state.get("clinical_features") or {}
        stage = int(clin.get("stage", 0))

        clin_text = (
            f"age {clin.get('age_at_diagnosis', '')} "
            f"stage {clin.get('stage', '')} "
            f"subtype {clin.get('subtype', '')} "
            f"treatment {clin.get('treatment', '')}"
        )

        patients = retriever.retrieve(
            query_embedding    = embedding,
            query_stage        = stage,
            query_clinical_text= clin_text,
            top_k              = cfg.retrieval_top_k,
        )

        stats = retriever.cohort_stats(patients)
        cohort_median     = stats["median_survival_months"] or 0.0
        cohort_event_rate = stats["event_rate"] or 0.0

    except Exception as exc:
        errors.append(f"retrieval_node: {exc}")

    return {
        **state,
        "similar_patients":       patients,
        "cohort_median_survival": cohort_median,
        "cohort_event_rate":      cohort_event_rate,
        "errors": errors,
    }
