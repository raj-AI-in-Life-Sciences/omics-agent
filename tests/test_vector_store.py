"""6 tests: FAISS indexing, retrieval, hybrid scores."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def tmp_index(tmp_path, embeddings, patient_metadata):
    from omics_agent.vector_store.indexer import PatientIndexer
    idx_path = str(tmp_path / "test.faiss")
    db_path  = str(tmp_path / "test.db")
    indexer  = PatientIndexer(idx_path, db_path)
    indexer.build(embeddings, patient_metadata)
    return indexer, embeddings, patient_metadata


def test_index_builds_correct_size(tmp_index):
    indexer, emb, meta = tmp_index
    assert indexer.n_patients == len(emb)


def test_search_returns_k_results(tmp_index):
    indexer, emb, _ = tmp_index
    scores, indices = indexer.search(emb[0], k=5)
    assert len(scores) == 5
    assert len(indices) == 5


def test_search_top1_is_self(tmp_index):
    indexer, emb, _ = tmp_index
    scores, indices = indexer.search(emb[0], k=1)
    assert int(indices[0]) == 0


def test_metadata_retrieval(tmp_index, patient_metadata):
    indexer, emb, meta = tmp_index
    scores, indices = indexer.search(emb[0], k=3)
    rows = indexer.get_metadata(indices)
    assert len(rows) == 3
    assert "patient_id" in rows[0]


def test_retriever_returns_hybrid_score(tmp_index, embeddings):
    indexer, emb, _ = tmp_index
    from omics_agent.vector_store.retriever import PatientRetriever
    retriever = PatientRetriever(indexer)
    results = retriever.retrieve(
        query_embedding    = emb[0],
        query_stage        = 2,
        query_clinical_text= "age 55 stage 2 subtype LumA",
        top_k              = 5,
    )
    assert len(results) <= 5
    for r in results:
        assert "hybrid_score" in r


def test_retriever_cohort_stats(tmp_index, embeddings, patient_metadata):
    indexer, emb, meta = tmp_index
    from omics_agent.vector_store.retriever import PatientRetriever
    retriever = PatientRetriever(indexer)
    results   = retriever.retrieve(emb[1], query_stage=2, query_clinical_text="", top_k=5)
    stats     = retriever.cohort_stats(results)
    assert "n" in stats
    assert stats["n"] >= 0
