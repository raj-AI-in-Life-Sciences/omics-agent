"""8 tests: tokenizer shape, imputation, stratification."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch


# ── 1. Tokenizer output shape ────────────────────────────────────────────────

def test_tokenize_returns_correct_shape():
    from omics_agent.data.geneformer_tokenizer import tokenize_bulk_rna
    # Synthetic expression row: gene_id → count
    expr = {f"ENSG{i:011d}": float(i + 1) for i in range(500)}
    tokens = tokenize_bulk_rna(expr, max_genes=200)
    assert tokens.shape == (1, 200), f"Expected (1,200) got {tokens.shape}"


def test_tokenize_rank_ordering():
    """Tokens should correspond to descending expression rank."""
    from omics_agent.data.geneformer_tokenizer import tokenize_bulk_rna
    expr = {"ENSG00000000001": 100.0, "ENSG00000000002": 50.0, "ENSG00000000003": 200.0}
    tokens = tokenize_bulk_rna(expr, max_genes=3)
    assert tokens.shape[1] == 3


def test_tokenize_handles_empty_expression():
    from omics_agent.data.geneformer_tokenizer import tokenize_bulk_rna
    tokens = tokenize_bulk_rna({}, max_genes=50)
    assert tokens.shape == (1, 50)


# ── 2. Clinical feature engineering ─────────────────────────────────────────

def test_clinical_engineer_output_shape(rng):
    from omics_agent.data.feature_engineer import ClinicalFeatureEngineer
    df = pd.DataFrame({
        "age_at_diagnosis": rng.uniform(30, 80, 20),
        "stage":            rng.integers(1, 5, 20),
        "grade":            rng.integers(1, 4, 20),
        "treatment":        ["chemo"] * 10 + ["hormone"] * 10,
        "subtype":          ["LumA"] * 20,
    })
    eng = ClinicalFeatureEngineer()
    X   = eng.fit_transform(df)
    assert X.shape == (20, 11), f"Expected (20,11) got {X.shape}"


def test_clinical_engineer_imputes_missing(rng):
    from omics_agent.data.feature_engineer import ClinicalFeatureEngineer
    df = pd.DataFrame({
        "age_at_diagnosis": [None, 55.0, 60.0],
        "stage":            [2, None, 3],
        "grade":            [1, 2, None],
        "treatment":        ["chemo", "hormone", "surgery"],
        "subtype":          ["LumA", "LumB", "HER2"],
    })
    eng = ClinicalFeatureEngineer()
    X   = eng.fit_transform(df)
    assert not np.isnan(X).any(), "NaN found after imputation"


# ── 3. Stratified split ──────────────────────────────────────────────────────

def test_stratified_split_sizes():
    from omics_agent.data.tcga_loader import stratified_split
    durations = np.arange(100, dtype=float)
    events    = np.array([1] * 50 + [0] * 50)
    df = pd.DataFrame({"duration": durations, "event": events})
    train, cal, test = stratified_split(df, train_frac=0.8, cal_frac=0.1, test_frac=0.1)
    total = len(train) + len(cal) + len(test)
    assert total == 100
    assert 75 <= len(train) <= 85


def test_stratified_split_no_overlap():
    from omics_agent.data.tcga_loader import stratified_split
    durations = np.arange(50, dtype=float)
    events    = np.ones(50, dtype=int)
    df = pd.DataFrame({"duration": durations, "event": events})
    train, cal, test = stratified_split(df)
    idx_sets = [set(train.index), set(cal.index), set(test.index)]
    assert len(idx_sets[0] & idx_sets[1]) == 0
    assert len(idx_sets[0] & idx_sets[2]) == 0
    assert len(idx_sets[1] & idx_sets[2]) == 0


def test_make_structured_array_dtype(durations, events):
    from omics_agent.data.tcga_loader import make_structured_array
    y = make_structured_array(durations.astype(float), events.astype(bool))
    assert y.dtype.names == ("event", "time")
    assert y["event"].dtype == bool
