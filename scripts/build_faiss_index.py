"""
Build FAISS index from TCGA patient embeddings.

Usage:
    python scripts/build_faiss_index.py \
        --rna  data/processed/brca_rna.csv \
        --clin data/processed/brca_clinical.csv \
        --index data/faiss/patients.index \
        --db    data/faiss/patients.db
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import torch

from omics_agent.data.tcga_loader import load_tcga, stratified_split
from omics_agent.data.feature_engineer import ClinicalFeatureEngineer
from omics_agent.embeddings.geneformer_encoder import GeneformerEncoder
from omics_agent.embeddings.cache import EmbeddingCache
from omics_agent.vector_store.indexer import PatientIndexer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rna",   required=True)
    parser.add_argument("--clin",  required=True)
    parser.add_argument("--index", default="data/faiss/patients.index")
    parser.add_argument("--db",    default="data/faiss/patients.db")
    parser.add_argument("--cache", default="data/embeddings/cache.h5")
    parser.add_argument("--batch", type=int, default=32)
    args = parser.parse_args()

    print("Loading TCGA data...")
    df = load_tcga(args.rna, args.clin)
    train, cal, test = stratified_split(df)
    print(f"  Train: {len(train)}, Cal: {len(cal)}, Test: {len(test)}")

    encoder = GeneformerEncoder()
    cache   = EmbeddingCache(args.cache)
    eng     = ClinicalFeatureEngineer().fit(train)

    print("Encoding train patients...")
    embeddings = []
    metadata   = []

    for i, row in enumerate(train.itertuples()):
        pid = str(getattr(row, "patient_id", f"p{i}"))
        if cache.contains(pid):
            emb = cache.load(pid)
        else:
            from omics_agent.data.geneformer_tokenizer import tokenize_bulk_rna
            # RNA seq expected as a dict {gene_id: count} stored in a column
            rna_expr = getattr(row, "rna_expr", {})
            tokens   = tokenize_bulk_rna(rna_expr)
            emb      = encoder.encode(tokens)
            cache.save(pid, emb)

        embeddings.append(emb)
        metadata.append({
            "patient_id":      pid,
            "survival_months": float(getattr(row, "duration_months", 0)),
            "event":           int(getattr(row, "event", 0)),
            "stage":           int(getattr(row, "stage", 0)),
            "subtype":         str(getattr(row, "subtype", "")),
            "clinical_text":   f"age {getattr(row, 'age_at_diagnosis', '')} stage {getattr(row, 'stage', '')}",
        })
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(train)} encoded")

    emb_matrix = np.array(embeddings, dtype=np.float32)
    print(f"Building FAISS index ({len(emb_matrix)} patients, dim={emb_matrix.shape[1]})...")
    indexer = PatientIndexer(args.index, args.db)
    indexer.build(emb_matrix, metadata)
    print(f"Index written to {args.index}, DB to {args.db}")


if __name__ == "__main__":
    main()
