"""
Shared pytest fixtures: 10 synthetic patients, 100 gene features.
No GPU, no API keys, no TCGA data required.
"""

from __future__ import annotations

import numpy as np
import pytest


N_PATIENTS  = 10
N_EMBED     = 256
N_CLINICAL  = 11
N_FEATURES  = N_EMBED + N_CLINICAL   # 267


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def embeddings(rng):
    """(N_PATIENTS, N_EMBED) Geneformer [CLS] embeddings."""
    return rng.standard_normal((N_PATIENTS, N_EMBED)).astype(np.float32)


@pytest.fixture
def clinical_matrix(rng):
    """(N_PATIENTS, N_CLINICAL) clinical feature matrix."""
    mat = rng.standard_normal((N_PATIENTS, N_CLINICAL)).astype(np.float32)
    return mat


@pytest.fixture
def X(embeddings, clinical_matrix):
    """(N_PATIENTS, 267) full feature matrix."""
    return np.concatenate([embeddings, clinical_matrix], axis=1)


@pytest.fixture
def durations(rng):
    """Survival durations in months (positive)."""
    return rng.uniform(6, 120, size=N_PATIENTS).astype(np.float32)


@pytest.fixture
def events(rng):
    """Binary event indicators (70% event rate)."""
    return rng.binomial(1, 0.7, size=N_PATIENTS).astype(np.int32)


@pytest.fixture
def structured_y(durations, events):
    """Structured numpy array for scikit-survival models."""
    from omics_agent.data.tcga_loader import make_structured_array
    return make_structured_array(durations.astype(float), events.astype(bool))


@pytest.fixture
def eval_times():
    """20 evenly spaced evaluation timepoints (12–120 months)."""
    return np.linspace(12, 120, 20).astype(np.float32)


@pytest.fixture
def patient_metadata(durations, events):
    return [
        {
            "patient_id":      f"TCGA-{i:03d}",
            "survival_months": float(durations[i]),
            "event":           int(events[i]),
            "stage":           (i % 4) + 1,
            "subtype":         ["LumA", "LumB", "HER2", "TNBC"][i % 4],
            "clinical_text":   f"age 55 stage {(i%4)+1} subtype {'LumA' if i%2==0 else 'TNBC'}",
        }
        for i in range(N_PATIENTS)
    ]


@pytest.fixture
def feature_names():
    embed_names = [f"embed_{i}" for i in range(N_EMBED)]
    clin_names  = [
        "age_at_diagnosis", "stage_i", "stage_ii", "stage_iii", "stage_iv",
        "grade_1", "grade_2", "grade_3",
        "treatment_chemo", "treatment_hormone", "treatment_surgery",
    ]
    return embed_names + clin_names
