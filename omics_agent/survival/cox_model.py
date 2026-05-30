"""CoxPH survival model for OmicsAgent. Uses PCA-50 on Geneformer embeddings."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from lifelines import CoxPHFitter
from typing import List

from omics_agent.data.tcga_loader import CLINICAL_FEATURE_NAMES


class CoxModel:
    """
    CoxPH with PCA dimensionality reduction on the Geneformer embedding.

    Geneformer [CLS] = 256 dims → PCA-50 → concat with 11 clinical dims → CoxPH.
    """

    def __init__(self, n_pca: int = 50, penalizer: float = 0.1):
        self.n_pca = n_pca
        self.penalizer = penalizer
        self.pca = PCA(n_components=n_pca)
        self.cox = CoxPHFitter(penalizer=penalizer)
        self._feature_names: List[str] = []

    def fit(
        self,
        X_embed: np.ndarray,    # (N, 256)
        X_clin:  np.ndarray,    # (N, 11)
        durations: np.ndarray,
        events:    np.ndarray,
    ) -> "CoxModel":
        X_pca = self.pca.fit_transform(X_embed)
        X_all = np.hstack([X_pca, X_clin])
        pca_names = [f"PC{i+1}" for i in range(self.n_pca)]
        self._feature_names = pca_names + list(CLINICAL_FEATURE_NAMES)

        df = pd.DataFrame(X_all, columns=self._feature_names)
        df["duration"] = durations.astype(float)
        df["event"]    = events.astype(int)
        self.cox.fit(df, duration_col="duration", event_col="event")
        return self

    def predict_partial_hazard(self, X_embed: np.ndarray, X_clin: np.ndarray) -> np.ndarray:
        X_pca = self.pca.transform(X_embed)
        X_all = np.hstack([X_pca, X_clin])
        df = pd.DataFrame(X_all, columns=self._feature_names)
        return self.cox.predict_partial_hazard(df).values.astype(np.float32)

    def get_coefficients(self) -> np.ndarray:
        """Return coefficient vector aligned with self._feature_names."""
        return self.cox.params_.values.astype(np.float32)

    @property
    def feature_names(self) -> List[str]:
        return self._feature_names
