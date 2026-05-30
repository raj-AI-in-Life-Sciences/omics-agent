"""Random Survival Forest for OmicsAgent (scikit-survival)."""

from __future__ import annotations

import numpy as np
from sksurv.ensemble import RandomSurvivalForest
from typing import List


class RSFModel:
    """
    Random Survival Forest trained on full 266-dim feature vector.

    Predicts both cumulative hazard (for C-index) and survival function (for IBS/conformal).
    """

    def __init__(self, n_estimators: int = 300, min_samples_leaf: int = 10, n_jobs: int = -1):
        self.rsf = RandomSurvivalForest(
            n_estimators=n_estimators,
            min_samples_leaf=min_samples_leaf,
            max_features="sqrt",
            n_jobs=n_jobs,
            random_state=42,
        )
        self.eval_times_: np.ndarray = np.array([])
        self._feature_names: List[str] = []

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,                      # structured array (event: bool, time: float)
        feature_names: List[str] | None = None,
    ) -> "RSFModel":
        self.rsf.fit(X, y)
        # Choose 20 evaluation time points from observed event times
        event_times = y["time"][y["event"]]
        self.eval_times_ = np.percentile(event_times, np.linspace(5, 95, 20))
        self._feature_names = feature_names or [f"f{i}" for i in range(X.shape[1])]
        return self

    def predict_risk(self, X: np.ndarray) -> np.ndarray:
        """Higher = higher risk (cumulative hazard at last eval time)."""
        return self.rsf.predict(X).astype(np.float32)

    def predict_survival_matrix(self, X: np.ndarray) -> np.ndarray:
        """Return (N, T) survival probabilities at self.eval_times_."""
        surv_fns = self.rsf.predict_survival_function(X)
        return np.row_stack([fn(self.eval_times_) for fn in surv_fns]).astype(np.float32)

    @property
    def feature_names(self) -> List[str]:
        return self._feature_names
