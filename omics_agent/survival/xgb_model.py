"""XGBoost AFT (Accelerated Failure Time) survival model for OmicsAgent."""

from __future__ import annotations

import numpy as np
import xgboost as xgb
from typing import List, Optional


class XGBSurvivalModel:
    """
    XGBoost survival model using AFT (aft_loss_dist=logistic).
    Produces a predicted survival time (higher = longer survival → lower risk).
    """

    def __init__(
        self,
        n_estimators: int = 300,
        max_depth: int = 5,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
    ):
        self.params = {
            "objective":        "survival:aft",
            "eval_metric":      "aft-nloglik",
            "aft_loss_dist":    "logistic",
            "tree_method":      "hist",
            "max_depth":        max_depth,
            "learning_rate":    learning_rate,
            "subsample":        subsample,
            "colsample_bytree": colsample_bytree,
            "n_estimators":     n_estimators,
            "verbosity":        0,
        }
        self.model: Optional[xgb.XGBModel] = None
        self._feature_names: List[str] = []

    def fit(
        self,
        X: np.ndarray,
        durations: np.ndarray,
        events: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> "XGBSurvivalModel":
        self._feature_names = feature_names or [f"f{i}" for i in range(X.shape[1])]

        # XGBoost AFT requires labels as (lower_bound, upper_bound)
        # For observed events: [t, t]; for censored: [t, +inf]
        y_lower = durations.copy()
        y_upper = np.where(events == 1, durations, np.inf)

        dtrain = xgb.DMatrix(X, feature_names=self._feature_names)
        dtrain.set_float_info("label_lower_bound", y_lower)
        dtrain.set_float_info("label_upper_bound", y_upper)

        n_round = self.params.pop("n_estimators")
        self.model = xgb.train(self.params, dtrain, num_boost_round=n_round, verbose_eval=False)
        self.params["n_estimators"] = n_round
        return self

    def predict_risk(self, X: np.ndarray) -> np.ndarray:
        """Lower predicted survival time → higher risk. Negate to get risk score."""
        dtest = xgb.DMatrix(X, feature_names=self._feature_names)
        pred_time = self.model.predict(dtest)
        # Negate so that larger value = higher risk (consistent with CoxPH / DeepSurv)
        return -pred_time.astype(np.float32)

    @property
    def feature_names(self) -> List[str]:
        return self._feature_names
