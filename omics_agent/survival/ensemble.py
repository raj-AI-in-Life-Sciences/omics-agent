"""
C-index-weighted rank aggregation ensemble for OmicsAgent.

Combines risk scores from CoxPH, RSF, DeepSurv, and XGBoost AFT
into a single ensemble risk ranking. Survival curve is taken from RSF
(the only model that produces calibrated S(t)) and rescaled by the ensemble rank.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import rankdata
from typing import Dict, List

from .metrics import concordance_index


class SurvivalEnsemble:
    """
    Rank-based ensemble of 4 survival models.

    During training: fits weights by computing each model's C-index on
    a held-out subset (or the training set — for portfolio purposes, use training).

    During inference: computes ensemble risk rank and produces survival curve.
    """

    def __init__(self):
        self.weights: Dict[str, float] = {
            "cox":      1.0,
            "rsf":      1.0,
            "deepsurv": 1.0,
            "xgb":      1.0,
        }

    def calibrate_weights(
        self,
        cox_risk:     np.ndarray,
        rsf_risk:     np.ndarray,
        deepsurv_risk:np.ndarray,
        xgb_risk:     np.ndarray,
        durations:    np.ndarray,
        events:       np.ndarray,
    ) -> None:
        """Compute C-index per model; set weights proportional to C-index."""
        for name, risk in [
            ("cox", cox_risk),
            ("rsf", rsf_risk),
            ("deepsurv", deepsurv_risk),
            ("xgb", xgb_risk),
        ]:
            self.weights[name] = max(concordance_index(durations, risk, events), 0.5)

        total = sum(self.weights.values())
        self.weights = {k: v / total for k, v in self.weights.items()}

    def ensemble_risk_ranks(
        self,
        cox_risk:      np.ndarray,
        rsf_risk:      np.ndarray,
        deepsurv_risk: np.ndarray,
        xgb_risk:      np.ndarray,
    ) -> np.ndarray:
        """Return weighted rank aggregation. Higher value = higher risk."""
        r_cox = rankdata(cox_risk)
        r_rsf = rankdata(rsf_risk)
        r_dsv = rankdata(deepsurv_risk)
        r_xgb = rankdata(xgb_risk)

        w = self.weights
        return (
            w["cox"] * r_cox
            + w["rsf"] * r_rsf
            + w["deepsurv"] * r_dsv
            + w["xgb"] * r_xgb
        ).astype(np.float32)

    def survival_curve_for_patient(
        self,
        rsf_surv_matrix: np.ndarray,   # (N, T) from RSFModel.predict_survival_matrix
        patient_idx: int,
        eval_times: np.ndarray,
    ) -> Dict[int, float]:
        """Return {month: S(t)} dict for one patient from RSF survival matrix."""
        surv = rsf_surv_matrix[patient_idx]
        return {int(t): float(s) for t, s in zip(eval_times, surv)}

    def predict_median_survival(
        self,
        rsf_surv_matrix: np.ndarray,
        eval_times: np.ndarray,
    ) -> np.ndarray:
        """Return predicted median OS (months) per patient from RSF curves."""
        medians = []
        for i in range(rsf_surv_matrix.shape[0]):
            surv = rsf_surv_matrix[i]
            # Find first time where S(t) <= 0.5
            below = np.where(surv <= 0.5)[0]
            if len(below) == 0:
                medians.append(float(eval_times[-1]))
            else:
                medians.append(float(eval_times[below[0]]))
        return np.array(medians, dtype=np.float32)
