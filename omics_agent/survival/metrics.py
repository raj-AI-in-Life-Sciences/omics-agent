"""
Survival analysis metrics for OmicsAgent.

  concordance_index  — Harrell's C-statistic (O(n²), use on test sets only)
  time_dependent_auc — IPCW AUC at specified timepoints (sksurv)
  integrated_brier_score — IBS (sksurv)
"""

from __future__ import annotations

import numpy as np
from typing import List


def concordance_index(
    durations: np.ndarray,
    risk_scores: np.ndarray,
    events: np.ndarray,
) -> float:
    """
    Harrell's concordance index.
    Higher risk_score should correspond to shorter survival (event sooner).
    """
    concordant = discordant = 0
    n = len(durations)
    for i in range(n):
        for j in range(i + 1, n):
            if durations[i] == durations[j]:
                continue
            if events[i] == 0 and events[j] == 0:
                continue
            # Determine which is the earlier event
            if events[i] == 1 and durations[i] < durations[j]:
                # i died first; concordant if risk_i > risk_j
                if risk_scores[i] > risk_scores[j]:
                    concordant += 1
                elif risk_scores[i] < risk_scores[j]:
                    discordant += 1
            elif events[j] == 1 and durations[j] < durations[i]:
                if risk_scores[j] > risk_scores[i]:
                    concordant += 1
                elif risk_scores[j] < risk_scores[i]:
                    discordant += 1
    total = concordant + discordant
    return concordant / total if total > 0 else 0.5


def time_dependent_auc(
    train_y: np.ndarray,    # structured array (event, time) — for IPCW
    test_y:  np.ndarray,
    risk_scores: np.ndarray,
    eval_times: np.ndarray,
) -> np.ndarray:
    """Compute IPCW cumulative/dynamic AUC at each evaluation timepoint."""
    try:
        from sksurv.metrics import cumulative_dynamic_auc
        auc_vals, _ = cumulative_dynamic_auc(train_y, test_y, risk_scores, eval_times)
        return auc_vals.astype(np.float32)
    except Exception:
        return np.full(len(eval_times), 0.5, dtype=np.float32)


def integrated_brier_score(
    train_y:       np.ndarray,   # structured array
    test_y:        np.ndarray,
    surv_matrix:   np.ndarray,   # (N_test, T) survival probabilities
    eval_times:    np.ndarray,
) -> float:
    """Integrated Brier Score (lower = better calibration)."""
    try:
        from sksurv.metrics import integrated_brier_score as _ibs
        return float(_ibs(train_y, test_y, surv_matrix, eval_times))
    except Exception:
        # Fallback: plain Brier score at median timepoint
        T_mid = len(eval_times) // 2
        surv_mid = surv_matrix[:, T_mid]
        ev = test_y["event"].astype(float)
        return float(np.mean((ev - (1 - surv_mid)) ** 2))


def all_metrics(
    train_y: np.ndarray,
    test_y:  np.ndarray,
    risk_scores: np.ndarray,
    surv_matrix: np.ndarray,
    eval_times: np.ndarray,
) -> dict:
    """Compute all metrics and return as dict."""
    cindex = concordance_index(test_y["time"], risk_scores, test_y["event"].astype(int))
    td_auc = time_dependent_auc(train_y, test_y, risk_scores, eval_times)
    ibs    = integrated_brier_score(train_y, test_y, surv_matrix, eval_times)
    return {
        "c_index": round(cindex, 4),
        "ibs":     round(ibs, 4),
        "td_auc":  {int(t): round(float(a), 4) for t, a in zip(eval_times, td_auc)},
    }
