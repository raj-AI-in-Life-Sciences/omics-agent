"""
Clinical feature engineering for OmicsAgent.

Preprocessing pipeline:
  - Age: numeric, impute to cohort median
  - AJCC stage: one-hot encode to 4 bins (I, II, III, IV)
  - Grade: one-hot encode (1, 2, 3)
  - Treatment flags: chemo, radio, surgery (binary)

Output: numpy array of shape (N, 11) matching CLINICAL_FEATURE_NAMES order.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional

from .tcga_loader import CLINICAL_FEATURE_NAMES


STAGE_MAP = {
    "stage i": 0, "stage ia": 0, "stage ib": 0,
    "stage ii": 1, "stage iia": 1, "stage iib": 1, "stage iic": 1,
    "stage iii": 2, "stage iiia": 2, "stage iiib": 2, "stage iiic": 2,
    "stage iv": 3, "stage iva": 3, "stage ivb": 3, "stage ivc": 3,
}


class ClinicalFeatureEngineer:
    """
    Fits on training data (for median imputation) and transforms any split.

    Usage:
        eng = ClinicalFeatureEngineer()
        X_train = eng.fit_transform(train_df)
        X_test  = eng.transform(test_df)
    """

    def __init__(self):
        self._age_median: Optional[float] = None

    def fit(self, df: pd.DataFrame) -> "ClinicalFeatureEngineer":
        ages = pd.to_numeric(df.get("age_at_diagnosis", pd.Series(dtype=float)), errors="coerce")
        self._age_median = float(ages.median())
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if self._age_median is None:
            raise RuntimeError("Call fit() before transform()")
        N = len(df)
        out = np.zeros((N, len(CLINICAL_FEATURE_NAMES)), dtype=np.float32)

        # Age (index 0)
        ages = pd.to_numeric(df.get("age_at_diagnosis", pd.Series([np.nan] * N)), errors="coerce")
        ages = ages.fillna(self._age_median).values
        out[:, 0] = (ages - 60.0) / 15.0   # rough z-score (TCGA median ~60)

        # Stage one-hot (indices 1-4)
        stages = df.get("ajcc_pathologic_stage", pd.Series([""] * N))
        for i, s in enumerate(stages):
            bucket = STAGE_MAP.get(str(s).lower().strip(), -1)
            if 0 <= bucket <= 3:
                out[i, 1 + bucket] = 1.0

        # Grade one-hot (indices 5-7)
        grades = df.get("tumor_grade", df.get("neoplasm_histologic_grade", pd.Series([""] * N)))
        for i, g in enumerate(grades):
            g_str = str(g).lower()
            if "g1" in g_str or "grade 1" in g_str or "low" in g_str:
                out[i, 5] = 1.0
            elif "g2" in g_str or "grade 2" in g_str or "intermediate" in g_str:
                out[i, 6] = 1.0
            elif "g3" in g_str or "grade 3" in g_str or "high" in g_str:
                out[i, 7] = 1.0

        # Treatment flags (indices 8-10)
        for flag_col, idx in [
            ("pharmaceutical_therapy_type", 8),
            ("radiation_therapy", 9),
            ("surgery_performed", 10),
        ]:
            col = df.get(flag_col, pd.Series([""] * N))
            out[:, idx] = (col.astype(str).str.lower().str.contains("yes|chemotherapy|true", na=False)).astype(float)

        return out

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        return self.fit(df).transform(df)

    def feature_names(self) -> list[str]:
        return list(CLINICAL_FEATURE_NAMES)
