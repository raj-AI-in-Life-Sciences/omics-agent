"""
TCGA data loader for OmicsAgent.

Supports TCGA-BRCA and TCGA-LGG cohorts.
Merges RNA-seq + clinical CSV, filters for patients with valid survival endpoints,
and creates stratified train / calibration / test splits.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional
from sklearn.model_selection import train_test_split


# Columns required in the merged dataframe
REQUIRED_COLUMNS = {
    "case_id",
    "vital_status",
    "days_to_event",        # days_to_death or days_to_last_follow_up
    "event",                # 1 = death, 0 = censored
    "age_at_diagnosis",
}

CLINICAL_FEATURE_NAMES = [
    "age_at_diagnosis",
    "stage_i", "stage_ii", "stage_iii", "stage_iv",
    "grade_1", "grade_2", "grade_3",
    "treatment_chemo", "treatment_radio", "treatment_surgery",
]


def load_tcga(
    rna_path: str | Path,
    clinical_path: str | Path,
    min_follow_up_days: int = 30,
) -> pd.DataFrame:
    """
    Load and merge TCGA RNA-seq + clinical data.

    Expected RNA-seq CSV: rows = patients (case_id as index), columns = gene ENSEMBL IDs, values = FPKM-UQ.
    Expected clinical CSV: rows = patients, columns include vital_status, days_to_death,
                           days_to_last_follow_up, age_at_diagnosis, ajcc_pathologic_stage, etc.

    Returns merged DataFrame with a 'event' column (1 = death) and 'survival_months' column.
    """
    rna = pd.read_csv(rna_path, index_col=0)
    cli = pd.read_csv(clinical_path, index_col=0)

    # Unify index name
    rna.index.name = "case_id"
    cli.index.name = "case_id"

    df = cli.join(rna, how="inner")

    # Compute survival endpoint
    df = _compute_survival_endpoint(df)

    # Filter short follow-up
    df = df[df["days_to_event"] >= min_follow_up_days]
    df = df.dropna(subset=["days_to_event", "event"])
    df["survival_months"] = df["days_to_event"] / 30.44

    return df.reset_index()


def _compute_survival_endpoint(df: pd.DataFrame) -> pd.DataFrame:
    """Derive 'event' (0/1) and 'days_to_event' from TCGA clinical columns."""
    # TCGA vital_status: "Dead" | "Alive"
    df["event"] = (df.get("vital_status", "Alive") == "Dead").astype(int)

    days_death    = pd.to_numeric(df.get("days_to_death",             pd.Series(dtype=float)), errors="coerce")
    days_follow   = pd.to_numeric(df.get("days_to_last_follow_up",   pd.Series(dtype=float)), errors="coerce")

    df["days_to_event"] = np.where(df["event"] == 1, days_death, days_follow)
    return df


def get_gene_columns(df: pd.DataFrame) -> list[str]:
    """Return all Ensembl gene ID columns (ENSG...)."""
    return [c for c in df.columns if c.startswith("ENSG")]


def stratified_split(
    df: pd.DataFrame,
    train_frac: float = 0.80,
    cal_frac: float   = 0.10,
    test_frac: float  = 0.10,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    80 / 10 / 10 stratified split on the event column.
    The calibration set is used ONLY for conformal prediction quantile fitting.
    """
    assert abs(train_frac + cal_frac + test_frac - 1.0) < 1e-6

    # First split off test
    train_cal, test = train_test_split(
        df,
        test_size=test_frac,
        stratify=df["event"],
        random_state=seed,
    )
    # Then split train vs calibration
    cal_relative = cal_frac / (train_frac + cal_frac)
    train, cal = train_test_split(
        train_cal,
        test_size=cal_relative,
        stratify=train_cal["event"],
        random_state=seed,
    )
    return train.reset_index(drop=True), cal.reset_index(drop=True), test.reset_index(drop=True)


def make_structured_array(
    durations: np.ndarray,
    events: np.ndarray,
) -> np.ndarray:
    """Create a structured array (event: bool, time: float) for scikit-survival."""
    dtype = np.dtype([("event", bool), ("time", float)])
    y = np.zeros(len(durations), dtype=dtype)
    y["event"] = events.astype(bool)
    y["time"]  = durations.astype(float)
    return y
