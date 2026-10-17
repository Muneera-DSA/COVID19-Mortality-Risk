"""Turn a cleaned cohort table into model inputs.

Both data sources (the capstone extract and the official line list) are first converted to the
same English cohort columns:

  age, female, pregnant, <10 comorbidities>, comorbidity_missing, pneumonia, pneumonia_missing,
  days_symptoms_to_visit (official data only), hospitalised, icu, intubated (post-triage),
  died (outcome), plus context columns (state, sector, admission date) used for validation only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def available(df: pd.DataFrame, features: list[str]) -> list[str]:
    return [f for f in features if f in df.columns]


def triage_features(df: pd.DataFrame) -> list[str]:
    return available(df, config.TRIAGE_FEATURES)


def matrix(df: pd.DataFrame, features: list[str]) -> np.ndarray:
    missing = [f for f in features if f not in df.columns]
    if missing:
        raise KeyError(f"features not in cohort: {missing}")
    return df[features].to_numpy(dtype=np.float32)


def check_cohort(df: pd.DataFrame) -> None:
    """Sanity rules every cohort must satisfy."""
    assert df["age"].between(0, 120).all(), "age out of range"
    assert set(df["died"].unique()) <= {0, 1}, "outcome must be 0/1"
    binary = [c for c in [*config.COMORBIDITY_EN.values(), "female", "pregnant", "pneumonia"] if c in df]
    for c in binary:
        assert set(df[c].unique()) <= {0, 1}, f"{c} must be 0/1"
    if "pregnant" in df and "female" in df:
        assert not ((df.pregnant == 1) & (df.female == 0)).any(), "pregnant male"
