# src/explainability/reliability.py
"""Reliability‑indicator calculation utilities.

The MVP defines a *human‑readable* reliability label (high / medium / low) that
reflects how much confidence the auditor can place in the combined anomaly
score.  The logic mirrors the helper used in ``pipeline.py`` but is extracted
here so that the workbench or other downstream components can import it
directly.

Factors considered:

* **Record count** – larger samples give more stable statistics.
* **Feature completeness** – the proportion of engineered features that are
  present (non‑missing).
* **Benford applicability** – whether Benford analysis could be performed;
  if it was skipped this reduces reliability.
"""

from typing import List
import pandas as pd


def compute_reliability(
    df: pd.DataFrame,
    benford_applicable: bool,
    feature_flag_suffixes: List[str] = ["_flag", "_zscore"],
) -> str:
    """Return ``"high"``, ``"medium"`` or ``"low"`` reliability.

    Parameters
    ----------
    df: pd.DataFrame
        The DataFrame *after* feature engineering (contains engineered columns).
    benford_applicable: bool
        Result of the Benford suitability check.
    feature_flag_suffixes: list[str]
        Suffixes used to identify engineered columns.
    """
    n_rows = len(df)
    # Identify engineered columns by suffix
    engineered_cols = [c for c in df.columns if any(c.endswith(s) for s in feature_flag_suffixes)]
    missing_ratio = df[engineered_cols].isna().mean().mean() if engineered_cols else 0.0

    if n_rows > 1000 and missing_ratio < 0.05 and benford_applicable:
        return "high"
    if n_rows > 200 and (missing_ratio < 0.10 or benford_applicable):
        return "medium"
    return "low"
