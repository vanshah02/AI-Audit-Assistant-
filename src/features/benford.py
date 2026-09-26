# src/features/benford.py
"""Benford analysis utilities.

The function ``benford_score`` expects a DataFrame that has already passed the
suitability check (i.e. enough volume and a suitable numeric range).  It returns
a score in ``[0, 1]`` where higher values indicate a larger deviation from
Benford's expected leading‑digit distribution.
"""

import numpy as np
import pandas as pd
from scipy.stats import chisquare


def benford_score(df: pd.DataFrame, amount_col: str = "amount") -> float:
    """Calculate a normalized chi‑square deviation from Benford's law.

    The implementation:
    1. Extract the leading digit of each non‑zero amount.
    2. Count occurrences of digits 1‑9.
    3. Compute a chi‑square statistic against the theoretical Benford
       probabilities.
    4. Map the raw statistic to ``[0, 1]`` with a logistic function so that the
       output can be combined with other anomaly scores.
    """
    leading = (
        df[amount_col]
        .astype(str)
        .str.lstrip("0")
        .str[0]
        .astype(int)
    )
    observed = leading.value_counts().sort_index()
    # Ensure the series has an entry for every digit 1‑9
    observed = observed.reindex(range(1, 10), fill_value=0)
    total = observed.sum()
    if total == 0:
        return 0.0
    expected = np.array([np.log10(1 + 1 / d) for d in range(1, 10)]) * total
    chi2, _ = chisquare(observed, f_exp=expected)
    # Logistic mapping – parameters chosen to give a smooth 0‑1 curve
    score = 1 / (1 + np.exp(-0.01 * (chi2 - 20)))
    return float(score)
