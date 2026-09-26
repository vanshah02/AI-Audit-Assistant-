# src/features/suitability.py
"""Utility functions to decide whether a particular analysis is applicable.

For the MVP we currently only expose a Benford‑suitability check, but the
module can be extended with other suitability predicates (e.g., enough data
for IsolationForest).  The function returns a tuple ``(applicable, reason)``
where ``reason`` is a human‑readable explanation that can be added to the
``limitations`` list when the analysis is skipped.
"""

import pandas as pd


def benford_suitability(df: pd.DataFrame, amount_col: str = "amount") -> tuple[bool, str]:
    """Determine if the dataset is appropriate for Benford analysis.

    Criteria (as defined in the spec):
    * At least 100 non‑zero amount values.
    * The numeric range spans at least three orders of magnitude (max/min >=
      1 000).
    * The column does **not** look like an identifier; a crude heuristic is to
      reject the analysis when **all** values are small integers (< 1 000 000).

    Returns
    -------
    (bool, str)
        ``True`` if the analysis can be run, ``False`` otherwise.  The string is
        a limitation message that can be appended to the audit record.
    """
    amounts = df[amount_col]
    non_zero = amounts[amounts != 0]
    if len(non_zero) < 100:
        return False, "Benford analysis unavailable — insufficient data volume"
    if non_zero.max() / max(non_zero.min(), 1) < 1_000:
        return False, "Benford analysis unavailable — amount range too narrow"
    # Reject if the column appears to be an ID rather than a monetary amount
    if (non_zero < 1_000_000).all():
        return False, "Benford analysis unavailable — values resemble identifiers"
    return True, ""
