# src/detection/rules.py
"""Deterministic rule checks used in the audit pipeline.

The two rules implemented for the MVP are:

* **duplicate_detection** – flags rows that have the same
  `transaction_id` (or a combination of key fields) appearing more than once.
* **split_invoice_detection** – flags a set of rows that share the same
  `vendor` and `date` where the sum of `amount` exceeds the
  `approval_threshold` for that vendor (the threshold is supplied by the caller).

Both functions return a ``pandas.Series`` of booleans indexed like the input
DataFrame, so they can be merged directly into the feature matrix.
"""

import pandas as pd


def duplicate_detection(df: pd.DataFrame, key_columns=None) -> pd.Series:
    """Return a boolean series indicating duplicate rows.

    Parameters
    ----------
    df : pd.DataFrame
        Canonical data frame containing at least ``transaction_id``.
    key_columns : list[str] | None
        Columns used to identify a duplicate. If ``None`` the function falls
        back to ``["transaction_id"]``.
    """
    if key_columns is None:
        key_columns = ["transaction_id"]
    # ``keep=False`` marks *all* occurrences of a duplicate as True
    dup_mask = df.duplicated(subset=key_columns, keep=False)
    return dup_mask


def split_invoice_detection(
    df: pd.DataFrame,
    vendor_col: str = "vendor",
    date_col: str = "date",
    amount_col: str = "amount",
    approval_thresholds: dict | None = None,
) -> pd.Series:
    """Detect split‑invoice patterns.

    A split invoice is when multiple rows for the same vendor on the same date
    collectively exceed a predefined approval threshold. The threshold can be
    supplied per‑vendor; if omitted a default of ``10_000`` is used.
    """
    if approval_thresholds is None:
        approval_thresholds = {}

    # Group by vendor + date and sum the amounts
    grouped = df.groupby([vendor_col, date_col])[amount_col].sum().reset_index()
    # Determine threshold per vendor (fallback to 10_000)
    grouped["threshold"] = grouped[vendor_col].map(
        lambda v: approval_thresholds.get(v, 10_000)
    )
    # Flag groups where total > threshold
    grouped["split_flag"] = grouped[amount_col] > grouped["threshold"]
    # Merge the flag back to the original rows
    split_flag_series = pd.merge(
        df[[vendor_col, date_col]],
        grouped[[vendor_col, date_col, "split_flag"]],
        on=[vendor_col, date_col],
        how="left",
    )["split_flag"].fillna(False)
    return split_flag_series
