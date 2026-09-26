# src/features/engineer.py
"""Feature engineering utilities.

The function ``build_feature_matrix`` takes a canonical DataFrame (already
mapped to the fields described in the spec) and returns a new DataFrame that
contains both the original columns and engineered features required by the
MVP.

Implemented MVP features:

* **round_number_flag** – True if the amount ends with a round number (e.g.
  divisible by 100).
* **weekend_flag** – True if the transaction date falls on Saturday or Sunday.
* **holiday_flag** – True if the date is a public holiday in the supplied
  ``jurisdiction`` (using the ``holidays`` library). If no jurisdiction is
  given the flag is always False.
* **senior_posting_flag** – True if the ``approver_role`` column exists and
  contains a value that matches one of the configured senior roles.
* **vendor_zscore** – Z‑score of the amount within the vendor's historic
  amounts (computed on the provided DataFrame).
* **days_since_vendor_created** – Difference in days between the transaction
  date and the ``vendor_created_date`` column.
* **split_invoice_flag** – Delegates to ``detection.rules.split_invoice_detection``
  and merges the result as a feature.
"""

import pandas as pd
import numpy as np
from datetime import datetime

try:
    import holidays
except ImportError:  # pragma: no cover
    holidays = None

from ..detection.rules import split_invoice_detection


def _is_round_number(amount: float) -> bool:
    # Consider "round" if amount is divisible by 100 (or 10 for small amounts)
    if amount == 0:
        return False
    base = 100 if amount >= 1000 else 10
    return amount % base == 0


def build_feature_matrix(
    df: pd.DataFrame,
    jurisdiction: str | None = None,
    senior_roles: list[str] | None = None,
    approval_thresholds: dict | None = None,
) -> pd.DataFrame:
    """Return a DataFrame enriched with engineered features.

    Parameters
    ----------
    df : pd.DataFrame
        Canonical data containing the fields defined in the spec.
    jurisdiction : str | None
        Country/region code accepted by the ``holidays`` package (e.g. "US",
        "GB"). If ``None`` holiday detection is disabled.
    senior_roles : list[str] | None
        List of role names that are considered senior/privileged. If ``None``
        the senior flag is always False.
    approval_thresholds : dict | None
        Mapping ``vendor -> threshold`` used by the split‑invoice rule.
    """
    # Work on a copy so we do not mutate the caller's DataFrame
    df = df.copy()

    # ------------------------------------------------------------------
    # 1. Round‑number flag
    # ------------------------------------------------------------------
    df["round_number_flag"] = df["amount"].apply(_is_round_number)

    # ------------------------------------------------------------------
    # 2. Weekend flag – pandas.Timestamp.weekday(): 0=Mon, 6=Sun
    # ------------------------------------------------------------------
    df["transaction_date"] = pd.to_datetime(df["date"], errors="coerce")
    df["weekend_flag"] = df["transaction_date"].dt.weekday.isin([5, 6])

    # ------------------------------------------------------------------
    # 3. Holiday flag (optional)
    # ------------------------------------------------------------------
    if jurisdiction and holidays:
        holiday_cal = holidays.country_holidays(jurisdiction)
        df["holiday_flag"] = df["transaction_date"].dt.date.apply(
            lambda d: d in holiday_cal
        )
    else:
        df["holiday_flag"] = False

    # ------------------------------------------------------------------
    # 4. Senior posting flag (optional based on role column)
    # ------------------------------------------------------------------
    if senior_roles and "approver_role" in df.columns:
        df["senior_posting_flag"] = df["approver_role"].isin(senior_roles)
    else:
        df["senior_posting_flag"] = False

    # ------------------------------------------------------------------
    # 5. Vendor‑relative z‑score for amount
    # ------------------------------------------------------------------
    if "vendor" in df.columns:
        vendor_stats = df.groupby("vendor")["amount"].agg(["mean", "std"]).rename(columns={"mean": "v_mean", "std": "v_std"})
        df = df.merge(vendor_stats, left_on="vendor", right_index=True, how="left")
        # Avoid division by zero
        df["vendor_std_adj"] = df["v_std"].replace({0: np.nan})
        df["vendor_zscore"] = (df["amount"] - df["v_mean"]) / df["vendor_std_adj"]
        df["vendor_zscore"] = df["vendor_zscore"].fillna(0)
        df.drop(columns=["v_mean", "v_std", "vendor_std_adj"], inplace=True)
    else:
        df["vendor_zscore"] = 0.0

    # ------------------------------------------------------------------
    # 6. Days since vendor was created (requires vendor_created_date column)
    # ------------------------------------------------------------------
    if "vendor_created_date" in df.columns:
        df["vendor_created_date_dt"] = pd.to_datetime(df["vendor_created_date"], errors="coerce")
        df["days_since_vendor_created"] = (df["transaction_date"] - df["vendor_created_date_dt"]).dt.days
        df["days_since_vendor_created"] = df["days_since_vendor_created"].fillna(-1).astype(int)
        df.drop(columns=["vendor_created_date_dt"], inplace=True)
    else:
        df["days_since_vendor_created"] = -1

    # ------------------------------------------------------------------
    # 7. Split‑invoice detection feature – leverages the rule implementation
    # ------------------------------------------------------------------
    df["split_invoice_flag"] = split_invoice_detection(
        df,
        vendor_col="vendor",
        date_col="date",
        amount_col="amount",
        approval_thresholds=approval_thresholds,
    )

    # Keep only the engineered feature columns (the downstream models will pick
    # what they need). The original raw columns stay in the DataFrame for later
    # evidence‑lineage merging.
    return df
