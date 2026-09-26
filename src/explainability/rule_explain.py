# src/explainability/rule_explain.py
"""Helpers to format deterministic rule explanations.

The rule‑based detectors return a boolean Series. For each flag we produce a
human‑readable explanation string and, when appropriate, a limitation note.
"""

from typing import Dict, List
import pandas as pd


def format_rule_explanations(df: pd.DataFrame) -> List[Dict]:
    """Return a list of rule‑explanation dictionaries.

    Each dictionary has the shape:
    {
        "flag": "duplicate_detection",
        "explanation": "Transaction appears multiple times with the same ID.",
        "limitation": null
    }
    The function inspects the boolean columns produced by the rule module and
    builds messages for rows where the flag is true.
    """
    explanations = []
    # Duplicate detection
    if "duplicate_detection" in df.columns:
        dup_rows = df[df["duplicate_detection"]]
        for idx in dup_rows.index:
            explanations.append({
                "transaction_id": df.at[idx, "transaction_id"],
                "flag": "duplicate_detection",
                "explanation": "Transaction ID is duplicated in the dataset.",
                "limitation": None,
            })
    # Split‑invoice detection
    if "split_invoice_flag" in df.columns:
        split_rows = df[df["split_invoice_flag"]]
        for idx in split_rows.index:
            explanations.append({
                "transaction_id": df.at[idx, "transaction_id"],
                "flag": "split_invoice_detection",
                "explanation": "Multiple invoices for the same vendor/date exceed the approval threshold.",
                "limitation": None,
            })
    return explanations
