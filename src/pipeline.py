# src/pipeline.py
"""Top‑level audit pipeline.

The script follows the MVP architecture:

1. Load a canonical CSV/Excel file (the ingestion & schema‑mapping step is
   assumed to have been performed upstream – the file must contain the column
   names defined in the spec).
2. Engineer feature matrix.
3. Fit IsolationForest and Autoencoder detectors on the *available* features.
4. Compute normalized anomaly scores and combine them.
5. Run deterministic rule checks.
6. Perform Benford suitability check (if the data meets volume/variance
   requirements) and, when possible, compute a Benford deviation score.
7. Assemble a reliability indicator based on:
   * record count per vendor (data volume)
   * percentage of missing engineered features
   * whether Benford analysis was applicable
8. Generate SHAP explanations for the ML detectors.
9. Build per‑transaction audit records conforming to the JSON schema
   described in the specification.
10. Write the records to ``audit_results.jsonl`` (one JSON object per line).

The module can be invoked as a script::

    python -m src.pipeline <path-to-canonical-data.csv>

Only standard libraries, pandas, scikit‑learn, shap and holidays are required.
"""

import sys
import json
import pandas as pd
from pathlib import Path
from datetime import datetime

from .features.engineer import build_feature_matrix
from .detection import IsolationForestDetector, AutoencoderDetector, combine_scores
from .detection.rules import duplicate_detection, split_invoice_detection
from .explainability.shap_explain import compute_shap_values
from .explainability.rule_explain import format_rule_explanations
from .evidence import build_audit_record

# ---------------------------------------------------------------------------
# Helper: Benford analysis
# ---------------------------------------------------------------------------
def _benford_suitability(df: pd.DataFrame, amount_col: str = "amount") -> bool:
    """Return ``True`` if the data is suitable for Benford analysis.

    Criteria (MVP):
    * At least 100 non‑zero amount records.
    * Amounts span at least three orders of magnitude (max/min >= 1000).
    * The column does not look like an identifier (all values are integers < 1e6).
    """
    amounts = df[amount_col]
    non_zero = amounts[amounts != 0]
    if len(non_zero) < 100:
        return False
    if non_zero.max() / max(non_zero.min(), 1) < 1_000:
        return False
    # Rough heuristic to avoid invoice numbers: if the majority are small ints
    if (non_zero < 1_000_000).all():
        # treat as possible IDs, skip Benford
        return False
    return True


def _benford_score(df: pd.DataFrame, amount_col: str = "amount") -> float:
    """Compute a simple distance from Benford's expected distribution.

    Returns a score in [0, 1] where higher values indicate stronger deviation.
    The implementation uses the chi‑square statistic normalised by the number of
    digits.
    """
    import numpy as np
    from scipy.stats import chisquare

    leading = df[amount_col].astype(str).str.lstrip("0").str[0].astype(int)
    observed = leading.value_counts().sort_index()
    # Ensure all digits 1‑9 are present
    observed = observed.reindex(range(1, 10), fill_value=0)
    total = observed.sum()
    if total == 0:
        return 0.0
    # Expected Benford probabilities
    expected = [np.log10(1 + 1 / d) for d in range(1, 10)]
    expected_counts = np.array(expected) * total
    chi2, _ = chisquare(observed, f_exp=expected_counts)
    # Normalise: chi2 can be large; we map to 0‑1 via a simple logistic
    score = 1 / (1 + np.exp(-0.01 * (chi2 - 20)))  # tuned rough sigmoidal
    return float(score)

# ---------------------------------------------------------------------------
# Reliability indicator helper
# ---------------------------------------------------------------------------
def _compute_reliability(df: pd.DataFrame, benford_applicable: bool) -> str:
    """Return a qualitative reliability label.

    Simple heuristic:
    * ``high`` – >1000 rows, <5% missing engineered features, Benford applicable.
    * ``medium`` – >200 rows and either missing features <10% or Benford applicable.
    * otherwise ``low``.
    """
    n_rows = len(df)
    # engineered features have known column suffixes; we count missing values
    feature_cols = [c for c in df.columns if c.endswith("_flag") or c.endswith("_zscore") or c.startswith("days_since_")]
    missing_ratio = df[feature_cols].isna().mean().mean() if feature_cols else 0.0
    if n_rows > 1000 and missing_ratio < 0.05 and benford_applicable:
        return "high"
    if n_rows > 200 and (missing_ratio < 0.10 or benford_applicable):
        return "medium"
    return "low"

# ---------------------------------------------------------------------------
# Main pipeline function
# ---------------------------------------------------------------------------
def run_pipeline(data_path: str, jurisdiction: str | None = None, senior_roles: list[str] | None = None, approval_thresholds: dict | None = None, output_path: str = "audit_results.jsonl"):
    """Execute the end‑to‑end audit pipeline.

    Parameters
    ----------
    data_path: str
        Path to the canonical CSV/Excel file.
    jurisdiction: str | None
        Country/region code for holiday detection (passed to feature engine).
    senior_roles: list[str] | None
        Roles considered senior for the senior‑posting flag.
    approval_thresholds: dict | None
        Vendor → amount threshold for split‑invoice detection.
    output_path: str
        Destination file for line‑delimited JSON audit records.
    """
    raw_df = pd.read_csv(data_path)
    # Ensure required canonical columns exist; missing ones are added as NaN
    required_cols = [
        "transaction_id",
        "date",
        "amount",
        "vendor",
        "approver",
        "account",
        "description",
        "vendor_created_date",
        "document_reference",
    ]
    for col in required_cols:
        if col not in raw_df.columns:
            raw_df[col] = pd.NA

    # ------------------------------------------------------------------
    # Feature engineering
    # ------------------------------------------------------------------
    feature_df = build_feature_matrix(
        raw_df,
        jurisdiction=jurisdiction,
        senior_roles=senior_roles,
        approval_thresholds=approval_thresholds,
    )

    # Select only numeric features for the detectors (ignore booleans)
    numeric_features = ["amount", "vendor_zscore", "days_since_vendor_created"]
    X = feature_df[numeric_features].fillna(0)

    # ------------------------------------------------------------------
    # Fit detectors
    # ------------------------------------------------------------------
    iso_det = IsolationForestDetector().fit(X)
    ae_det = AutoencoderDetector().fit(X)
    iso_score = iso_det.score(X)
    ae_score = ae_det.score(X)
    combined_score = combine_scores(iso_score, ae_score)

    # ------------------------------------------------------------------
    # Rule checks
    # ------------------------------------------------------------------
    dup_flag = duplicate_detection(raw_df)
    split_flag = split_invoice_detection(
        raw_df,
        vendor_col="vendor",
        date_col="date",
        amount_col="amount",
        approval_thresholds=approval_thresholds,
    )
    # Append rule flags to the dataframe for later merging
    raw_df["duplicate_detection"] = dup_flag
    raw_df["split_invoice_flag"] = split_flag

    # ------------------------------------------------------------------
    # Benford analysis
    # ------------------------------------------------------------------
    benford_applicable = _benford_suitability(raw_df)
    benford_score = _benford_score(raw_df) if benford_applicable else None
    limitations = []
    if not benford_applicable:
        limitations.append("Benford analysis unavailable — insufficient or unsuitable data distribution")
    else:
        # Treat Benford deviation as an additional flag if score exceeds 0.6
        raw_df["benford_deviation_flag"] = benford_score > 0.6

    # ------------------------------------------------------------------
    # Reliability indicator
    # ------------------------------------------------------------------
    reliability = _compute_reliability(feature_df, benford_applicable)

    # ------------------------------------------------------------------
    # SHAP explanations
    # ------------------------------------------------------------------
    shap_dict = compute_shap_values(iso_det, ae_det, X)
    # Convert to simple dict of feature -> contribution (average absolute value)
    shap_agg = {
        "iso": shap_dict["iso"].abs().mean().to_dict(),
        "ae": shap_dict["ae"].abs().mean().to_dict(),
    }

    # ------------------------------------------------------------------
    # Rule explanations
    # ------------------------------------------------------------------
    rule_expls = format_rule_explanations(raw_df)

    # ------------------------------------------------------------------
    # Assemble audit records
    # ------------------------------------------------------------------
    records = []
    source_file = Path(data_path).name
    for idx, row in raw_df.iterrows():
        record = build_audit_record(
            row=row.to_dict(),
            anomaly_score=float(combined_score[idx]),
            reliability_indicator=reliability,
            ml_explanation=shap_agg,
            rule_explanations=[r for r in rule_expls if r["transaction_id"] == row.get("transaction_id")],
            limitations=limitations,
            source_file=source_file,
            source_record_id=int(idx) + 1,  # 1‑based line number for traceability
        )
        records.append(record)

    # Write JSON lines
    with open(output_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    print(f"Pipeline completed – {len(records)} audit records written to {output_path}")

# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m src.pipeline <data.csv> [jurisdiction] [senior_roles_comma] [approval_json]")
        sys.exit(1)
    data_path = sys.argv[1]
    jurisdiction = sys.argv[2] if len(sys.argv) > 2 else None
    senior_roles = sys.argv[3].split(",") if len(sys.argv) > 3 else None
    approval_thresholds = json.loads(sys.argv[4]) if len(sys.argv) > 4 else None
    run_pipeline(data_path, jurisdiction, senior_roles, approval_thresholds)
