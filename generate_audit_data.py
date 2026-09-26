"""
Synthetic GL / audit transaction generator.

Produces (in /mnt/user-data/outputs/):
  - vendor_master.csv             : vendor records (id, name, category, created_date, bank hash)
  - gl_transactions_labeled.csv   : transactions WITH is_anomaly + anomaly_type (train/eval)
  - gl_transactions_unlabeled.csv : same transactions, no labels (feed to your detection pipeline)
  - client_variant_schema.csv     : same transactions, renamed/reordered/reformatted columns,
                                     no labels -- simulates an unfamiliar client export, for
                                     testing the ingestion schema-mapping layer end-to-end
  - ground_truth.csv              : transaction_id, is_anomaly, anomaly_type (for scoring)

Anomaly types injected: duplicate_payment, split_invoice, new_vendor_large_payment,
weekend_senior_posting, benford_violation.

Run:
    pip install pandas numpy --break-system-packages
    python generate_audit_data.py
"""

import random
import string
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

RNG_SEED = 42
random.seed(RNG_SEED)
np.random.seed(RNG_SEED)

# ---------------------------------------------------------------------------
# Config -- tweak these to change dataset size/shape
# ---------------------------------------------------------------------------
START_DATE = datetime(2025, 1, 1)
END_DATE = datetime(2025, 12, 31)
N_VENDORS = 45
N_NEW_VENDORS_DURING_PERIOD = 5      # created mid-period; feeds "new vendor" anomaly
AVG_TXNS_PER_BUSINESS_DAY = 18
ANOMALY_RATE = 0.025                  # ~2.5% of normal-volume transactions become anomalies

ACCOUNTS = [
    "6010-OfficeSupplies", "6020-Consulting", "6030-Travel", "6040-Utilities",
    "6050-Marketing", "6060-ITServices", "6070-Maintenance", "6080-Insurance",
    "6090-Freight", "6100-Legal",
]

VENDOR_CATEGORY_MEAN_AMOUNT = {
    "OfficeSupplies": 400, "Consulting": 8000, "Travel": 1200, "Utilities": 2200,
    "Marketing": 5000, "ITServices": 6000, "Maintenance": 1500, "Insurance": 9000,
    "Freight": 900, "Legal": 7000,
}

# name, seniority, approval_limit
APPROVERS = [
    ("R. Shah", "junior", 25000),
    ("A. Mehta", "junior", 25000),
    ("N. Iyer", "senior", 150000),
    ("K. Verma", "senior", 150000),
    ("P. Bose", "senior", 500000),
]

VENDOR_NAME_PARTS_1 = ["Bright", "Summit", "Prime", "Northgate", "Bluewave", "Silverline",
                        "Metro", "Coreline", "Vertex", "Redstone", "Horizon", "Anchor"]
VENDOR_NAME_PARTS_2 = ["Supplies", "Consulting", "Logistics", "Traders", "Solutions",
                        "Services", "Industries", "Enterprises", "Systems", "Partners"]

OUT_DIR = "/mnt/user-data/outputs"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def business_days(start, end):
    days, d = [], start
    while d <= end:
        if d.weekday() < 5:
            days.append(d)
        d += timedelta(days=1)
    return days


def weekend_days(start, end):
    days, d = [], start
    while d <= end:
        if d.weekday() >= 5:
            days.append(d)
        d += timedelta(days=1)
    return days


def random_vendor_name(used_names):
    while True:
        name = f"{random.choice(VENDOR_NAME_PARTS_1)} {random.choice(VENDOR_NAME_PARTS_2)}"
        if name not in used_names:
            used_names.add(name)
            return name


def bank_hash():
    return "BANK-" + "".join(random.choices(string.digits, k=10))


def approver_for_amount(amount):
    eligible = sorted([a for a in APPROVERS if a[2] >= amount], key=lambda a: a[2])
    if not eligible:
        eligible = [APPROVERS[-1]]
    weights = [1 / (i + 1) for i in range(len(eligible))]
    return random.choices(eligible, weights=weights, k=1)[0]


def new_transaction_id(counter):
    tid = f"TXN{counter[0]:07d}"
    counter[0] += 1
    return tid


# ---------------------------------------------------------------------------
# 1. Vendor master
# ---------------------------------------------------------------------------
def build_vendor_master():
    used_names = set()
    vendors = []
    categories = list(VENDOR_CATEGORY_MEAN_AMOUNT.keys())

    for i in range(N_VENDORS - N_NEW_VENDORS_DURING_PERIOD):
        vendors.append({
            "vendor_id": f"V{i:04d}",
            "vendor_name": random_vendor_name(used_names),
            "category": random.choice(categories),
            "created_date": START_DATE - timedelta(days=random.randint(60, 2000)),
            "bank_account_hash": bank_hash(),
        })

    for j in range(N_NEW_VENDORS_DURING_PERIOD):
        i = N_VENDORS - N_NEW_VENDORS_DURING_PERIOD + j
        vendors.append({
            "vendor_id": f"V{i:04d}",
            "vendor_name": random_vendor_name(used_names),
            "category": random.choice(categories),
            "created_date": START_DATE + timedelta(days=random.randint(30, 300)),
            "bank_account_hash": bank_hash(),
        })

    return pd.DataFrame(vendors)


# ---------------------------------------------------------------------------
# 2. Normal transactions
# ---------------------------------------------------------------------------
def build_normal_transactions(vendor_df, counter):
    rows = []
    old_vendors = vendor_df[vendor_df["created_date"] < START_DATE]

    for day in business_days(START_DATE, END_DATE):
        for _ in range(np.random.poisson(AVG_TXNS_PER_BUSINESS_DAY)):
            vendor = old_vendors.sample(1).iloc[0]
            mean_amt = VENDOR_CATEGORY_MEAN_AMOUNT[vendor["category"]]
            amount = round(float(np.random.lognormal(mean=np.log(mean_amt), sigma=0.5)), 2)
            approver = approver_for_amount(amount)
            matches = [a for a in ACCOUNTS if vendor["category"] in a]
            account = matches[0] if matches else random.choice(ACCOUNTS)

            rows.append({
                "transaction_id": new_transaction_id(counter),
                "date": day,
                "vendor_id": vendor["vendor_id"],
                "vendor_name": vendor["vendor_name"],
                "account": account,
                "amount": amount,
                "approver": approver[0],
                "approver_seniority": approver[1],
                "description": f"Payment to {vendor['vendor_name']} - {vendor['category']}",
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3. Anomaly injection
# ---------------------------------------------------------------------------
def inject_duplicate_payments(df, counter, n):
    rows = []
    for _, row in df.sample(n).iterrows():
        rows.append({
            **row.to_dict(),
            "transaction_id": new_transaction_id(counter),
            "date": row["date"] + timedelta(days=random.choice([0, 1, 2])),
        })
    out = pd.DataFrame(rows)
    out["anomaly_type"] = "duplicate_payment"
    return out


def inject_split_invoices(vendor_df, counter, n_events):
    old_vendors = vendor_df[vendor_df["created_date"] < START_DATE]
    limit = APPROVERS[0][2]
    rows = []
    for _ in range(n_events):
        vendor = old_vendors.sample(1).iloc[0]
        total = limit * random.uniform(1.6, 2.6)
        n_parts = random.choice([2, 3])
        parts = np.random.dirichlet(np.ones(n_parts)) * total
        base_day = random.choice(business_days(START_DATE, END_DATE))
        for k, part_amt in enumerate(parts):
            amt = round(min(float(part_amt), limit * 0.97), 2)
            approver = approver_for_amount(amt)
            rows.append({
                "transaction_id": new_transaction_id(counter),
                "date": base_day + timedelta(days=k),
                "vendor_id": vendor["vendor_id"],
                "vendor_name": vendor["vendor_name"],
                "account": random.choice(ACCOUNTS),
                "amount": amt,
                "approver": approver[0],
                "approver_seniority": approver[1],
                "description": f"Payment to {vendor['vendor_name']} - split invoice part {k+1}",
                "anomaly_type": "split_invoice",
            })
    return pd.DataFrame(rows)


def inject_new_vendor_large_payment(vendor_df, counter, n):
    new_vendors = vendor_df[vendor_df["created_date"] >= START_DATE]
    n = min(n, len(new_vendors))
    rows = []
    for _, vendor in new_vendors.sample(n, replace=False).iterrows():
        amount = round(random.uniform(40000, 120000), 2)
        approver = approver_for_amount(amount)
        rows.append({
            "transaction_id": new_transaction_id(counter),
            "date": vendor["created_date"] + timedelta(days=random.randint(0, 5)),
            "vendor_id": vendor["vendor_id"],
            "vendor_name": vendor["vendor_name"],
            "account": random.choice(ACCOUNTS),
            "amount": amount,
            "approver": approver[0],
            "approver_seniority": approver[1],
            "description": f"Payment to {vendor['vendor_name']} - new vendor onboarding",
            "anomaly_type": "new_vendor_large_payment",
        })
    return pd.DataFrame(rows)


def inject_weekend_senior_postings(vendor_df, counter, n):
    old_vendors = vendor_df[vendor_df["created_date"] < START_DATE]
    w_days = weekend_days(START_DATE, END_DATE)
    senior = [a for a in APPROVERS if a[1] == "senior"]
    rows = []
    for _ in range(n):
        vendor = old_vendors.sample(1).iloc[0]
        amount = round(random.uniform(20000, 80000), 2)
        approver = random.choice(senior)
        rows.append({
            "transaction_id": new_transaction_id(counter),
            "date": random.choice(w_days),
            "vendor_id": vendor["vendor_id"],
            "vendor_name": vendor["vendor_name"],
            "account": random.choice(ACCOUNTS),
            "amount": amount,
            "approver": approver[0],
            "approver_seniority": approver[1],
            "description": f"Payment to {vendor['vendor_name']} - weekend posting",
            "anomaly_type": "weekend_senior_posting",
        })
    return pd.DataFrame(rows)


def inject_benford_violations(vendor_df, counter, n):
    """Fabricated amounts with an unnaturally high rate of leading digit 9 --
    organic amounts almost never cluster like this at scale (Benford's Law)."""
    old_vendors = vendor_df[vendor_df["created_date"] < START_DATE]
    b_days = business_days(START_DATE, END_DATE)
    rows = []
    for _ in range(n):
        vendor = old_vendors.sample(1).iloc[0]
        magnitude = random.choice([100, 1000, 10000])
        amount = round(9 * magnitude * random.uniform(1.0, 1.09), 2)
        approver = approver_for_amount(amount)
        rows.append({
            "transaction_id": new_transaction_id(counter),
            "date": random.choice(b_days),
            "vendor_id": vendor["vendor_id"],
            "vendor_name": vendor["vendor_name"],
            "account": random.choice(ACCOUNTS),
            "amount": amount,
            "approver": approver[0],
            "approver_seniority": approver[1],
            "description": f"Payment to {vendor['vendor_name']} - adjustment",
            "anomaly_type": "benford_violation",
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 4. Assemble + write outputs
# ---------------------------------------------------------------------------
def main():
    vendor_df = build_vendor_master()
    counter = [1]

    normal_df = build_normal_transactions(vendor_df, counter)
    n_anomalies = int(len(normal_df) * ANOMALY_RATE)

    n_dup = max(2, n_anomalies // 5)
    n_split_events = max(1, n_anomalies // 15)   # each event -> 2-3 rows
    n_new_vendor = max(2, n_anomalies // 6)
    n_weekend = max(2, n_anomalies // 6)
    n_benford = max(2, n_anomalies // 5)

    dup_df = inject_duplicate_payments(normal_df, counter, n_dup)
    split_df = inject_split_invoices(vendor_df, counter, n_split_events)
    newv_df = inject_new_vendor_large_payment(vendor_df, counter, n_new_vendor)
    wknd_df = inject_weekend_senior_postings(vendor_df, counter, n_weekend)
    benf_df = inject_benford_violations(vendor_df, counter, n_benford)

    normal_df["anomaly_type"] = "none"
    all_df = pd.concat([normal_df, dup_df, split_df, newv_df, wknd_df, benf_df],
                        ignore_index=True)
    all_df["is_anomaly"] = all_df["anomaly_type"] != "none"
    all_df = all_df.sample(frac=1, random_state=RNG_SEED).reset_index(drop=True)
    all_df["date"] = pd.to_datetime(all_df["date"]).dt.strftime("%Y-%m-%d")

    vendor_out = vendor_df.copy()
    vendor_out["created_date"] = pd.to_datetime(vendor_out["created_date"]).dt.strftime("%Y-%m-%d")
    vendor_out.to_csv(f"{OUT_DIR}/vendor_master.csv", index=False)

    all_df[["transaction_id", "is_anomaly", "anomaly_type"]].to_csv(
        f"{OUT_DIR}/ground_truth.csv", index=False)

    labeled_cols = ["transaction_id", "date", "vendor_id", "vendor_name", "account",
                     "amount", "approver", "approver_seniority", "description",
                     "is_anomaly", "anomaly_type"]
    all_df[labeled_cols].to_csv(f"{OUT_DIR}/gl_transactions_labeled.csv", index=False)

    unlabeled_cols = ["transaction_id", "date", "vendor_id", "vendor_name", "account",
                       "amount", "approver", "approver_seniority", "description"]
    all_df[unlabeled_cols].to_csv(f"{OUT_DIR}/gl_transactions_unlabeled.csv", index=False)

    # "client variant" schema: renamed/reordered columns + different date format --
    # simulates an unfamiliar client export, for testing the ingestion mapping layer.
    variant = all_df[unlabeled_cols].copy()
    variant["date"] = pd.to_datetime(variant["date"]).dt.strftime("%d-%b-%Y")
    variant = variant.rename(columns={
        "transaction_id": "Entry Ref", "date": "Posting Date",
        "vendor_id": "Supplier Code", "vendor_name": "Supplier",
        "account": "GL Code", "amount": "Value (INR)",
        "approver": "Authorised By", "approver_seniority": "Authoriser Level",
        "description": "Narration",
    })
    variant = variant[["Posting Date", "Supplier", "Supplier Code", "Value (INR)",
                        "GL Code", "Narration", "Authorised By", "Authoriser Level",
                        "Entry Ref"]]
    variant.to_csv(f"{OUT_DIR}/client_variant_schema.csv", index=False)

    print(f"Vendors: {len(vendor_df)}")
    print(f"Total transactions: {len(all_df)}")
    print(f"Anomalies: {all_df['is_anomaly'].sum()} ({all_df['is_anomaly'].mean()*100:.2f}%)")
    print(all_df[all_df["is_anomaly"]]["anomaly_type"].value_counts())
    print(f"\nFiles written to {OUT_DIR}/:")
    print(" - vendor_master.csv")
    print(" - gl_transactions_labeled.csv   (has is_anomaly + anomaly_type -- for training/eval)")
    print(" - gl_transactions_unlabeled.csv (no labels -- feed this to your detection pipeline)")
    print(" - client_variant_schema.csv     (renamed/reformatted -- feed this to test ingestion mapping)")
    print(" - ground_truth.csv              (transaction_id, is_anomaly, anomaly_type -- for scoring)")


if __name__ == "__main__":
    main()
