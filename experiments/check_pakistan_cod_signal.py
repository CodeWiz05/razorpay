"""
check_pakistan_cod_signal.py
==============================
Fast external validity check: does COD correlate with order
cancellation/refund in a REAL, row-level, COD-inclusive e-commerce
dataset (Pakistan, ~500K real orders, 2016-2018)? This is a check of
the CORE CAUSAL CLAIM the whole project rests on, using real
transaction rows -- not a claim that Pakistan's exact rates transfer to
India, and not a retraining/tuning exercise on this data.
"""
import urllib.request
import zipfile
import io
import pandas as pd

main_csv = "C:/Users/samaa/Downloads/Pakistan Largest Ecommerce Dataset.csv"

df = pd.read_csv(main_csv, low_memory=False)
print(f"\nShape: {df.shape}")
print("\nColumns:")
print(df.columns.tolist())
print("\nFirst 3 rows:")
print(df.head(3))

# Best-effort auto-detect the payment-method and order-status columns
payment_col = next((c for c in df.columns if "payment" in c.lower()), None)
status_col = next((c for c in df.columns if "status" in c.lower()), None)
print(f"\nDetected payment column: {payment_col}")
print(f"Detected status column: {status_col}")

if payment_col and status_col:
    print(f"\nUnique values in {payment_col}:")
    print(df[payment_col].value_counts().head(15))
    print(f"\nUnique values in {status_col}:")
    print(df[status_col].value_counts().head(15))
else:
    print("\nCould not auto-detect one or both columns -- paste the column "
          "list above and we'll adjust the detection logic.")


"""
check_pakistan_cod_signal.py -- part 2
========================================
Joint crosstab of payment_method x status, to test whether COD orders
show a higher cancellation/refund-equivalent rate than non-COD orders
in REAL transaction data.

ASSUMPTION (stated explicitly, not hidden): "canceled", "order_refunded",
and "refund" are treated as the return/RTO-equivalent outcome. "cod" and
"payment_review" appearing as STATUS values (not payment methods) are
data artifacts in this dataset and excluded from the denominator, since
their meaning is ambiguous. This is a real-data-quality issue worth
naming plainly if this check is used in the writeup -- unlike the
synthetic data, this file wasn't built by us and inherits whatever
quirks the original source has.
"""
import pandas as pd

MAIN_CSV = "C:/Users/samaa/Downloads/Pakistan Largest Ecommerce Dataset.csv"  # reuse the path printed last time

df = pd.read_csv(MAIN_CSV, low_memory=False)

# Normalize: lowercase, strip whitespace (real Kaggle exports are often messy)
df["payment_method"] = df["payment_method"].astype(str).str.strip().str.lower()
df["status"] = df["status"].astype(str).str.strip().str.lower()

# Exclude ambiguous/artifact status values from the analysis entirely
ambiguous_status = {"cod", "payment_review", "pending", "processing", "holded", "pending_paypal", "nan"}
df_clean = df[~df["status"].isin(ambiguous_status)].copy()

RETURN_EQUIVALENT = {"canceled", "order_refunded", "refund"}
df_clean["return_equivalent"] = df_clean["status"].isin(RETURN_EQUIVALENT).astype(int)

df_clean["is_cod"] = df_clean["payment_method"].isin({"cod", "cashatdoorstep"}).astype(int)

print(f"Rows used in analysis: {len(df_clean):,} / {len(df):,} "
      f"({len(df_clean)/len(df):.1%} -- rest excluded as ambiguous status)")

print("\nReturn-equivalent rate by payment type:")
summary = df_clean.groupby("is_cod")["return_equivalent"].agg(["mean", "count"])
summary.index = summary.index.map({0: "Non-COD", 1: "COD"})
print(summary)

cod_rate = summary.loc["COD", "mean"]
noncod_rate = summary.loc["Non-COD", "mean"]
print(f"\nCOD:Non-COD ratio: {cod_rate/noncod_rate:.2f}x")

# Breakdown by individual status, for transparency on what's driving this
print("\nStatus breakdown, COD vs Non-COD (% within each group):")
print(pd.crosstab(df_clean["status"], df_clean["is_cod"].map({0:"Non-COD",1:"COD"}), normalize="columns").round(3))