"""
verify_leakage_invariant.py
=============================
Run this against the FRESH data_orders.csv after regenerating it.
Independently recomputes the expanding-window customer_past_return_rate
and customer_prior_orders from scratch and checks it matches what's
stored in the file, row by row. Also checks the stored values don't
accidentally match the full-history (leaky) version instead.
"""
import pandas as pd
import numpy as np

DATA_PATH = "C:/Numair/Coding/Razorpay/data_orders.csv"

df = pd.read_csv(DATA_PATH, parse_dates=["order_date"])
df = df.sort_values("order_date").reset_index(drop=True)

counts = {}
recomputed_rate = np.zeros(len(df))
recomputed_prior = np.zeros(len(df), dtype=int)
for i in range(len(df)):
    cid = df.at[i, "customer_id"]
    n_orders, n_returns = counts.get(cid, (0, 0))
    recomputed_prior[i] = n_orders
    recomputed_rate[i] = (n_returns / n_orders) if n_orders >= 2 else 0.12
    counts[cid] = (n_orders + 1, n_returns + int(df.at[i, "returned"]))

df["recomputed_rate"] = np.round(recomputed_rate, 3)
df["recomputed_prior"] = recomputed_prior

mismatch_rate = (df["recomputed_rate"] != df["customer_past_return_rate"]).sum()
mismatch_prior = (df["recomputed_prior"] != df["customer_prior_orders"]).sum()
print(f"Mismatches in past_return_rate: {mismatch_rate} / {len(df)}")
print(f"Mismatches in prior_orders: {mismatch_prior} / {len(df)}")

full_hist_rate = df.groupby("customer_id")["returned"].transform("mean")
leaky_match = np.isclose(full_hist_rate, df["customer_past_return_rate"], atol=0.01).sum()
print(f"Rows matching FULL-HISTORY (leaky) rate: {leaky_match} / {len(df)}  (should be far less than total)")

print("\nDistribution check:")
print(f"Overall return rate: {df['returned'].mean():.2%}")
print(df.groupby("payment_mode")["returned"].mean())
ratio = df.groupby("payment_mode")["returned"].mean()
print(f"COD:Prepaid ratio: {ratio['COD']/ratio['Prepaid']:.2f}x")