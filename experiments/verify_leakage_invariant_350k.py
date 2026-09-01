import pandas as pd
import numpy as np

DATA_PATH = "C:/Numair/Coding/Razorpay/data_orders_350k.csv"

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
print(f"Rows matching FULL-HISTORY (leaky) rate: {leaky_match} / {len(df)}")

print(f"\nOrders per customer: mean={df.groupby('customer_id').size().mean():.2f}  "
      f"(target ~2.73, matching production's ratio)")