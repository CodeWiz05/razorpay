"""
features.py
============
Loads orders.csv and builds the model-ready feature matrix.

LEAKAGE WARNING: customer_past_return_rate / customer_prior_orders use an
expanding window (prior orders only, per customer, in time order). If
rebuilding from real data, do NOT compute over full history then join back
-- this is the most common leakage bug in return/fraud modeling. The
train/val/test split in train.py is temporal (order_date), not random.
"""
import pandas as pd
from pathlib import Path

CATEGORICAL = ["category", "payment_mode", "pincode_tier"]
NUMERIC = [
    "price", "discount_pct", "delivery_days", "is_apparel",
    "customer_prior_orders", "customer_past_return_rate", "is_festive",
    "is_bracketed", "size_variant_count", "product_past_return_rate" 
]
TARGET = "returned"
DATA_PATH = str(Path(__file__).resolve().parent / "data_orders.csv")


def load_features(path=DATA_PATH):
    df = pd.read_csv(path, parse_dates=["order_date"])
    df = pd.get_dummies(df, columns=CATEGORICAL, drop_first=False)
    dummy_cols = [c for c in df.columns
                  if c.startswith(tuple(f"{c_}_" for c_ in CATEGORICAL))]
    feature_cols = NUMERIC + dummy_cols
    return df, feature_cols