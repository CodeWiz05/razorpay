"""
features.py
============
Loads orders.csv and builds the model-ready feature matrix.

LEAKAGE NOTES (read before changing anything):
  - `customer_past_return_rate` and `customer_prior_orders` were already
    computed in generate_data.py using ONLY orders strictly before the
    current one, per customer, in time order. If you rebuild this from a
    real dataset, you MUST reproduce that expanding-window logic -- do NOT
    compute a customer's return rate over their FULL history and then
    join it back onto their earlier orders. That is the single most common
    leakage bug in return/fraud modeling.
  - The train/val/test split in train.py is TEMPORAL (by order_date), not
    random. A random split leaks future customer behavior into training.
"""
import pandas as pd

CATEGORICAL = ["category", "payment_mode", "pincode_tier"]
NUMERIC = [
    "price", "discount_pct", "delivery_days", "is_apparel",
    "customer_prior_orders", "customer_past_return_rate",
]
TARGET = "returned"
DATA_PATH = "C:/Numair/Coding/Razorpay/data_orders.csv"


def load_features(path=DATA_PATH):
    df = pd.read_csv(path, parse_dates=["order_date"])
    df = pd.get_dummies(df, columns=CATEGORICAL, drop_first=False)
    dummy_cols = [c for c in df.columns
                  if c.startswith(tuple(f"{c_}_" for c_ in CATEGORICAL))]
    feature_cols = NUMERIC + dummy_cols
    return df, feature_cols