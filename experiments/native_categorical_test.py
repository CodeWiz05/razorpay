"""
experiments/native_categorical_test.py
========================================
Isolated test: does HGB's native categorical splitting (categorical_features
="from_dtype") recover the apparel x prepaid interaction (is_bracketed)
better than manual one-hot encoding? Compares permutation importance of
is_bracketed and Prepaid-segment ROC-AUC under both encodings, same model,
same hyperparams, same sample weighting. Self-contained -- does not touch
features.py or train.py.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.inspection import permutation_importance

DATA_PATH = str(Path(__file__).resolve().parent.parent / "data_orders.csv")
TARGET = "returned"
CATEGORICAL = ["category", "payment_mode", "pincode_tier"]
NUMERIC = [
    "price", "discount_pct", "delivery_days", "is_apparel",
    "customer_prior_orders", "customer_past_return_rate", "is_festive",
    "is_bracketed", "size_variant_count",
]

df_raw = pd.read_csv(DATA_PATH, parse_dates=["order_date"]).sort_values("order_date").reset_index(drop=True)

t1 = df_raw["order_date"].quantile(8 / 12)
t2 = df_raw["order_date"].quantile(10 / 12)

def split(d):
    return (d[d["order_date"] < t1], d[(d["order_date"] >= t1) & (d["order_date"] < t2)], d[d["order_date"] >= t2])

def sample_weights(train_df, ytr):
    train_mode = train_df["payment_mode"].map({"Prepaid": "Prepaid", "COD": "COD"})
    bucket_counts = train_mode.astype(str).str.cat(ytr.astype(str), sep="_").value_counts()
    n_buckets = 4
    w = np.ones(len(ytr))
    for i in range(len(ytr)):
        key = f"{train_mode.iloc[i]}_{ytr.iloc[i]}"
        w[i] = len(ytr) / (n_buckets * bucket_counts[key])
    return w ** 0.3

def fit_and_eval(Xtr, ytr, Xval, yval, sw, feature_cols, categorical_features=None, label=""):
    hgb = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.06, max_depth=5, random_state=42,
        early_stopping=True, validation_fraction=0.15,
        categorical_features=categorical_features,
    )
    hgb.fit(Xtr, ytr, sample_weight=sw)
    val_proba = hgb.predict_proba(Xval)[:, 1]
    ppd_mask = (Xval["payment_mode"] == "Prepaid").values if "payment_mode" in Xval.columns \
        else (Xval["payment_mode_Prepaid"] == True).values
    ppd_rocauc = roc_auc_score(yval[ppd_mask], val_proba[ppd_mask])
    ppd_prauc = average_precision_score(yval[ppd_mask], val_proba[ppd_mask])
    perm = permutation_importance(hgb, Xval, yval, scoring="average_precision", n_repeats=10, random_state=42)
    bracket_idx = feature_cols.index("is_bracketed")
    print(f"\n[{label}] Prepaid ROC-AUC={ppd_rocauc:.3f}  Prepaid PR-AUC={ppd_prauc:.3f}  "
          f"is_bracketed perm-importance={perm.importances_mean[bracket_idx]:.4f}")
    return hgb, val_proba

# ---- Encoding A: one-hot (current production approach) ----
df_oh = pd.get_dummies(df_raw, columns=CATEGORICAL, drop_first=False)
dummy_cols = [c for c in df_oh.columns if c.startswith(tuple(f"{c_}_" for c_ in CATEGORICAL))]
feature_cols_oh = NUMERIC + dummy_cols
train_oh, val_oh, test_oh = split(df_oh)
sw_oh = sample_weights(df_raw[df_raw["order_date"] < t1], train_oh[TARGET])
fit_and_eval(train_oh[feature_cols_oh], train_oh[TARGET], val_oh[feature_cols_oh], val_oh[TARGET],
             sw_oh, feature_cols_oh, categorical_features=None, label="ONE-HOT (current)")

# ---- Encoding B: native categorical dtype ----
df_native = df_raw.copy()
for c in CATEGORICAL:
    df_native[c] = df_native[c].astype("category")
feature_cols_native = NUMERIC + CATEGORICAL
train_nat, val_nat, test_nat = split(df_native)
sw_nat = sample_weights(train_nat, train_nat[TARGET])
fit_and_eval(train_nat[feature_cols_native], train_nat[TARGET], val_nat[feature_cols_native], val_nat[TARGET],
             sw_nat, feature_cols_native, categorical_features="from_dtype", label="NATIVE CATEGORICAL")