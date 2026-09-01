"""
two_model_split_350k.py
=========================
Runs the SAME two-model-split architecture test as
two_model_split_investigation.py, but on the 350k scaled dataset --
answers the actual question: does a ~6x larger prepaid-positive pool
let the Prepaid model learn a real signal (higher standalone PR-AUC/
ROC-AUC), or does it stay near-random regardless of volume?

Self-contained: builds its own feature matrix from data_orders_350k.csv
rather than importing features.py (which is hardcoded to the production
60k file) -- keeps this experiment fully isolated from the production
pipeline.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    average_precision_score, roc_auc_score, precision_score, recall_score,
    f1_score, confusion_matrix, matthews_corrcoef,
)

DATA_PATH = "C:/Numair/Coding/Razorpay/data_orders_350k.csv"
TARGET = "returned"
CATEGORICAL = ["category", "payment_mode", "pincode_tier"]
NUMERIC = [
    "price", "discount_pct", "delivery_days", "is_apparel",
    "customer_prior_orders", "customer_past_return_rate",
]

df = pd.read_csv(DATA_PATH, parse_dates=["order_date"])
df = pd.get_dummies(df, columns=CATEGORICAL, drop_first=False)
dummy_cols = [c for c in df.columns if c.startswith(tuple(f"{c_}_" for c_ in CATEGORICAL))]
feature_cols = NUMERIC + dummy_cols
df = df.sort_values("order_date").reset_index(drop=True)

t1 = df["order_date"].quantile(8 / 12)
t2 = df["order_date"].quantile(10 / 12)
train = df[df["order_date"] < t1]
val = df[(df["order_date"] >= t1) & (df["order_date"] < t2)]
test = df[df["order_date"] >= t2]

COST_FN, COST_FP = 180.0, 25.0

def cost_optimal_threshold(proba, y_true, cost_fn=COST_FN, cost_fp=COST_FP):
    ths = np.linspace(0.05, 0.95, 181)
    best_t, best_cost = None, np.inf
    for t in ths:
        pred = (proba >= t).astype(int)
        fp = ((pred == 1) & (y_true == 0)).sum()
        fn = ((pred == 0) & (y_true == 1)).sum()
        c = fp * cost_fp + fn * cost_fn
        if c < best_cost:
            best_cost, best_t = c, t
    return best_t, best_cost

def segment(d, cod=True):
    mask = (d["payment_mode_COD"] == True) if cod else (d["payment_mode_Prepaid"] == True)
    return d[mask]

train_cod, train_ppd = segment(train, True), segment(train, False)
val_cod, val_ppd = segment(val, True), segment(val, False)
test_cod, test_ppd = segment(test, True), segment(test, False)

print(f"Train  COD={len(train_cod):,} (pos={train_cod[TARGET].sum()})   "
      f"Prepaid={len(train_ppd):,} (pos={train_ppd[TARGET].sum()})")
print(f"Val    COD={len(val_cod):,} (pos={val_cod[TARGET].sum()})   "
      f"Prepaid={len(val_ppd):,} (pos={val_ppd[TARGET].sum()})")
print(f"Test   COD={len(test_cod):,} (pos={test_cod[TARGET].sum()})   "
      f"Prepaid={len(test_ppd):,} (pos={test_ppd[TARGET].sum()})")
print(f"\n[Compare: production 60k had Train Prepaid pos=677 -- this run should be ~5.8x that]")

def fit_hgb(Xtr, ytr):
    m = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.06, max_depth=5,
        class_weight="balanced", random_state=42,
        early_stopping=True, validation_fraction=0.15,
    )
    m.fit(Xtr, ytr)
    return m

hgb_cod = fit_hgb(train_cod[feature_cols], train_cod[TARGET])
hgb_ppd = fit_hgb(train_ppd[feature_cols], train_ppd[TARGET])

val_proba_cod = hgb_cod.predict_proba(val_cod[feature_cols])[:, 1]
val_proba_ppd = hgb_ppd.predict_proba(val_ppd[feature_cols])[:, 1]

print(f"\n[COD model]     val PR-AUC={average_precision_score(val_cod[TARGET], val_proba_cod):.3f} "
      f"ROC-AUC={roc_auc_score(val_cod[TARGET], val_proba_cod):.3f}")
print(f"[Prepaid model] val PR-AUC={average_precision_score(val_ppd[TARGET], val_proba_ppd):.3f} "
      f"ROC-AUC={roc_auc_score(val_ppd[TARGET], val_proba_ppd):.3f}")
print(f"[Compare: production 60k Prepaid model had val PR-AUC=0.119 ROC-AUC=0.729]")

t_cod, _ = cost_optimal_threshold(val_proba_cod, val_cod[TARGET].values)
t_ppd, _ = cost_optimal_threshold(val_proba_ppd, val_ppd[TARGET].values)
print(f"\nThresholds: COD={t_cod:.3f}  Prepaid={t_ppd:.3f}  (compare: 60k had COD=0.180 Prepaid=0.780)")

test_proba_cod = hgb_cod.predict_proba(test_cod[feature_cols])[:, 1]
test_proba_ppd = hgb_ppd.predict_proba(test_ppd[feature_cols])[:, 1]
test_pred_cod = (test_proba_cod >= t_cod).astype(int)
test_pred_ppd = (test_proba_ppd >= t_ppd).astype(int)

y_combined = pd.concat([test_cod[TARGET], test_ppd[TARGET]]).values
pred_combined = np.concatenate([test_pred_cod, test_pred_ppd])

tn, fp, fn, tp = confusion_matrix(y_combined, pred_combined).ravel()
cost = fp * COST_FP + fn * COST_FN
mcc = matthews_corrcoef(y_combined, pred_combined)

fn_mask = (pred_combined == 0) & (y_combined == 1)
ppd_mask_combined = np.concatenate([np.zeros(len(test_cod), dtype=bool), np.ones(len(test_ppd), dtype=bool)])
fn_prepaid_count = (fn_mask & ppd_mask_combined).sum()

print("\n================ 350k TWO-MODEL SPLIT: combined TEST result ================")
print(f"TN={tn} FP={fp} FN={fn} TP={tp}  MCC={mcc:.3f}  Cost=₹{cost:,.0f}")
print(f"FN prepaid count={int(fn_prepaid_count)} ({fn_prepaid_count/fn_mask.sum():.1%} of all FN)")
print(f"[Compare: 60k two-model split had FN=167, prepaid FN=165 (98.8%), cost=₹127,810]")