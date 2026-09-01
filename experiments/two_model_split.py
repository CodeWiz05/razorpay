"""
two_model_split_investigation.py
===================================
ARCHITECTURE EXPERIMENT: does splitting into two independent models
(hgb_cod, hgb_prepaid) beat the single shared model + macro-cost-threshold
approach currently in production?

MOTIVATION: tonight's price/delivery_days recalibration (COD-specific
terms) made COD's signal cleaner but, as an uncosted side effect, made
prepaid's already-thin signal thinner still. The shared model's single
threshold, chosen by macro-cost, has to drag lower to keep chasing
prepaid's shrinking positive pool -- which floods COD with false
positives it doesn't need. Research this session also found COD and
prepaid are driven by different causal mechanisms entirely (COD:
doorstep-refusal psychology; Prepaid: post-delivery fit/quality) -- a
single shared decision boundary may be structurally the wrong fit for
two different generative processes, not just a tuning problem.

DISCIPLINE: reloads data and the FROZEN production model independently
(never touches train.py's live state). VAL used for threshold selection
per segment. TEST touched exactly once per model, at the end.
"""
import numpy as np
import pandas as pd
import joblib
import json
import sys
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    average_precision_score, roc_auc_score, precision_score, recall_score,
    f1_score, confusion_matrix, matthews_corrcoef,
)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from features import load_features, TARGET

OUT = "C:/Numair/Coding/Razorpay/outputs"

# ---------------- Reload data split (identical to train.py) ----------------
df, feature_cols = load_features()
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

# ---------------- Split each of train/val/test by payment_mode -------------
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

# ---------------- Fit two independent models --------------------------
# class_weight="balanced" here addresses WITHIN-segment class imbalance
# (COD ~28% positive, Prepaid ~3.7% positive) -- a different, legitimate
# use from the shared-model version, which was trying to paper over
# CROSS-segment volume imbalance instead. No sample-weighting hack needed
# here since there's no cross-segment tension left to correct for.
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

# ---------------- Independent cost-optimal threshold per segment -----------
t_cod, val_cost_cod = cost_optimal_threshold(val_proba_cod, val_cod[TARGET].values)
t_ppd, val_cost_ppd = cost_optimal_threshold(val_proba_ppd, val_ppd[TARGET].values)
print(f"\nThresholds (independent, no macro/blended tension needed): COD={t_cod:.3f}  Prepaid={t_ppd:.3f}")

# ---------------- FROZEN TEST evaluation, once per model -------------------
test_proba_cod = hgb_cod.predict_proba(test_cod[feature_cols])[:, 1]
test_proba_ppd = hgb_ppd.predict_proba(test_ppd[feature_cols])[:, 1]
test_pred_cod = (test_proba_cod >= t_cod).astype(int)
test_pred_ppd = (test_proba_ppd >= t_ppd).astype(int)

# ---------------- Combine into overall metrics (matching production's shape) -
y_combined = pd.concat([test_cod[TARGET], test_ppd[TARGET]]).values
pred_combined = np.concatenate([test_pred_cod, test_pred_ppd])
proba_combined = np.concatenate([test_proba_cod, test_proba_ppd])

tn, fp, fn, tp = confusion_matrix(y_combined, pred_combined).ravel()
cost = fp * COST_FP + fn * COST_FN
prauc = average_precision_score(y_combined, proba_combined)
mcc = matthews_corrcoef(y_combined, pred_combined)
precision = precision_score(y_combined, pred_combined)
recall = recall_score(y_combined, pred_combined)
f1 = f1_score(y_combined, pred_combined)

fn_mask = (pred_combined == 0) & (y_combined == 1)
# reconstruct which combined-array rows are prepaid (all test_ppd rows come after test_cod rows)
ppd_mask_combined = np.concatenate([np.zeros(len(test_cod), dtype=bool), np.ones(len(test_ppd), dtype=bool)])
fn_prepaid_count = (fn_mask & ppd_mask_combined).sum()
fn_prepaid_share = fn_prepaid_count / fn_mask.sum() if fn_mask.sum() > 0 else float("nan")

print("\n================ TWO-MODEL SPLIT: combined TEST result ================")
print(f"TN={tn} FP={fp} FN={fn} TP={tp}")
print(f"Precision={precision:.3f}  Recall={recall:.3f}  F1={f1:.3f}  PR-AUC={prauc:.3f}  MCC={mcc:.3f}")
print(f"Cost=₹{cost:,.0f}")
print(f"FN prepaid count={int(fn_prepaid_count)} ({fn_prepaid_share:.1%} of all FN)")

# ---------------- Load the FROZEN production (shared) model for comparison -
model_shared = joblib.load(f"{OUT}/model.joblib")
summary_shared = json.load(open(f"{OUT}/summary.json"))
t_shared = summary_shared["best_threshold"]

Xtest_full = test[feature_cols]
ytest_full = test[TARGET]
proba_shared = model_shared.predict_proba(Xtest_full)[:, 1]
pred_shared = (proba_shared >= t_shared).astype(int)

tn_s, fp_s, fn_s, tp_s = confusion_matrix(ytest_full, pred_shared).ravel()
cost_s = fp_s * COST_FP + fn_s * COST_FN
prauc_s = average_precision_score(ytest_full, proba_shared)
mcc_s = matthews_corrcoef(ytest_full, pred_shared)

ppd_mask_full = (test["payment_mode_Prepaid"] == True).values
fn_mask_s = (pred_shared == 0) & (ytest_full.values == 1)
fn_prepaid_s = (ppd_mask_full & fn_mask_s).sum()

print("\n================ SHARED MODEL (current production) : TEST result ================")
print(f"TN={tn_s} FP={fp_s} FN={fn_s} TP={tp_s}")
print(f"PR-AUC={prauc_s:.3f}  MCC={mcc_s:.3f}  Cost=₹{cost_s:,.0f}")
print(f"FN prepaid count={int(fn_prepaid_s)} ({fn_prepaid_s/fn_mask_s.sum():.1%} of all FN)")

print("\n================ HEAD-TO-HEAD ================")
print(f"{'Metric':<20}{'Shared (prod)':<16}{'Two-model split':<16}")
print(f"{'PR-AUC':<20}{prauc_s:<16.3f}{prauc:<16.3f}")
print(f"{'MCC':<20}{mcc_s:<16.3f}{mcc:<16.3f}")
print(f"{'Cost (Rs.)':<20}{cost_s:<16,.0f}{cost:<16,.0f}")
print(f"{'FP':<20}{int(fp_s):<16}{int(fp):<16}")
print(f"{'FN':<20}{int(fn_s):<16}{int(fn):<16}")
print(f"{'FN prepaid count':<20}{int(fn_prepaid_s):<16}{int(fn_prepaid_count):<16}")