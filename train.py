"""
train.py
=========
Temporal train/val/test split -> baseline + gradient-boosted model ->
threshold selection on VALIDATION only -> single frozen evaluation on TEST.

Split (by order_date, NOT random):
  Train : first 8 months
  Val   : next 2 months   (used ONLY for threshold + hyperparameter choice)
  Test  : final 2 months  (touched exactly once, at the very end)
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    precision_recall_curve, average_precision_score, roc_auc_score,
    precision_score, recall_score, f1_score, confusion_matrix, brier_score_loss
)
from sklearn.isotonic import IsotonicRegression
from sklearn.inspection import permutation_importance
import joblib
import json

from features import load_features, TARGET

OUT = "C:/Numair/Coding/Razorpay/outputs"

df, feature_cols = load_features()
df = df.sort_values("order_date").reset_index(drop=True)

t1 = df["order_date"].quantile(8 / 12)
t2 = df["order_date"].quantile(10 / 12)

train = df[df["order_date"] < t1]
val = df[(df["order_date"] >= t1) & (df["order_date"] < t2)]
test = df[df["order_date"] >= t2]

print(f"Train: {len(train):,} orders ({train['order_date'].min().date()} to {train['order_date'].max().date()})")
print(f"Val:   {len(val):,} orders ({val['order_date'].min().date()} to {val['order_date'].max().date()})")
print(f"Test:  {len(test):,} orders ({test['order_date'].min().date()} to {test['order_date'].max().date()})")
print(f"Return rate -> train {train[TARGET].mean():.2%} | val {val[TARGET].mean():.2%} | test {test[TARGET].mean():.2%}")

Xtr, ytr = train[feature_cols], train[TARGET]
Xval, yval = val[feature_cols], val[TARGET]
Xtest, ytest = test[feature_cols], test[TARGET]

# ---------------- Baseline: rule-of-thumb (flag all COD apparel orders) ----
baseline_flag_val = ((val["payment_mode_COD"] == True) & (val["is_apparel"] == 1)).astype(int)
baseline_precision = precision_score(yval, baseline_flag_val)
baseline_recall = recall_score(yval, baseline_flag_val)
print(f"\n[Rule baseline: flag all COD+apparel] val precision={baseline_precision:.3f} recall={baseline_recall:.3f}")

# ---------------- Model 1: Logistic Regression (interpretable) -------------
logreg = LogisticRegression(max_iter=1000, class_weight="balanced")
logreg.fit(Xtr, ytr)
val_proba_lr = logreg.predict_proba(Xval)[:, 1]
print(f"[LogReg] val PR-AUC={average_precision_score(yval, val_proba_lr):.3f} "
      f"ROC-AUC={roc_auc_score(yval, val_proba_lr):.3f}")

# ---------------- Model 2: HistGradientBoosting (PRIMARY model) ------------
hgb = HistGradientBoostingClassifier(
    max_iter=300, learning_rate=0.06, max_depth=5,
    class_weight="balanced", random_state=42, early_stopping=True,
    validation_fraction=0.15,
)
hgb.fit(Xtr, ytr)
val_proba_hgb = hgb.predict_proba(Xval)[:, 1]
print(f"[HGB]    val PR-AUC={average_precision_score(yval, val_proba_hgb):.3f} "
      f"ROC-AUC={roc_auc_score(yval, val_proba_hgb):.3f}")

# ---------------- Model 3: XGBoost (comparison only) ----
scale_pos_weight = (ytr == 0).sum() / (ytr == 1).sum()
xgb = XGBClassifier(
    n_estimators=300, learning_rate=0.06, max_depth=5,
    scale_pos_weight=scale_pos_weight, random_state=42,
    eval_metric="aucpr", n_jobs=-1,
)
xgb.fit(Xtr, ytr)
val_proba_xgb = xgb.predict_proba(Xval)[:, 1]
print(f"[XGB]    val PR-AUC={average_precision_score(yval, val_proba_xgb):.3f} "
      f"ROC-AUC={roc_auc_score(yval, val_proba_xgb):.3f}")

# ---------------- Cost-sensitive threshold selection (on VAL only) ---------
# Cost assumptions (documented, editable):
#   FN (missed return, order ships normally) -> cost = avg reverse-logistics
#       + restocking loss if the return does happen anyway later ~ INR 180
#   FP (order wrongly flagged -> extra verification/OTP/address-confirmation
#       step for a customer who would NOT have returned) -> cost = customer
#       friction / support-team review cost ~ INR 25
COST_FN = 180.0
COST_FP = 25.0

thresholds = np.linspace(0.05, 0.95, 181)
costs = []
for t in thresholds:
    pred = (val_proba_hgb >= t).astype(int)
    fp = ((pred == 1) & (yval.values == 0)).sum()
    fn = ((pred == 0) & (yval.values == 1)).sum()
    total_cost = fp * COST_FP + fn * COST_FN
    costs.append(total_cost)
costs = np.array(costs)
best_idx = costs.argmin()
best_threshold = thresholds[best_idx]

# Baseline "flag nothing" cost, for savings comparison
cost_flag_none = (yval.values == 1).sum() * COST_FN
cost_flag_all = (yval.values == 0).sum() * COST_FP
print(f"\nCost-optimal threshold (chosen on VAL, cost FN=₹{COST_FN} FP=₹{COST_FP}): {best_threshold:.3f}")
print(f"  Val cost at that threshold: ₹{costs[best_idx]:,.0f}  "
      f"vs flag-none ₹{cost_flag_none:,.0f}  vs flag-all ₹{cost_flag_all:,.0f}")

# ---------------- FROZEN, single-shot evaluation on TEST -------------------
test_proba = hgb.predict_proba(Xtest)[:, 1]
test_pred = (test_proba >= best_threshold).astype(int)

test_precision = precision_score(ytest, test_pred)
test_recall = recall_score(ytest, test_pred)
test_f1 = f1_score(ytest, test_pred)
test_prauc = average_precision_score(ytest, test_proba)
test_rocauc = roc_auc_score(ytest, test_proba)
tn, fp, fn, tp = confusion_matrix(ytest, test_pred).ravel()
test_cost = fp * COST_FP + fn * COST_FN
test_cost_flag_none = (ytest.values == 1).sum() * COST_FN
savings = test_cost_flag_none - test_cost

print("\n================ FINAL HELD-OUT TEST METRICS (touched once) ================")
print(f"n_test={len(ytest):,}  positive_rate={ytest.mean():.2%}")
print(f"Precision: {test_precision:.3f}")
print(f"Recall:    {test_recall:.3f}")
print(f"F1:        {test_f1:.3f}")
print(f"PR-AUC:    {test_prauc:.3f}")
print(f"ROC-AUC:   {test_rocauc:.3f}")
print(f"Confusion matrix: TN={tn} FP={fp} FN={fn} TP={tp}")
print(f"Cost at chosen threshold: ₹{test_cost:,.0f}")
print(f"Cost if flagging nothing: ₹{test_cost_flag_none:,.0f}")
print(f"Estimated cost saved by model: ₹{savings:,.0f} "
      f"({savings/test_cost_flag_none:.1%} reduction)")

# Compare against the naive rule baseline on the SAME test set
baseline_flag_test = ((test["payment_mode_COD"] == True) & (test["is_apparel"] == 1)).astype(int)
bp = precision_score(ytest, baseline_flag_test)
br = recall_score(ytest, baseline_flag_test)
bcost = (((baseline_flag_test == 1) & (ytest.values == 0)).sum() * COST_FP +
         ((baseline_flag_test == 0) & (ytest.values == 1)).sum() * COST_FN)
print(f"\n[Rule baseline on TEST] precision={bp:.3f} recall={br:.3f} cost=₹{bcost:,.0f}")

xgb_test_proba = xgb.predict_proba(Xtest)[:, 1]
xgb_test_prauc = average_precision_score(ytest, xgb_test_proba)
xgb_test_rocauc = roc_auc_score(ytest, xgb_test_proba)
#print(f"\n[XGB comparison on TEST] PR-AUC={xgb_test_prauc:.3f} ROC-AUC={xgb_test_rocauc:.3f}")

print(f"\n[XGB comparison on TEST] PR-AUC={xgb_test_prauc:.3f} ROC-AUC={xgb_test_rocauc:.3f}")

# ---------------- Calibration (isotonic, fit on VAL) ------------------------
# Recreates the calibration step your project notes describe but that has
# no corresponding file in this codebase -- see chat for why this is being
# added now rather than assumed already done.
calibrator = IsotonicRegression(out_of_bounds="clip")
calibrator.fit(val_proba_hgb, yval)
test_proba_calibrated = calibrator.predict(test_proba)
brier_raw = brier_score_loss(ytest, test_proba)
brier_cal = brier_score_loss(ytest, test_proba_calibrated)
print(f"\nBrier score (raw):        {brier_raw:.4f}")
print(f"Brier score (calibrated): {brier_cal:.4f}")

# ---------------- Reason codes: importance-ranked, direction-aware ---------
# Global permutation importance = WHICH features matter most overall.
# Mean feature value among returned vs. non-returned TRAIN orders = WHICH
# DIRECTION each feature pushes risk. Serving time then checks a given
# order's own values against that direction -- coarse, FICO-style reason
# codes, not a raw model-internals dump. See serve.py for how this is used.
perm_result = permutation_importance(
    hgb, Xval, yval, scoring="average_precision", n_repeats=10, random_state=42
)
importances = perm_result.importances_mean

reason_reference = {}
for i, feat in enumerate(feature_cols):
    mean_returned = Xtr.loc[ytr == 1, feat].mean()
    mean_not_returned = Xtr.loc[ytr == 0, feat].mean()
    direction = "high" if mean_returned >= mean_not_returned else "low"
    reason_reference[feat] = {
        "importance": float(importances[i]),
        "direction": direction,
        "mean_returned": float(mean_returned),
        "mean_not_returned": float(mean_not_returned),
    }

with open("outputs/reason_code_reference.json", "w") as f:
    json.dump(reason_reference, f, indent=2)
with open("outputs/feature_columns.json", "w") as f:
    json.dump(feature_cols, f, indent=2)
joblib.dump(calibrator, "outputs/calibrator.joblib")
print("Saved calibrator.joblib, reason_code_reference.json, feature_columns.json to outputs/")

# ---------------- Save everything for the report / artifact ----------------
joblib.dump(hgb, f"{OUT}/model.joblib")
np.save(f"{OUT}/val_proba_hgb.npy", val_proba_hgb)
np.save(f"{OUT}/val_y.npy", yval.values)
np.save(f"{OUT}/test_proba.npy", test_proba)
np.save(f"{OUT}/test_y.npy", ytest.values)

summary = {
    "best_threshold": best_threshold,
    "test_precision": test_precision, "test_recall": test_recall,
    "test_f1": test_f1, "test_prauc": test_prauc, "test_rocauc": test_rocauc,
    "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    "test_cost": test_cost, "test_cost_flag_none": test_cost_flag_none,
    "savings": savings, "savings_pct": savings / test_cost_flag_none,
    "baseline_precision": bp, "baseline_recall": br, "baseline_cost": bcost,
    "brier_raw": brier_raw, "brier_calibrated": brier_cal,
    "cost_fn": COST_FN, "cost_fp": COST_FP,
}
pd.Series(summary).to_json(f"{OUT}/summary.json", indent=2)
print("\nSaved model + arrays + summary.json to outputs/")