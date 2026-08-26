"""
train_fraud.py
===============
SECONDARY ARTIFACT — demonstrates the same evaluation methodology used in
the primary Return-Risk Scorer (temporal split, cost-sensitive threshold
chosen on VAL only, single frozen TEST evaluation, calibration) applied to
a genuinely REAL dataset, since no usable real return dataset was found.

DATA: ULB European credit-card-fraud dataset (Kaggle "creditcard.csv").
  - 284,807 real anonymized transactions over ~48 hours.
  - Features V1-V28 are PCA components (privacy-anonymized) -- NOT
    human-interpretable. This means, unlike the primary artifact:
      * NO reason codes are produced here. A permutation-importance list
        naming "V14" conveys nothing to a reviewer -- this is a stated
        limitation of the dataset, not an oversight.
      * NO customer/account identifier exists, so there is no equivalent
        of `customer_past_return_rate`. There is nothing here that could
        replicate the primary artifact's leakage bug (no per-entity
        history to leak). The only leakage risk in THIS artifact is
        ordinary train/test temporal contamination, guarded against the
        same way: strict time-ordered split, test touched once.

TIME SPAN CAVEAT: the dataset covers ~48 hours, not months. The temporal
split below is proportional (60/20/20 by Time percentile) rather than
calendar-based. This still prevents the model from training on the future,
but it is a much thinner test of temporal generalization than the primary
artifact's 8/2/2-month split -- state this plainly in any writeup, do not
imply a stronger claim than 2 days of data supports.

COST CAVEAT: FN/FP costs below are placeholders grounded loosely in this
dataset's own Amount field (mean fraudulent transaction ~$122), NOT
researched chargeback/investigation figures, and NOT in INR -- do not mix
these numbers with the primary artifact's Rs.180/Rs.25 cost model in any
combined report. They are illustrative only, exactly like the primary
artifact's costs.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    precision_recall_curve, average_precision_score, roc_auc_score,
    precision_score, recall_score, f1_score, confusion_matrix
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import brier_score_loss
import joblib

DATA_PATH = "C:/Numair/Coding/Razorpay/creditcard.csv"
OUT = "C:/Numair/Coding/Razorpay/outputs"

df = pd.read_csv(DATA_PATH).sort_values("Time").reset_index(drop=True)
feature_cols = [c for c in df.columns if c.startswith("V")] + ["Amount"]
TARGET = "Class"

# ---------------- Temporal split (proportional, not calendar-based) --------
t1 = df["Time"].quantile(0.60)
t2 = df["Time"].quantile(0.80)

train = df[df["Time"] < t1]
val = df[(df["Time"] >= t1) & (df["Time"] < t2)]
test = df[df["Time"] >= t2]

print(f"Train: {len(train):,} txns (0 to {t1/3600:.1f}h)  fraud_rate={train[TARGET].mean():.4%}")
print(f"Val:   {len(val):,} txns ({t1/3600:.1f}h to {t2/3600:.1f}h)  fraud_rate={val[TARGET].mean():.4%}")
print(f"Test:  {len(test):,} txns ({t2/3600:.1f}h to {df['Time'].max()/3600:.1f}h)  fraud_rate={test[TARGET].mean():.4%}")

Xtr, ytr = train[feature_cols], train[TARGET]
Xval, yval = val[feature_cols], val[TARGET]
Xtest, ytest = test[feature_cols], test[TARGET]

hgb_candidate = HistGradientBoostingClassifier(
    max_iter=300, learning_rate=0.06, max_depth=5,
    class_weight="balanced", random_state=42, early_stopping=True,
    validation_fraction=0.15,
)
hgb_candidate.fit(Xtr, ytr)
val_proba_hgb = hgb_candidate.predict_proba(Xval)[:, 1]
prauc_hgb = average_precision_score(yval, val_proba_hgb)
print(f"\n[HGB] val PR-AUC={prauc_hgb:.3f} ROC-AUC={roc_auc_score(yval, val_proba_hgb):.3f}")

scale_pos_weight = (ytr == 0).sum() / (ytr == 1).sum()
xgb_candidate = XGBClassifier(
    n_estimators=300, learning_rate=0.06, max_depth=5,
    scale_pos_weight=scale_pos_weight, random_state=42,
    eval_metric="aucpr", n_jobs=-1,
)
xgb_candidate.fit(Xtr, ytr)
val_proba_xgb = xgb_candidate.predict_proba(Xval)[:, 1]
prauc_xgb = average_precision_score(yval, val_proba_xgb)
print(f"[XGB] val PR-AUC={prauc_xgb:.3f} ROC-AUC={roc_auc_score(yval, val_proba_xgb):.3f}")

# Winner decided HERE, in this run, on VAL only -- not by comparing against
# a different script execution that may have used a different library
# version (see the sklearn/XGBoost reproducibility issue earlier in this
# project's history: cross-run comparisons are not apples-to-apples).
if prauc_xgb >= prauc_hgb:
    hgb, val_proba, primary_name = xgb_candidate, val_proba_xgb, "XGB"
else:
    hgb, val_proba, primary_name = hgb_candidate, val_proba_hgb, "HGB"
print(f"Primary model selected: {primary_name} (higher VAL PR-AUC)")

# ---------------- Cost-sensitive threshold selection (VAL only) ------------
# PLACEHOLDER costs -- not researched, not INR, not comparable to the
# primary artifact's Rs.180/Rs.25. Grounded loosely in this dataset's own
# mean fraud Amount (~$122) as a lower-bound proxy for FN cost; FP cost is
# an arbitrary small friction estimate analogous in spirit to the primary
# artifact's Rs.25 (not derived from any real decline-cost figure).
COST_FN = 120.0   # ~ mean fraudulent transaction amount in this dataset
COST_FP = 5.0     # placeholder friction cost of a false decline

thresholds = np.linspace(0.01, 0.99, 197)
costs = []
for t in thresholds:
    pred = (val_proba >= t).astype(int)
    fp = ((pred == 1) & (yval.values == 0)).sum()
    fn = ((pred == 0) & (yval.values == 1)).sum()
    costs.append(fp * COST_FP + fn * COST_FN)
costs = np.array(costs)
best_idx = costs.argmin()
best_threshold = thresholds[best_idx]

cost_flag_none = (yval.values == 1).sum() * COST_FN
print(f"\nCost-optimal threshold (VAL, FN=${COST_FN:.0f} FP=${COST_FP:.0f}): {best_threshold:.3f}")
print(f"  Val cost at that threshold: ${costs[best_idx]:,.0f}  vs flag-none ${cost_flag_none:,.0f}")

# ---------------- Calibration (isotonic, same choice as primary) -----------
# Fit on VAL, applied at inference -- mirrors calibrate.py's approach.
calibrated = CalibratedClassifierCV(FrozenEstimator(hgb), method="isotonic")
calibrated.fit(Xval, yval)

# ---------------- FROZEN, single-shot evaluation on TEST -------------------
test_proba = hgb.predict_proba(Xtest)[:, 1]
test_proba_cal = calibrated.predict_proba(Xtest)[:, 1]
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

print("\n================ FRAUD ARTIFACT — FROZEN TEST METRICS (touched once) ================")
print(f"n_test={len(ytest):,}  positive_rate={ytest.mean():.4%}")
print(f"Precision: {test_precision:.3f}")
print(f"Recall:    {test_recall:.3f}")
print(f"F1:        {test_f1:.3f}")
print(f"PR-AUC:    {test_prauc:.3f}")
print(f"ROC-AUC:   {test_rocauc:.3f}")
print(f"Confusion matrix: TN={tn} FP={fp} FN={fn} TP={tp}")
print(f"Cost at chosen threshold: ${test_cost:,.0f}")
print(f"Cost if flagging nothing: ${test_cost_flag_none:,.0f}")
print(f"Estimated cost saved: ${savings:,.0f} ({savings/test_cost_flag_none:.1%} reduction)")
brier_raw = brier_score_loss(ytest, test_proba)
brier_cal = brier_score_loss(ytest, test_proba_cal)
brier_ratio = brier_raw / brier_cal if brier_cal > 0 else float("inf")
print(f"\nBrier score (raw {primary_name}):        {brier_raw:.5f}")
print(f"Brier score (isotonic-calib): {brier_cal:.5f}")
print(f"Raw-to-calibrated ratio: {brier_ratio:.2f}x")
print("NOTE: absolute Brier values are naturally tiny at a 0.13% base rate")
print("(always predicting ~0.0013 would already score near-zero) -- the")
print("improvement RATIO is the meaningful number, not the absolute value.")
print("NOTE: calibration reliability diagram omitted for this artifact --")
print("with only 75 positives and this base rate, quantile-binned")
print("calibration curves collapse near the origin and are not informative.")
print("\nNOTE: No reason codes are generated for this artifact -- V1-V28 are")
print("PCA-anonymized and not human-interpretable. See module docstring.")

# ---------------- Save everything -------------------------------------------
joblib.dump(hgb, f"{OUT}/model_fraud.joblib")
np.save(f"{OUT}/val_proba_fraud.npy", val_proba)
np.save(f"{OUT}/val_y_fraud.npy", yval.values)
np.save(f"{OUT}/test_proba_fraud.npy", test_proba)
np.save(f"{OUT}/test_proba_fraud_calibrated.npy", test_proba_cal)
np.save(f"{OUT}/test_y_fraud.npy", ytest.values)

summary = {
    "dataset": "ULB creditcard.csv (real, not synthetic)",
    "time_span_hours": round(df["Time"].max() / 3600, 1),
    "split_method": "proportional 60/20/20 by Time percentile (no calendar structure available)",
    "best_threshold": best_threshold,
    "test_precision": test_precision, "test_recall": test_recall,
    "test_f1": test_f1, "test_prauc": test_prauc, "test_rocauc": test_rocauc,
    "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    "test_cost": test_cost, "test_cost_flag_none": test_cost_flag_none,
    "savings": savings, "savings_pct": savings / test_cost_flag_none,
    "cost_fn": COST_FN, "cost_fp": COST_FP,
    "primary_model": primary_name, "val_prauc_hgb": prauc_hgb, "val_prauc_xgb": prauc_xgb,
    "cost_currency_note": "USD-scale placeholders derived from this dataset's own Amount field, NOT INR, NOT comparable to primary artifact's Rs.180/Rs.25",
    "brier_raw": brier_raw, "brier_calibrated": brier_cal,
    "brier_note": f"absolute values are naturally tiny at 0.13% base rate; the {brier_ratio:.2f}x raw-to-calibrated ratio is the meaningful figure, not the absolute value",
    "reason_codes": "not produced -- features are PCA-anonymized, see module docstring",
}
pd.Series(summary).to_json(f"{OUT}/summary_fraud.json", indent=2)
print(f"\nSaved model + arrays + summary_fraud.json to {OUT}/")