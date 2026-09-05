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

from pathlib import Path
OUT = str(Path(__file__).resolve().parent / "outputs")
Path(OUT).mkdir(exist_ok=True)

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
# Drops one reference dummy per categorical group for interpretable
# coefficients. Scoped to LogReg only -- HGB/XGBoost use the full dummy set.
logreg_ref_cols = {"payment_mode_COD", "category_Grocery", "pincode_tier_Tier1"}
logreg_feature_cols = [c for c in feature_cols if c not in logreg_ref_cols]

logreg = LogisticRegression(max_iter=1000, class_weight="balanced")
logreg.fit(Xtr[logreg_feature_cols], ytr)
val_proba_lr = logreg.predict_proba(Xval[logreg_feature_cols])[:, 1]
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

# ---------------- Baseline HGB kept as a documented reference, NOT shipped ---
hgb_baseline_ref = hgb
val_proba_hgb_baseline_ref = val_proba_hgb

# ---------------- PRODUCTION MODEL: damped segment-weighted HGB -----------
# See README Section 2 [methodology] for why this replaces the baseline above.
# From here on, `hgb` / `val_proba_hgb` refer to THIS model -- every
# downstream step operates on the production model, not the baseline.
train_mode = train["payment_mode_Prepaid"].map({True: "Prepaid", False: "COD"})
bucket_counts = train_mode.astype(str).str.cat(ytr.astype(str), sep="_").value_counts()
n_buckets = 4
sample_weight_full = np.ones(len(ytr))
for i in range(len(ytr)):
    key = f"{train_mode.iloc[i]}_{ytr.iloc[i]}"
    sample_weight_full[i] = len(ytr) / (n_buckets * bucket_counts[key])
sample_weight_damped = sample_weight_full ** 0.3

hgb = HistGradientBoostingClassifier(
    max_iter=300, learning_rate=0.06, max_depth=5,
    random_state=42, early_stopping=True, validation_fraction=0.15,
    # NOTE: no class_weight="balanced" -- sample_weight_damped replaces it.
)
hgb.fit(Xtr, ytr, sample_weight=sample_weight_damped)
val_proba_hgb = hgb.predict_proba(Xval)[:, 1]
print(f"[HGB, PRODUCTION damped-weighted] val PR-AUC={average_precision_score(yval, val_proba_hgb):.3f} "
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
# Cost derivation: see README Section 3.6 [cost-assumptions].
COST_FN = 180.0
COST_FP = 25.0

# ---------------- Calibrate BEFORE threshold selection ---------------------
# Do NOT move this after threshold search -- see README Section 2
# [methodology] for why that ordering was a bug (fixed here).
calibrator = IsotonicRegression(out_of_bounds="clip")
calibrator.fit(val_proba_hgb, yval)
val_proba_hgb_calibrated = calibrator.predict(val_proba_hgb)

# Closed-form cost-optimal threshold (zero sampling variance) -- used as a
# stability check against the grid search below. See README Section 2.
closed_form_threshold = COST_FP / (COST_FP + COST_FN)
print(f"\n[Closed-form cost-optimal threshold] p = {closed_form_threshold:.4f}")

# Selection criterion: MACRO-averaged cost-per-order, not blended total --
# blended cost favors the majority segment (COD) by volume. See README
# Section 2 [methodology] and experiments/prepaid_gap_investigation.py.
val_cod_mask = (val["payment_mode_COD"] == True).values
val_ppd_mask = (val["payment_mode_Prepaid"] == True).values

thresholds = np.linspace(0.05, 0.95, 181)
blended_costs = []
macro_costs = []
for t in thresholds:
    pred = (val_proba_hgb_calibrated >= t).astype(int)

    # Blended (kept + reported for transparency/comparison, NOT used to select)
    fp_all = ((pred == 1) & (yval.values == 0)).sum()
    fn_all = ((pred == 0) & (yval.values == 1)).sum()
    blended_costs.append(fp_all * COST_FP + fn_all * COST_FN)

    # Macro: cost-per-order within each segment, averaged across segments
    def cost_per_order(mask):
        fp = ((pred == 1) & (yval.values == 0) & mask).sum()
        fn = ((pred == 0) & (yval.values == 1) & mask).sum()
        n = mask.sum()
        return (fp * COST_FP + fn * COST_FN) / n if n > 0 else np.nan

    cod_cpo = cost_per_order(val_cod_mask)
    ppd_cpo = cost_per_order(val_ppd_mask)
    macro_costs.append((cod_cpo + ppd_cpo) / 2)

blended_costs = np.array(blended_costs)
macro_costs = np.array(macro_costs)

# PRIMARY selection: lowest macro-cost
best_idx = macro_costs.argmin()
best_threshold = thresholds[best_idx]
print(f"  [Compare] macro-searched threshold (calibrated scale): {best_threshold:.4f}  "
      f"vs closed-form: {closed_form_threshold:.4f}  "
      f"(diff: {abs(best_threshold - closed_form_threshold):.4f})")

# What blended-cost selection WOULD have chosen, for direct comparison
blended_best_idx = blended_costs.argmin()
blended_best_threshold = thresholds[blended_best_idx]

cost_flag_none = (yval.values == 1).sum() * COST_FN
cost_flag_all = (yval.values == 0).sum() * COST_FP
print(f"\nMACRO-cost-optimal threshold (chosen on VAL, cost FN=₹{COST_FN} FP=₹{COST_FP}): {best_threshold:.3f}")
print(f"  VAL macro cost/order at that threshold: ₹{macro_costs[best_idx]:.2f}")
print(f"  (for comparison) BLENDED-cost-optimal threshold would have been: {blended_best_threshold:.3f}")
print(f"  Val blended cost at MACRO-selected threshold: ₹{blended_costs[best_idx]:,.0f}  "
      f"vs flag-none ₹{cost_flag_none:,.0f}  vs flag-all ₹{cost_flag_all:,.0f}")
# ---------------- FROZEN, single-shot evaluation on TEST -------------------
test_proba = hgb.predict_proba(Xtest)[:, 1]
test_proba_calibrated = calibrator.predict(test_proba)
test_pred = (test_proba_calibrated >= best_threshold).astype(int)

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
print(f"\n[XGB comparison on TEST] PR-AUC={xgb_test_prauc:.3f} ROC-AUC={xgb_test_rocauc:.3f}")

# ---------------- Item 6: MCC added to standard metrics ----------------
from sklearn.metrics import matthews_corrcoef
test_mcc = matthews_corrcoef(ytest, test_pred)
print(f"MCC:       {test_mcc:.3f}")

# ---------------- Brier score: raw vs. calibrated --------------------------
# Calibrator was already fit above (before threshold selection); this only
# reports the improvement, it does not fit anything new.
brier_raw = brier_score_loss(ytest, test_proba)
brier_cal = brier_score_loss(ytest, test_proba_calibrated)
print(f"\nBrier score (raw):        {brier_raw:.4f}")
print(f"Brier score (calibrated): {brier_cal:.4f}")

# ---------------- Reason codes: importance-ranked, direction-aware ---------
# Coarse, FICO-style reason codes, not a raw model-internals dump.
# See README Section 2 [methodology] and serve.py for how this is used.
perm_result = permutation_importance(
    hgb, Xval, yval, scoring="average_precision", n_repeats=10, random_state=42
)
importances = perm_result.importances_mean

reason_reference = {}

# Direction/midpoint computed as MACRO-averaged conditional means (mean
# across COD and Prepaid segments, then averaged) rather than one pooled
# marginal split. A pooled split is confounded by payment_mode for any
# feature that interacts with it: is_bracketed only exists in Prepaid
# apparel orders, but Prepaid's much lower baseline return rate swamps
# the real within-segment effect in a pooled comparison (Simpson's
# paradox) -- this silently flipped is_bracketed's stored direction to
# "low", so a NON-bracketed order (value=0) was being labeled as elevated
# risk. Same macro-not-blended principle already used for cost-optimal
# threshold selection, applied here for consistency. See README Section 5
# for the bug this fixes.
#
# EXCEPTION: payment_mode_COD / payment_mode_Prepaid keep the original
# pooled calculation -- segmenting by payment_mode and then asking for
# payment_mode's own conditional mean within that segment is tautological
# (every COD row has payment_mode_COD=1 by definition), so macro-averaging
# those two specifically would produce a degenerate 0.5/0.5 split.
SEGMENT_MASKS = [
    (Xtr["payment_mode_COD"] == True).values,
    (Xtr["payment_mode_Prepaid"] == True).values,
]


def macro_conditional_means(feat):
    means_ret, means_not = [], []
    for mask in SEGMENT_MASKS:
        seg_y = ytr.values[mask]
        seg_x = Xtr.loc[mask, feat].values
        if (seg_y == 1).sum() > 0:
            means_ret.append(seg_x[seg_y == 1].mean())
        if (seg_y == 0).sum() > 0:
            means_not.append(seg_x[seg_y == 0].mean())
    return sum(means_ret) / len(means_ret), sum(means_not) / len(means_not)


for i, feat in enumerate(feature_cols):
    if feat in ("payment_mode_COD", "payment_mode_Prepaid"):
        mean_returned = Xtr.loc[ytr == 1, feat].mean()
        mean_not_returned = Xtr.loc[ytr == 0, feat].mean()
    else:
        mean_returned, mean_not_returned = macro_conditional_means(feat)
    direction = "high" if mean_returned >= mean_not_returned else "low"
    reason_reference[feat] = {
        "importance": float(importances[i]),
        "direction": direction,
        "mean_returned": float(mean_returned),
        "mean_not_returned": float(mean_not_returned),
    }

with open(f"{OUT}/reason_code_reference.json", "w") as f:
    json.dump(reason_reference, f, indent=2)
with open(f"{OUT}/feature_columns.json", "w") as f:
    json.dump(feature_cols, f, indent=2)
joblib.dump(calibrator, f"{OUT}/calibrator.joblib")
print("Saved calibrator.joblib, reason_code_reference.json, feature_columns.json to outputs/")

# ---------------- Save everything for the report / artifact ----------------
joblib.dump(hgb, f"{OUT}/model.joblib")
np.save(f"{OUT}/val_proba_hgb.npy", val_proba_hgb)
np.save(f"{OUT}/val_y.npy", yval.values)
np.save(f"{OUT}/test_proba.npy", test_proba)
np.save(f"{OUT}/test_y.npy", ytest.values)

# ---------------- Baseline reference: same discipline, for comparison ------
val_cod_mask_b = val_cod_mask  # same masks, reused
val_ppd_mask_b = val_ppd_mask

def _macro_cost_threshold(proba, y_true, cod_mask, ppd_mask, cost_fn=COST_FN, cost_fp=COST_FP):
    ths = np.linspace(0.05, 0.95, 181)
    best_t, best_macro = None, np.inf
    for t in ths:
        pred = (proba >= t).astype(int)
        def cpo(mask):
            fp_ = ((pred == 1) & (y_true == 0) & mask).sum()
            fn_ = ((pred == 0) & (y_true == 1) & mask).sum()
            n_ = mask.sum()
            return (fp_ * cost_fp + fn_ * cost_fn) / n_ if n_ > 0 else np.nan
        macro = (cpo(cod_mask) + cpo(ppd_mask)) / 2
        if macro < best_macro:
            best_macro, best_t = macro, t
    return best_t, best_macro

t_baseline, _ = _macro_cost_threshold(val_proba_hgb_baseline_ref, yval.values, val_cod_mask_b, val_ppd_mask_b)
test_proba_baseline_ref = hgb_baseline_ref.predict_proba(Xtest)[:, 1]
test_pred_baseline_ref = (test_proba_baseline_ref >= t_baseline).astype(int)

baseline_ref_summary = {
    "note": "Unweighted HGB (class_weight='balanced'). NOT the shipped model -- kept as documented reference. See prepaid-gap investigation for why production uses damped segment-weighting instead.",
    "threshold": t_baseline,
    "test_precision": precision_score(ytest, test_pred_baseline_ref),
    "test_recall": recall_score(ytest, test_pred_baseline_ref),
    "test_f1": f1_score(ytest, test_pred_baseline_ref),
    "test_prauc": average_precision_score(ytest, test_proba_baseline_ref),
    "test_mcc": matthews_corrcoef(ytest, test_pred_baseline_ref),
}
tn_b, fp_b, fn_b, tp_b = confusion_matrix(ytest, test_pred_baseline_ref).ravel()
baseline_ref_summary["test_cost"] = fp_b * COST_FP + fn_b * COST_FN
baseline_ref_summary["tn"], baseline_ref_summary["fp"] = int(tn_b), int(fp_b)
baseline_ref_summary["fn"], baseline_ref_summary["tp"] = int(fn_b), int(tp_b)

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
    "test_mcc": test_mcc,
    "baseline_reference": baseline_ref_summary,
}
pd.Series(summary).to_json(f"{OUT}/summary.json", indent=2)
print("\nSaved model + arrays + summary.json to outputs/")

# Diagnostic decomposition of the ALREADY-FROZEN test_pred -- not a new
# TEST touch, just breaking down the single result already computed above.
test_ppd_mask_final = (test["payment_mode_Prepaid"] == True).values
fn_mask_final = (test_pred == 0) & (ytest.values == 1)
fn_prepaid_count_final = (test_ppd_mask_final & fn_mask_final).sum()
fn_prepaid_share_final = fn_prepaid_count_final / fn_mask_final.sum()
print(f"\nProduction model FN breakdown: {int(fn_mask_final.sum())} total FN, "
      f"{int(fn_prepaid_count_final)} prepaid ({fn_prepaid_share_final:.1%})")

# Recall by segment, not FN share -- FN share can mislead when the FN count
# itself changes. See README Section 5 [what-we-tried-and-rejected].
tp_mask_final = (test_pred == 1) & (ytest.values == 1)
cod_mask_final = ~test_ppd_mask_final

ppd_positives = (test_ppd_mask_final & (ytest.values == 1)).sum()
cod_positives = (cod_mask_final & (ytest.values == 1)).sum()
ppd_tp = (test_ppd_mask_final & tp_mask_final).sum()
cod_tp = (cod_mask_final & tp_mask_final).sum()

ppd_recall = ppd_tp / ppd_positives if ppd_positives > 0 else float("nan")
cod_recall = cod_tp / cod_positives if cod_positives > 0 else float("nan")
print(f"Production model recall by segment: "
      f"Prepaid={ppd_recall:.3f} ({ppd_tp}/{ppd_positives})  "
      f"COD={cod_recall:.3f} ({cod_tp}/{cod_positives})")

# Same weighting, different architecture -- tests whether the prepaid gap
# is architecture-agnostic. See README Section 6 [known-limitations].
xgb_damped = XGBClassifier(
    n_estimators=300, learning_rate=0.06, max_depth=5,
    random_state=42, eval_metric="aucpr", n_jobs=-1,
    # No scale_pos_weight -- sample_weight_damped already corrects; combining
    # both would double-correct (same reasoning as the HGB version above).
)
xgb_damped.fit(Xtr, ytr, sample_weight=sample_weight_damped)
val_proba_xgb_damped = xgb_damped.predict_proba(Xval)[:, 1]
print(f"[XGB damped-weighted] val PR-AUC={average_precision_score(yval, val_proba_xgb_damped):.3f} "
      f"ROC-AUC={roc_auc_score(yval, val_proba_xgb_damped):.3f}")

t_xgb_damped, _ = _macro_cost_threshold(val_proba_xgb_damped, yval.values, val_cod_mask, val_ppd_mask)
test_proba_xgb_damped = xgb_damped.predict_proba(Xtest)[:, 1]
test_pred_xgb_damped = (test_proba_xgb_damped >= t_xgb_damped).astype(int)

fn_mask_xgbd = (test_pred_xgb_damped == 0) & (ytest.values == 1)
fn_prepaid_xgbd = (test_ppd_mask_final & fn_mask_xgbd).sum()
tn_x, fp_x, fn_x, tp_x = confusion_matrix(ytest, test_pred_xgb_damped).ravel()
cost_x = fp_x * COST_FP + fn_x * COST_FN

print(f"\n[XGB damped, frozen TEST] total FN={int(fn_mask_xgbd.sum())}  "
      f"prepaid FN={int(fn_prepaid_xgbd)} ({fn_prepaid_xgbd/fn_mask_xgbd.sum():.1%})  "
      f"cost=₹{cost_x:,.0f}  PR-AUC={average_precision_score(ytest, test_proba_xgb_damped):.3f}  "
      f"MCC={matthews_corrcoef(ytest, test_pred_xgb_damped):.3f}")

tp_mask_xgbd = (test_pred_xgb_damped == 1) & (ytest.values == 1)
ppd_tp_xgbd = (test_ppd_mask_final & tp_mask_xgbd).sum()
cod_tp_xgbd = (cod_mask_final & tp_mask_xgbd).sum()
ppd_recall_xgbd = ppd_tp_xgbd / ppd_positives if ppd_positives > 0 else float("nan")
cod_recall_xgbd = cod_tp_xgbd / cod_positives if cod_positives > 0 else float("nan")
print(f"[XGB damped] recall by segment: "
      f"Prepaid={ppd_recall_xgbd:.3f} ({ppd_tp_xgbd}/{ppd_positives})  "
      f"COD={cod_recall_xgbd:.3f} ({cod_tp_xgbd}/{cod_positives})")


def segment_optimal_threshold(proba, y_true, mask, cost_fn=COST_FN, cost_fp=COST_FP):
    ths = np.linspace(0.05, 0.95, 181)
    best_t, best_cost = None, np.inf
    proba_seg, y_seg = proba[mask], y_true[mask]
    for t in ths:
        pred = (proba_seg >= t).astype(int)
        fp = ((pred == 1) & (y_seg == 0)).sum()
        fn = ((pred == 0) & (y_seg == 1)).sum()
        c = (fp * cost_fp + fn * cost_fn) / mask.sum()
        if c < best_cost:
            best_cost, best_t = c, t
    return best_t, best_cost

t_cod_seg, cost_cod_seg = segment_optimal_threshold(val_proba_hgb_calibrated, yval.values, val_cod_mask)
t_ppd_seg, cost_ppd_seg = segment_optimal_threshold(val_proba_hgb_calibrated, yval.values, val_ppd_mask)
print(f"Segment-specific thresholds (calibrated): COD={t_cod_seg:.3f} (₹{cost_cod_seg:.2f}/order)  "
      f"Prepaid={t_ppd_seg:.3f} (₹{cost_ppd_seg:.2f}/order)")

test_cod_mask = (test["payment_mode_COD"] == True).values
test_ppd_mask = (test["payment_mode_Prepaid"] == True).values
test_pred_seg = np.where(test_cod_mask,
                          (test_proba_calibrated >= t_cod_seg).astype(int),
                          (test_proba_calibrated >= t_ppd_seg).astype(int))

tn_s, fp_s, fn_s, tp_s = confusion_matrix(ytest, test_pred_seg).ravel()
cost_s = fp_s * COST_FP + fn_s * COST_FN
ppd_tp_s = (test_ppd_mask & (test_pred_seg == 1) & (ytest.values == 1)).sum()
ppd_pos = (test_ppd_mask & (ytest.values == 1)).sum()
cod_tp_s = (test_cod_mask & (test_pred_seg == 1) & (ytest.values == 1)).sum()
cod_pos = (test_cod_mask & (ytest.values == 1)).sum()
print(f"[Segment-threshold, frozen TEST] TN={tn_s} FP={fp_s} FN={fn_s} TP={tp_s}  cost=₹{cost_s:,.0f}")
print(f"Recall: Prepaid={ppd_tp_s/ppd_pos:.3f} ({ppd_tp_s}/{ppd_pos})  COD={cod_tp_s/cod_pos:.3f} ({cod_tp_s}/{cod_pos})")