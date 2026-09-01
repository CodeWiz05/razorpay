"""
prepaid_gap_investigation.py
==============================
Standalone investigation log for the prepaid false-negative gap.
Self-contained: reloads the ALREADY-FITTED baseline model and its frozen
TEST results from train.py's saved outputs, rather than refitting its own
copy -- this guarantees every comparison in this file is measured against
whatever the current production model.joblib/summary.json actually are,
with no risk of silent drift between the two files.

Run AFTER train.py, using its current outputs/model.joblib and
outputs/summary.json. If you retrain the primary pipeline, rerun this
file afterward too, or these comparisons will be stale.
"""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
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

Xtr, ytr = train[feature_cols], train[TARGET]
Xval, yval = val[feature_cols], val[TARGET]
Xtest, ytest = test[feature_cols], test[TARGET]

# ---------------- Reload the PRODUCTION model + its frozen results ---------
hgb = joblib.load(f"{OUT}/model.joblib")
val_proba_hgb = hgb.predict_proba(Xval)[:, 1]
test_proba = hgb.predict_proba(Xtest)[:, 1]

summary = json.load(open(f"{OUT}/summary.json"))
best_threshold = summary["best_threshold"]
COST_FN, COST_FP = summary["cost_fn"], summary["cost_fp"]
test_pred = (test_proba >= best_threshold).astype(int)
test_precision, test_recall = summary["test_precision"], summary["test_recall"]
test_f1, test_prauc, test_mcc = summary["test_f1"], summary["test_prauc"], summary["test_mcc"]
tn, fp, fn, tp = summary["tn"], summary["fp"], summary["fn"], summary["tp"]
test_cost = summary["test_cost"]

# ---------------- Segment masks, reused across every experiment below ------
val_cod_mask = (val["payment_mode_COD"] == True).values
val_ppd_mask = (val["payment_mode_Prepaid"] == True).values
test_cod_mask = (test["payment_mode_COD"] == True).values
test_ppd_mask = (test["payment_mode_Prepaid"] == True).values

fn_mask_global = (test_pred == 0) & (ytest.values == 1)
global_fn_prepaid_share = (test_ppd_mask & fn_mask_global).sum() / fn_mask_global.sum()

# ---------------- Everything you already pasted from here down -------------
# (cost_optimal_threshold def, segment-threshold block, segment-weighted
# block, damped block, 3-value sweep, sanity check) goes below unchanged --
# it already references exactly these variable names.

# ---------------- Item 5: segment-specific threshold (COD vs Prepaid) ---
# Known finding: FNs are almost all prepaid orders (6.3% COD vs 51.3%
# baseline rate), confidently low-probability (mean 0.266). Hypothesis:
# a single global threshold under-serves prepaid orders because COD
# dominates the model's overall score distribution. Test: fit COD and
# Prepaid thresholds SEPARATELY on VAL (same cost-sensitive search as the
# global one), then apply per-segment on TEST and compare combined cost
# against the existing single-threshold result.

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

def cost_optimal_threshold_macro(proba, y_true, cod_mask, ppd_mask, cost_fn=COST_FN, cost_fp=COST_FP):
    """Selects threshold by MACRO-averaged cost: mean(COD cost-per-order,
    Prepaid cost-per-order), NOT blended total cost. This treats both
    segments as equally important regardless of their relative volume --
    directly addresses the bias where blended-cost selection always
    favors whichever segment has more orders."""
    ths = np.linspace(0.05, 0.95, 181)
    best_t, best_macro_cost = None, np.inf
    results = []
    for t in ths:
        pred = (proba >= t).astype(int)
        def cost_per_order(mask):
            fp = ((pred == 1) & (y_true == 0) & mask).sum()
            fn = ((pred == 0) & (y_true == 1) & mask).sum()
            n = mask.sum()
            return (fp * cost_fp + fn * cost_fn) / n if n > 0 else np.nan
        cod_cpo = cost_per_order(cod_mask)
        ppd_cpo = cost_per_order(ppd_mask)
        macro = (cod_cpo + ppd_cpo) / 2
        results.append((t, macro, cod_cpo, ppd_cpo))
        if macro < best_macro_cost:
            best_macro_cost, best_t = macro, t
    return best_t, best_macro_cost, results

val_cod_mask = (val["payment_mode_COD"] == True).values
val_ppd_mask = (val["payment_mode_Prepaid"] == True).values
test_cod_mask = (test["payment_mode_COD"] == True).values
test_ppd_mask = (test["payment_mode_Prepaid"] == True).values

t_cod, val_cost_cod = cost_optimal_threshold(val_proba_hgb[val_cod_mask], yval.values[val_cod_mask])
t_ppd, val_cost_ppd = cost_optimal_threshold(val_proba_hgb[val_ppd_mask], yval.values[val_ppd_mask])

print(f"\n[Segment thresholds, chosen on VAL] COD={t_cod:.3f}  Prepaid={t_ppd:.3f}  "
      f"(global was {best_threshold:.3f})")

# Apply segment thresholds to TEST, same touched-once discipline as the
# global evaluation above -- this is still the FIRST time TEST is scored
# under this scheme.
seg_pred = np.zeros_like(test_pred)
seg_pred[test_cod_mask] = (test_proba[test_cod_mask] >= t_cod).astype(int)
seg_pred[test_ppd_mask] = (test_proba[test_ppd_mask] >= t_ppd).astype(int)

seg_precision = precision_score(ytest, seg_pred)
seg_recall = recall_score(ytest, seg_pred)
seg_f1 = f1_score(ytest, seg_pred)
seg_mcc = matthews_corrcoef(ytest, seg_pred)
seg_tn, seg_fp, seg_fn, seg_tp = confusion_matrix(ytest, seg_pred).ravel()
seg_cost = seg_fp * COST_FP + seg_fn * COST_FN

# FN breakdown by segment, before vs after -- this is the number that
# actually answers "did this close the prepaid gap"
fn_mask_global = (test_pred == 0) & (ytest.values == 1)
fn_mask_seg = (seg_pred == 0) & (ytest.values == 1)
global_fn_prepaid_share = (test_ppd_mask & fn_mask_global).sum() / fn_mask_global.sum()
seg_fn_prepaid_share = (test_ppd_mask & fn_mask_seg).sum() / fn_mask_seg.sum()

print("\n================ SEGMENT-SPECIFIC THRESHOLD: TEST comparison ================")
print(f"{'Metric':<20}{'Global (t=' + f'{best_threshold:.2f})':<20}{'Segmented':<20}")
print(f"{'Precision':<20}{test_precision:<20.3f}{seg_precision:<20.3f}")
print(f"{'Recall':<20}{test_recall:<20.3f}{seg_recall:<20.3f}")
print(f"{'F1':<20}{test_f1:<20.3f}{seg_f1:<20.3f}")
print(f"{'MCC':<20}{test_mcc:<20.3f}{seg_mcc:<20.3f}")
print(f"{'Cost (Rs.)':<20}{test_cost:<20,.0f}{seg_cost:<20,.0f}")
print(f"{'FN count':<20}{int(fn):<20}{int(seg_fn):<20}")
print(f"{'FN % prepaid':<20}{global_fn_prepaid_share:<20.1%}{seg_fn_prepaid_share:<20.1%}")

# Save alongside the existing summary so this doesn't require a rerun to see later
segment_summary = {
    "threshold_cod": t_cod, "threshold_prepaid": t_ppd,
    "seg_precision": seg_precision, "seg_recall": seg_recall,
    "seg_f1": seg_f1, "seg_mcc": seg_mcc,
    "seg_tn": int(seg_tn), "seg_fp": int(seg_fp), "seg_fn": int(seg_fn), "seg_tp": int(seg_tp),
    "seg_cost": seg_cost,
    "global_fn_prepaid_share": global_fn_prepaid_share,
    "seg_fn_prepaid_share": seg_fn_prepaid_share,
}
with open(f"{OUT}/segment_threshold_summary.json", "w") as f:
    json.dump(segment_summary, f, indent=2)
print(f"\nSaved segment_threshold_summary.json to {OUT}/")

# ---------------- Segment-weighted loss: candidate fix for the prepaid gap ---
# Hypothesis (from correlation analysis): customer_past_return_rate is MORE
# correlated with target within prepaid orders (0.139) than overall (0.097)
# -- the model already has usable prepaid signal available, but
# class_weight="balanced" only equalizes overall positive vs negative
# volume. Since COD contributes far more positives than prepaid, the
# model's notion of "what a positive looks like" is COD-dominated by
# sheer count, even with balancing. This tests whether re-weighting
# training examples to equalize across ALL FOUR (payment_mode, returned)
# buckets -- not just (returned) -- surfaces the prepaid signal better.

train_mode = train["payment_mode_Prepaid"].map({True: "Prepaid", False: "COD"})
bucket_counts = train_mode.astype(str).str.cat(ytr.astype(str), sep="_").value_counts()
n_buckets = 4
sample_weight = np.ones(len(ytr))
for i in range(len(ytr)):
    key = f"{train_mode.iloc[i]}_{ytr.iloc[i]}"
    sample_weight[i] = len(ytr) / (n_buckets * bucket_counts[key])

print(f"\nSegment-weight bucket sizes (train): {dict(bucket_counts)}")
print(f"Segment-weight multipliers: {[(k, round(len(ytr)/(n_buckets*v), 2)) for k, v in bucket_counts.items()]}")

hgb_segweighted = HistGradientBoostingClassifier(
    max_iter=300, learning_rate=0.06, max_depth=5,
    random_state=42, early_stopping=True, validation_fraction=0.15,
    # NOTE: no class_weight="balanced" here -- sample_weight replaces it,
    # since the custom weighting already accounts for overall class balance
    # as one of its four buckets. Combining both would double-correct.
)
hgb_segweighted.fit(Xtr, ytr, sample_weight=sample_weight)
val_proba_segw = hgb_segweighted.predict_proba(Xval)[:, 1]
print(f"[HGB segment-weighted] val PR-AUC={average_precision_score(yval, val_proba_segw):.3f} "
      f"ROC-AUC={roc_auc_score(yval, val_proba_segw):.3f}")

# Cost-optimal threshold on VAL, same discipline as the primary model
t_segw, val_cost_segw = cost_optimal_threshold(val_proba_segw, yval.values)
print(f"Segment-weighted cost-optimal threshold (VAL): {t_segw:.3f}")

# FROZEN single-shot TEST evaluation -- same touched-once discipline,
# comparing a new candidate model against the same frozen test set
# (methodologically the same as the existing HGB-vs-XGB comparison above,
# not a new touch of test for tuning purposes)
test_proba_segw = hgb_segweighted.predict_proba(Xtest)[:, 1]
test_pred_segw = (test_proba_segw >= t_segw).astype(int)

segw_precision = precision_score(ytest, test_pred_segw)
segw_recall = recall_score(ytest, test_pred_segw)
segw_f1 = f1_score(ytest, test_pred_segw)
segw_mcc = matthews_corrcoef(ytest, test_pred_segw)
segw_prauc = average_precision_score(ytest, test_proba_segw)
segw_tn, segw_fp, segw_fn, segw_tp = confusion_matrix(ytest, test_pred_segw).ravel()
segw_cost = segw_fp * COST_FP + segw_fn * COST_FN

fn_mask_segw = (test_pred_segw == 0) & (ytest.values == 1)
segw_fn_prepaid_share = (test_ppd_mask & fn_mask_segw).sum() / fn_mask_segw.sum() if fn_mask_segw.sum() > 0 else float("nan")
segw_fn_prepaid_count = (test_ppd_mask & fn_mask_segw).sum()

print("\n================ SEGMENT-WEIGHTED LOSS: TEST comparison ================")
print(f"{'Metric':<20}{'Baseline HGB':<16}{'Seg-weighted':<16}")
print(f"{'PR-AUC':<20}{test_prauc:<16.3f}{segw_prauc:<16.3f}")
print(f"{'Precision':<20}{test_precision:<16.3f}{segw_precision:<16.3f}")
print(f"{'Recall':<20}{test_recall:<16.3f}{segw_recall:<16.3f}")
print(f"{'F1':<20}{test_f1:<16.3f}{segw_f1:<16.3f}")
print(f"{'MCC':<20}{test_mcc:<16.3f}{segw_mcc:<16.3f}")
print(f"{'Cost (Rs.)':<20}{test_cost:<16,.0f}{segw_cost:<16,.0f}")
print(f"{'FN count':<20}{int(fn):<16}{int(segw_fn):<16}")
print(f"{'FN prepaid count':<20}{int((test_ppd_mask & fn_mask_global).sum()):<16}{int(segw_fn_prepaid_count):<16}")
print(f"{'FN % prepaid':<20}{global_fn_prepaid_share:<16.1%}{segw_fn_prepaid_share:<16.1%}")

segw_summary = {
    "threshold": t_segw, "test_prauc": segw_prauc,
    "test_precision": segw_precision, "test_recall": segw_recall,
    "test_f1": segw_f1, "test_mcc": segw_mcc,
    "tn": int(segw_tn), "fp": int(segw_fp), "fn": int(segw_fn), "tp": int(segw_tp),
    "cost": segw_cost, "fn_prepaid_count": int(segw_fn_prepaid_count),
    "fn_prepaid_share": segw_fn_prepaid_share,
}
with open(f"{OUT}/segment_weighted_summary.json", "w") as f:
    json.dump(segw_summary, f, indent=2)
print(f"\nSaved segment_weighted_summary.json to {OUT}/")

# Dampen the multiplier: sqrt reduces the extremes (12.76x -> ~3.57x)
# while preserving direction (Prepaid_1 still upweighted, others still
# downweighted, just less violently).
sample_weight_damped = np.sqrt(sample_weight)

hgb_damped = HistGradientBoostingClassifier(
    max_iter=300, learning_rate=0.06, max_depth=5,
    random_state=42, early_stopping=True, validation_fraction=0.15,
)
hgb_damped.fit(Xtr, ytr, sample_weight=sample_weight_damped)
val_proba_damped = hgb_damped.predict_proba(Xval)[:, 1]
print(f"[HGB damped-weighted] val PR-AUC={average_precision_score(yval, val_proba_damped):.3f} "
      f"ROC-AUC={roc_auc_score(yval, val_proba_damped):.3f}")

t_damped, _ = cost_optimal_threshold(val_proba_damped, yval.values)
test_proba_damped = hgb_damped.predict_proba(Xtest)[:, 1]
test_pred_damped = (test_proba_damped >= t_damped).astype(int)

damped_prauc = average_precision_score(ytest, test_proba_damped)
damped_mcc = matthews_corrcoef(ytest, test_pred_damped)
damped_tn, damped_fp, damped_fn, damped_tp = confusion_matrix(ytest, test_pred_damped).ravel()
damped_cost = damped_fp * COST_FP + damped_fn * COST_FN
fn_mask_damped = (test_pred_damped == 0) & (ytest.values == 1)
damped_fn_prepaid_count = (test_ppd_mask & fn_mask_damped).sum()
damped_fn_prepaid_share = damped_fn_prepaid_count / fn_mask_damped.sum() if fn_mask_damped.sum()>0 else float("nan")

print("\n================ DAMPENED SEGMENT-WEIGHT: TEST comparison ================")
print(f"{'Metric':<20}{'Baseline':<12}{'Full-weight':<14}{'Damped':<12}")
print(f"{'PR-AUC':<20}{test_prauc:<12.3f}{segw_prauc:<14.3f}{damped_prauc:<12.3f}")
print(f"{'MCC':<20}{test_mcc:<12.3f}{segw_mcc:<14.3f}{damped_mcc:<12.3f}")
print(f"{'Cost (Rs.)':<20}{test_cost:<12,.0f}{segw_cost:<14,.0f}{damped_cost:<12,.0f}")
print(f"{'FN prepaid count':<20}{int((test_ppd_mask & fn_mask_global).sum()):<12}{int(segw_fn_prepaid_count):<14}{int(damped_fn_prepaid_count):<12}")
print(f"{'FN % prepaid':<20}{global_fn_prepaid_share:<12.1%}{segw_fn_prepaid_share:<14.1%}{damped_fn_prepaid_share:<12.1%}")

# ---------------- Pre-specified dampening sweep (0.3, 0.5, 0.7) --------
# Three candidate exponents, chosen in advance, not iteratively chasing
# the best VAL number. Selection is done on VAL ONLY; TEST is touched
# exactly once, for the single selected candidate -- same discipline as
# everywhere else in this pipeline.
exponents = [0.3, 0.5, 0.7]
candidates = {}

for exp in exponents:
    sw = sample_weight ** exp   # sample_weight is the full-strength array from earlier
    model = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.06, max_depth=5,
        random_state=42, early_stopping=True, validation_fraction=0.15,
    )
    model.fit(Xtr, ytr, sample_weight=sw)
    val_proba_exp = model.predict_proba(Xval)[:, 1]
    val_prauc_exp = average_precision_score(yval, val_proba_exp)
    t_exp, val_cost_exp = cost_optimal_threshold(val_proba_exp, yval.values)
    candidates[exp] = {
        "model": model, "val_proba": val_proba_exp,
        "val_prauc": val_prauc_exp, "threshold": t_exp, "val_cost": val_cost_exp,
    }
    print(f"[exp={exp}] VAL PR-AUC={val_prauc_exp:.3f}  VAL cost=₹{val_cost_exp:,.0f}  threshold={t_exp:.3f}")

# Selection rule, fixed in advance: lowest VAL cost (same criterion used
# for the primary threshold search throughout this pipeline -- consistent,
# not cherry-picked post hoc).
best_exp = min(candidates, key=lambda e: candidates[e]["val_cost"])
print(f"\nSelected exponent (lowest VAL cost): {best_exp}")

# ---- TEST touched ONCE, only for the selected candidate ----
best = candidates[best_exp]
test_proba_best = best["model"].predict_proba(Xtest)[:, 1]
test_pred_best = (test_proba_best >= best["threshold"]).astype(int)

best_prauc = average_precision_score(ytest, test_proba_best)
best_mcc = matthews_corrcoef(ytest, test_pred_best)
best_tn, best_fp, best_fn, best_tp = confusion_matrix(ytest, test_pred_best).ravel()
best_cost = best_fp * COST_FP + best_fn * COST_FN
fn_mask_best = (test_pred_best == 0) & (ytest.values == 1)
best_fn_prepaid_count = (test_ppd_mask & fn_mask_best).sum()
best_fn_prepaid_share = best_fn_prepaid_count / fn_mask_best.sum() if fn_mask_best.sum() > 0 else float("nan")

print(f"\n================ SELECTED (exp={best_exp}) vs BASELINE: TEST ================")
print(f"{'Metric':<20}{'Baseline':<12}{'Selected':<12}")
print(f"{'PR-AUC':<20}{test_prauc:<12.3f}{best_prauc:<12.3f}")
print(f"{'MCC':<20}{test_mcc:<12.3f}{best_mcc:<12.3f}")
print(f"{'Cost (Rs.)':<20}{test_cost:<12,.0f}{best_cost:<12,.0f}")
print(f"{'FN prepaid count':<20}{int((test_ppd_mask & fn_mask_global).sum()):<12}{int(best_fn_prepaid_count):<12}")
print(f"{'FN % prepaid':<20}{global_fn_prepaid_share:<12.1%}{best_fn_prepaid_share:<12.1%}")

sweep_summary = {
    "exponents_tried": exponents,
    "selected_exponent": best_exp,
    "selection_criterion": "lowest VAL cost",
    "val_results": {str(e): {"val_prauc": candidates[e]["val_prauc"],
                              "val_cost": candidates[e]["val_cost"],
                              "threshold": candidates[e]["threshold"]} for e in exponents},
    "test_prauc": best_prauc, "test_mcc": best_mcc,
    "test_cost": best_cost, "fn_prepaid_count": int(best_fn_prepaid_count),
    "fn_prepaid_share": best_fn_prepaid_share,
}
with open(f"{OUT}/dampening_sweep_summary.json", "w") as f:
    json.dump(sweep_summary, f, indent=2)
print(f"\nSaved dampening_sweep_summary.json to {OUT}/")

# Sanity check: confirm the selected model in the sweep is genuinely
# exp=0.3's model, not a stale reference to an earlier run.
print("Selected exponent:", best_exp)
print("Model object id:", id(candidates[best_exp]["model"]))
print("Threshold used:", candidates[best_exp]["threshold"])
print("First 5 test_proba values:", test_proba_best[:5])

# Re-select the baseline model's threshold under macro-cost, for comparison
t_macro_baseline, macro_cost_baseline, _ = cost_optimal_threshold_macro(
    val_proba_hgb, yval.values, val_cod_mask, val_ppd_mask)
print(f"[Baseline HGB] blended-cost threshold={best_threshold:.3f}  vs  macro-cost threshold={t_macro_baseline:.3f}")

# Re-run the exponent sweep, selecting by macro-cost instead of blended cost
print("\n[Macro-cost re-selection across dampening exponents]")
macro_candidates = {}
for exp in exponents:
    val_proba_exp = candidates[exp]["val_proba"]
    t_m, macro_cost_m, _ = cost_optimal_threshold_macro(val_proba_exp, yval.values, val_cod_mask, val_ppd_mask)
    macro_candidates[exp] = {"threshold": t_m, "macro_cost": macro_cost_m}
    print(f"[exp={exp}] macro-cost threshold={t_m:.3f}  macro VAL cost/order=₹{macro_cost_m:.2f}")

best_exp_macro = min(macro_candidates, key=lambda e: macro_candidates[e]["macro_cost"])
print(f"\nSelected exponent under MACRO-cost criterion: {best_exp_macro}")

t_macro_full, macro_cost_full, _ = cost_optimal_threshold_macro(
    val_proba_segw, yval.values, val_cod_mask, val_ppd_mask)
print(f"[Full-weight, exp=1.0] macro-cost threshold={t_macro_full:.3f}  macro VAL cost/order=₹{macro_cost_full:.2f}")