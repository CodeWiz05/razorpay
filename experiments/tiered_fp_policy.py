"""
experiments/tiered_fp_policy.py
================================
Tests a two-threshold intervention policy against the existing single
threshold, using ONLY already-saved artifacts (calibrator.joblib,
model.joblib) -- no retraining, so no seed variance to worry about.

Mechanism: a lower-probability band gets a cheap SMS/auto-call nudge
(COST_FP_LOW) instead of skipping those orders entirely; a higher band
keeps the existing full verification call (COST_FP_HIGH, same as the
current single COST_FP). This can only reduce cost relative to the
single-threshold policy, since it is a strict refinement of it (setting
t_low = t_high recovers the exact current policy).

COST_FP_LOW = 2.0: automated SMS/WhatsApp/robo-call confirmation, per
India COD-verification tooling (WordPress "COD Order Confirmation for
India" plugin: Rs 2/call, Rs 0.40/SMS).
COST_FP_HIGH = 25.0: unchanged, existing full-verification-call estimate,
sits at the top of the cited Rs 8-25/order AI-voice-bot COD-verification
range (Caller Digital) -- kept as-is per the COST_FP research pass.
"""
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from features import load_features, TARGET

OUT = str(Path(__file__).resolve().parent.parent / "outputs")

COST_FN = 180.0
COST_FP_HIGH = 25.0   # full verification call (existing FP cost)
COST_FP_LOW = 2.0     # cheap SMS / auto-call nudge

CURRENT_SINGLE_THRESHOLD = 0.110  # for side-by-side comparison

# ---- Recreate the exact same temporal split as train.py (deterministic) ----
df, feature_cols = load_features()
df = df.sort_values("order_date").reset_index(drop=True)
t1 = df["order_date"].quantile(8 / 12)
t2 = df["order_date"].quantile(10 / 12)
val = df[(df["order_date"] >= t1) & (df["order_date"] < t2)]
test = df[df["order_date"] >= t2]

Xval, yval = val[feature_cols], val[TARGET]
Xtest, ytest = test[feature_cols], test[TARGET]

model = joblib.load(f"{OUT}/model.joblib")
calibrator = joblib.load(f"{OUT}/calibrator.joblib")

val_proba_cal = calibrator.predict(model.predict_proba(Xval)[:, 1])
test_proba_cal = calibrator.predict(model.predict_proba(Xtest)[:, 1])

val_cod_mask = (val["payment_mode_COD"] == True).values
val_ppd_mask = (val["payment_mode_Prepaid"] == True).values
test_cod_mask = (test["payment_mode_COD"] == True).values
test_ppd_mask = (test["payment_mode_Prepaid"] == True).values


def tiered_cost(proba, y_true, t_low, t_high, cod_mask, ppd_mask):
    tier = np.where(proba >= t_high, 2, np.where(proba >= t_low, 1, 0))
    y = y_true.values if hasattr(y_true, "values") else y_true

    def cost_per_order(mask):
        m = mask
        fp_low = ((tier == 1) & (y == 0) & m).sum()
        fp_high = ((tier == 2) & (y == 0) & m).sum()
        fn = ((tier == 0) & (y == 1) & m).sum()
        n = m.sum()
        return (fp_low * COST_FP_LOW + fp_high * COST_FP_HIGH + fn * COST_FN) / n if n > 0 else np.nan

    macro = (cost_per_order(cod_mask) + cost_per_order(ppd_mask)) / 2
    return macro, tier


# ---- Grid search both thresholds on VAL only ----
grid = np.linspace(0.02, 0.95, 94)
best = (None, None, np.inf)
for t_low in grid:
    for t_high in grid:
        if t_high < t_low:
            continue
        macro, _ = tiered_cost(val_proba_cal, yval, t_low, t_high, val_cod_mask, val_ppd_mask)
        if macro < best[2]:
            best = (t_low, t_high, macro)

t_low_best, t_high_best, val_macro_best = best
print(f"[VAL] best tiered thresholds: t_low={t_low_best:.3f}  t_high={t_high_best:.3f}  "
      f"macro cost/order=Rs {val_macro_best:.2f}")

# Reference: current single-threshold macro cost on the SAME val data
single_macro, _ = tiered_cost(val_proba_cal, yval, CURRENT_SINGLE_THRESHOLD,
                               CURRENT_SINGLE_THRESHOLD, val_cod_mask, val_ppd_mask)
print(f"[VAL] current single-threshold ({CURRENT_SINGLE_THRESHOLD}) macro cost/order: Rs {single_macro:.2f}")
print(f"[VAL] improvement: {(1 - val_macro_best/single_macro)*100:.1f}%")

# ---- Frozen, single-shot evaluation on TEST ----
test_macro, test_tier = tiered_cost(test_proba_cal, ytest, t_low_best, t_high_best,
                                     test_cod_mask, test_ppd_mask)
y = ytest.values
fp_low = ((test_tier == 1) & (y == 0)).sum()
fp_high = ((test_tier == 2) & (y == 0)).sum()
fn = ((test_tier == 0) & (y == 1)).sum()
tp = ((test_tier > 0) & (y == 1)).sum()
total_cost = fp_low * COST_FP_LOW + fp_high * COST_FP_HIGH + fn * COST_FN

print(f"\n[TEST, frozen] tier1(nudge) FP={fp_low}  tier2(call) FP={fp_high}  FN={fn}  TP={tp}")
print(f"[TEST] total cost: Rs {total_cost:,.0f}")

# Comparison: same TEST set, current single-threshold policy
single_tier = np.where(test_proba_cal >= CURRENT_SINGLE_THRESHOLD, 2, 0)
fp_single = ((single_tier == 2) & (y == 0)).sum()
fn_single = ((single_tier == 0) & (y == 1)).sum()
cost_single = fp_single * COST_FP_HIGH + fn_single * COST_FN
print(f"[TEST] current single-threshold: FP={fp_single}  FN={fn_single}  cost=Rs {cost_single:,.0f}")
print(f"[TEST] cost reduction vs current: {(1 - total_cost/cost_single)*100:.1f}%")

# Recall by segment, both policies, for direct comparison to your frozen numbers
for name, tier_arr in [("tiered", test_tier), ("single", single_tier)]:
    flagged = tier_arr > 0
    for seg_name, mask in [("Prepaid", test_ppd_mask), ("COD", test_cod_mask)]:
        pos = (mask & (y == 1)).sum()
        caught = (mask & flagged & (y == 1)).sum()
        recall = caught / pos if pos > 0 else float("nan")
        print(f"[{name}] {seg_name} recall: {recall:.3f} ({caught}/{pos})")


# =============================================================================
# POLICY 3: lift-based tiered thresholds
# Thresholds are set as a multiple of EACH SEGMENT'S OWN base rate, not a
# global probability cutoff -- same relative-lift logic already validated
# for the PR-AUC cross-population comparison. This ties K to something
# already defended in the README rather than to an unconstrained cost
# optimum. See conversation for the reasoning on why raw-probability cheap
# tiers are indefensible at scale (destroys the "unusual order" signal the
# intervention depends on).
# =============================================================================
val_y_arr = yval.values
base_rate_cod = val_y_arr[val_cod_mask].mean()
base_rate_ppd = val_y_arr[val_ppd_mask].mean()
print(f"\n[Lift-based policy] VAL base rates: COD={base_rate_cod:.3f}  Prepaid={base_rate_ppd:.3f}")


def lift_tiered_cost(proba, y_true, lift_low, lift_high, cod_mask, ppd_mask,
                      base_cod, base_ppd):
    y = y_true.values if hasattr(y_true, "values") else y_true
    t_low_cod = min(0.95, lift_low * base_cod)
    t_high_cod = min(0.95, lift_high * base_cod)
    t_low_ppd = min(0.95, lift_low * base_ppd)
    t_high_ppd = min(0.95, lift_high * base_ppd)

    tier = np.zeros(len(proba), dtype=int)
    tier[cod_mask & (proba >= t_low_cod)] = 1
    tier[cod_mask & (proba >= t_high_cod)] = 2
    tier[ppd_mask & (proba >= t_low_ppd)] = 1
    tier[ppd_mask & (proba >= t_high_ppd)] = 2

    def cost_per_order(mask):
        fp_low = ((tier == 1) & (y == 0) & mask).sum()
        fp_high = ((tier == 2) & (y == 0) & mask).sum()
        fn = ((tier == 0) & (y == 1) & mask).sum()
        n = mask.sum()
        return (fp_low * COST_FP_LOW + fp_high * COST_FP_HIGH + fn * COST_FN) / n if n > 0 else np.nan

    macro = (cost_per_order(cod_mask) + cost_per_order(ppd_mask)) / 2
    return macro, tier, (t_low_cod, t_high_cod, t_low_ppd, t_high_ppd)


lift_grid = np.arange(0.25, 10.01, 0.25)
best_lift = (None, None, np.inf, None)
for ll in lift_grid:
    for lh in lift_grid:
        if lh < ll:
            continue
        macro, _, thr = lift_tiered_cost(val_proba_cal, yval, ll, lh, val_cod_mask, val_ppd_mask,
                                          base_rate_cod, base_rate_ppd)
        if macro < best_lift[2]:
            best_lift = (ll, lh, macro, thr)

ll_best, lh_best, val_macro_lift, thr_best = best_lift
print(f"[VAL] best lift multiples: lift_low={ll_best:.2f}  lift_high={lh_best:.2f}  "
      f"macro cost/order=Rs {val_macro_lift:.2f}")
print(f"[VAL] implied thresholds: COD nudge>={thr_best[0]:.3f} call>={thr_best[1]:.3f}  "
      f"Prepaid nudge>={thr_best[2]:.3f} call>={thr_best[3]:.3f}")

test_macro_lift, test_tier_lift, _ = lift_tiered_cost(
    test_proba_cal, ytest, ll_best, lh_best, test_cod_mask, test_ppd_mask,
    base_rate_cod, base_rate_ppd)

fp_low_l = ((test_tier_lift == 1) & (y == 0)).sum()
fp_high_l = ((test_tier_lift == 2) & (y == 0)).sum()
fn_l = ((test_tier_lift == 0) & (y == 1)).sum()
total_cost_l = fp_low_l * COST_FP_LOW + fp_high_l * COST_FP_HIGH + fn_l * COST_FN
flagged_l = (test_tier_lift > 0).sum()

print(f"\n[TEST, frozen, lift policy] tier1(nudge) FP={fp_low_l}  tier2(call) FP={fp_high_l}  FN={fn_l}")
print(f"[TEST] total flagged: {flagged_l} / {len(y)} ({flagged_l/len(y)*100:.1f}%)")
print(f"[TEST] total cost: Rs {total_cost_l:,.0f}  (vs current single-threshold Rs {cost_single:,.0f}, "
      f"vs uncapped tiered Rs {total_cost:,.0f})")

for seg_name, mask in [("Prepaid", test_ppd_mask), ("COD", test_cod_mask)]:
    pos = (mask & (y == 1)).sum()
    caught = (mask & (test_tier_lift > 0) & (y == 1)).sum()
    recall = caught / pos if pos > 0 else float("nan")
    print(f"[lift policy] {seg_name} recall: {recall:.3f} ({caught}/{pos})")