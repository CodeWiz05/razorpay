"""
experiments/listing_feature_paired_test.py
============================================
Paired 8-seed test: does product_past_return_rate improve Prepaid recall?

Design: same seed trains BOTH a "without" model (current production
feature set) and a "with" model (production + product_past_return_rate),
so inter-seed training noise (documented: Prepaid recall 0.130-0.511
across seeds with ZERO changes) cancels out of the per-seed DIFFERENCE
even though it doesn't cancel out of either model's raw recall. This is
the same paired-comparison discipline used for the segment-specific
isotonic calibration test.

Run this AFTER regenerating data_orders.csv with the updated
generate_data.py (must include product_id / product_past_return_rate).
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import recall_score
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from features import load_features, TARGET

COST_FN = 180.0
COST_FP = 25.0
N_SEEDS = 8

df, feature_cols = load_features()
df = df.sort_values("order_date").reset_index(drop=True)

# ---- Leakage self-check on the NEW feature before trusting anything ----
# Same discipline as verify_leakage_invariant.py, applied to
# product_past_return_rate specifically, per the project's standing rule
# that any new "history" feature must be re-verified.
if "product_past_return_rate" not in df.columns:
    raise SystemExit(
        "product_past_return_rate not found -- regenerate data_orders.csv "
        "with the updated generate_data.py before running this experiment."
    )

prod_counts = {}
recomputed = np.zeros(len(df))
for i in range(len(df)):
    pid = df.at[i, "product_id"] if "product_id" in df.columns else None
    if pid is None:
        raise SystemExit("product_id column missing -- cannot verify leakage safety.")
    n, r = prod_counts.get(pid, (0, 0))
    recomputed[i] = (r / n) if n >= 3 else 0.16
    prod_counts[pid] = (n + 1, r + int(df.at[i, TARGET]))

mismatches = (np.round(recomputed, 3) != df["product_past_return_rate"]).sum()
print(f"[Leakage check] product_past_return_rate mismatches: {mismatches} / {len(df)} "
      f"(should be 0)")
full_hist_rate = df.groupby("product_id")[TARGET].transform("mean")
leaky_match = np.isclose(full_hist_rate, df["product_past_return_rate"], atol=0.01).sum()
print(f"[Leakage check] rows matching FULL-HISTORY (leaky) rate: {leaky_match} / {len(df)} "
      f"(should be far less than total)")
if mismatches > 0:
    raise SystemExit("Leakage check FAILED -- do not trust results below until fixed.")

prod_counts_running = {}
prior_count_at_row = np.zeros(len(df), dtype=int)
for i in range(len(df)):
    pid = df.at[i, "product_id"]
    n, r = prod_counts_running.get(pid, (0, 0))
    prior_count_at_row[i] = n
    prod_counts_running[pid] = (n + 1, r + int(df.at[i, TARGET]))

df["_prior_ct"] = prior_count_at_row
is_leaky_match = np.isclose(full_hist_rate, df["product_past_return_rate"], atol=0.01)
for lo, hi in [(0, 10), (10, 30), (30, 60), (60, 200)]:
    mask = (df["_prior_ct"] >= lo) & (df["_prior_ct"] < hi)
    print(f"  prior orders [{lo},{hi}): leaky-match rate = {is_leaky_match[mask].mean():.1%}  (n={mask.sum()})")

# ---- Temporal split, identical to train.py ----
t1 = df["order_date"].quantile(8 / 12)
t2 = df["order_date"].quantile(10 / 12)
train = df[df["order_date"] < t1]
val = df[(df["order_date"] >= t1) & (df["order_date"] < t2)]
test = df[df["order_date"] >= t2]

feature_cols_without = [c for c in feature_cols if c != "product_past_return_rate"]
feature_cols_with = feature_cols  # includes it

ytr, yval, ytest = train[TARGET], val[TARGET], test[TARGET]
val_cod_mask = (val["payment_mode_COD"] == True).values
val_ppd_mask = (val["payment_mode_Prepaid"] == True).values
test_ppd_mask = (test["payment_mode_Prepaid"] == True).values

# Sample weights identical across both models -- depend only on
# payment_mode x returned bucket, not on feature set.
train_mode = train["payment_mode_Prepaid"].map({True: "Prepaid", False: "COD"})
bucket_counts = train_mode.astype(str).str.cat(ytr.astype(str), sep="_").value_counts()
n_buckets = 4
sample_weight_full = np.ones(len(ytr))
for i in range(len(ytr)):
    key = f"{train_mode.iloc[i]}_{ytr.iloc[i]}"
    sample_weight_full[i] = len(ytr) / (n_buckets * bucket_counts[key])
sample_weight_damped = sample_weight_full ** 0.3


def macro_cost_threshold(proba, y_true, cod_mask, ppd_mask):
    ths = np.linspace(0.05, 0.95, 181)
    best_t, best_macro = None, np.inf
    for t in ths:
        pred = (proba >= t).astype(int)
        def cpo(mask):
            fp = ((pred == 1) & (y_true == 0) & mask).sum()
            fn = ((pred == 0) & (y_true == 1) & mask).sum()
            n = mask.sum()
            return (fp * COST_FP + fn * COST_FN) / n if n > 0 else np.nan
        macro = (cpo(cod_mask) + cpo(ppd_mask)) / 2
        if macro < best_macro:
            best_macro, best_t = macro, t
    return best_t


def train_eval(feature_set, seed):
    Xtr, Xval, Xtest = train[feature_set], val[feature_set], test[feature_set]
    hgb = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.06, max_depth=5,
        random_state=seed, early_stopping=True, validation_fraction=0.15,
    )
    hgb.fit(Xtr, ytr, sample_weight=sample_weight_damped)
    val_proba = hgb.predict_proba(Xval)[:, 1]

    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(val_proba, yval)
    val_proba_cal = calibrator.predict(val_proba)

    t = macro_cost_threshold(val_proba_cal, yval.values, val_cod_mask, val_ppd_mask)

    test_proba = hgb.predict_proba(Xtest)[:, 1]
    test_proba_cal = calibrator.predict(test_proba)
    test_pred = (test_proba_cal >= t).astype(int)

    ppd_pos = (test_ppd_mask & (ytest.values == 1)).sum()
    ppd_tp = (test_ppd_mask & (test_pred == 1) & (ytest.values == 1)).sum()
    ppd_recall = ppd_tp / ppd_pos if ppd_pos > 0 else float("nan")
    return ppd_recall, t


print(f"\nRunning {N_SEEDS} paired seeds (with vs. without product_past_return_rate)...")
diffs = []
for seed in range(N_SEEDS):
    recall_without, t_without = train_eval(feature_cols_without, seed)
    recall_with, t_with = train_eval(feature_cols_with, seed)
    diff = recall_with - recall_without
    diffs.append(diff)
    print(f"  seed={seed}: without={recall_without:.3f} (t={t_without:.3f})  "
          f"with={recall_with:.3f} (t={t_with:.3f})  diff={diff:+.3f}")

diffs = np.array(diffs)
mean_diff = diffs.mean()
se_diff = diffs.std(ddof=1) / np.sqrt(len(diffs))
n_positive = (diffs > 0).sum()

print(f"\n=== PAIRED RESULT ({N_SEEDS} seeds) ===")
print(f"Mean paired diff (Prepaid recall, with - without): {mean_diff:+.4f}")
print(f"SE of paired diff: {se_diff:.4f}")
print(f"Diff / SE ratio: {mean_diff/se_diff:.2f}")
print(f"Positive in {n_positive}/{N_SEEDS} seeds")
print("\nInterpretation guide (matching the segment-calibration precedent):")
print("  |diff/SE| >> 1 and consistent sign across seeds -> real effect, trust it.")
print("  |diff/SE| ~ 1 or mixed signs -> not distinguishable from training noise, do NOT adopt on this evidence alone.")