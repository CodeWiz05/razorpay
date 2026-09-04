"""
segment_calibration_test.py
=============================
Tests per-segment isotonic calibration (fit COD and Prepaid calibrators
SEPARATELY, apply each within its own segment) against the shipped single
pooled calibrator -- both using the SAME shared closed-form threshold
(COST_FP/(COST_FP+COST_FN) = 0.122), never a re-searched segment-specific
cutpoint. That was already tested and shown to overfit the small Prepaid
validation slice (see README Section 5) -- this is a genuinely different
lever: WHAT PROBABILITY gets shown for a Prepaid order, not WHERE the
cutoff sits.

HYPOTHESIS: a single isotonic calibrator fit on POOLED COD+Prepaid val
scores may look well-calibrated in aggregate while being systematically
biased for Prepaid specifically, since COD's much larger raw-score
magnitude (COD_COEF alone adds +2.40 to the logit) can dominate the shape
of the pooled curve. If true, Prepaid orders that are genuinely risky
would come out numerically lower than they should, even though the
model's underlying RANKING of Prepaid orders (ROC-AUC) is fine.

DESIGN: PAIRED per-seed comparison, not independent-sample averaging.
Prior investigation showed prepaid recall swings 0.13-0.51 across random
seeds with ZERO other changes -- that's training-time model instability,
much larger than ordinary sampling noise (~3pp SE at n~230 positives).
Comparing raw averages across seeds would drown the calibration effect in
that noise. Instead: train ONE model per seed, then evaluate BOTH
calibration strategies on that SAME model. The paired difference
(segment-calibrated recall MINUS pooled recall, same seed, same model)
cancels the inter-seed training variance and isolates the calibration
strategy's effect specifically.

Self-contained: rebuilds its own feature matrix from data_orders.csv
rather than importing features.py/train.py, per the existing
experiments/ isolation pattern (see two_model_split.py).
"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, average_precision_score

DATA_PATH = str(Path(__file__).resolve().parent.parent / "data_orders.csv")
TARGET = "returned"
CATEGORICAL = ["category", "payment_mode", "pincode_tier"]
NUMERIC = [
    "price", "discount_pct", "delivery_days", "is_apparel",
    "customer_prior_orders", "customer_past_return_rate", "is_festive",
    "is_bracketed", "size_variant_count",
]
COST_FN, COST_FP = 180.0, 25.0
CLOSED_FORM_THRESHOLD = COST_FP / (COST_FP + COST_FN)
N_SEEDS = 8  # retrains the model N times -- the dominant noise source

df_raw = pd.read_csv(DATA_PATH, parse_dates=["order_date"]).sort_values("order_date").reset_index(drop=True)
df = pd.get_dummies(df_raw, columns=CATEGORICAL, drop_first=False)
df["payment_mode"] = df_raw["payment_mode"]  # keep the string column too -- dummies replaced it above
dummy_cols = [c for c in df.columns if c.startswith(tuple(f"{c_}_" for c_ in CATEGORICAL))]
feature_cols = NUMERIC + dummy_cols

t1 = df["order_date"].quantile(8 / 12)
t2 = df["order_date"].quantile(10 / 12)
train = df[df["order_date"] < t1]
val = df[(df["order_date"] >= t1) & (df["order_date"] < t2)]
test = df[df["order_date"] >= t2]

val_cod_mask = (val["payment_mode_COD"] == True).values
val_ppd_mask = (val["payment_mode_Prepaid"] == True).values
test_cod_mask = (test["payment_mode_COD"] == True).values
test_ppd_mask = (test["payment_mode_Prepaid"] == True).values

print(f"Val:  COD={val_cod_mask.sum():,} (pos={val[TARGET].values[val_cod_mask].sum()})  "
      f"Prepaid={val_ppd_mask.sum():,} (pos={val[TARGET].values[val_ppd_mask].sum()})")
print(f"Test: COD={test_cod_mask.sum():,} (pos={test[TARGET].values[test_cod_mask].sum()})  "
      f"Prepaid={test_ppd_mask.sum():,} (pos={test[TARGET].values[test_ppd_mask].sum()})")
print(f"Closed-form shared threshold: {CLOSED_FORM_THRESHOLD:.4f}\n")


def damped_sample_weights(payment_mode, y, damp=0.3, n_buckets=4):
    """Same recipe as the production model: bucket-balanced across
    payment_mode x returned, damped by exponent 0.3."""
    key = payment_mode.astype(str) + "_" + y.astype(str)
    bucket_counts = key.value_counts()
    w = key.map(lambda k: len(key) / (n_buckets * bucket_counts[k])).values
    return w ** damp


def evaluate(pred, y_true, cod_mask, ppd_mask):
    tp = ((pred == 1) & (y_true == 1)).sum()
    fp = ((pred == 1) & (y_true == 0)).sum()
    fn = ((pred == 0) & (y_true == 1)).sum()
    cost = fp * COST_FP + fn * COST_FN
    cod_pos = (cod_mask & (y_true == 1))
    ppd_pos = (ppd_mask & (y_true == 1))
    cod_recall = ((pred == 1) & cod_pos).sum() / cod_pos.sum() if cod_pos.sum() else np.nan
    ppd_recall = ((pred == 1) & ppd_pos).sum() / ppd_pos.sum() if ppd_pos.sum() else np.nan
    return dict(cost=cost, cod_recall=cod_recall, ppd_recall=ppd_recall, fp=fp, tp=tp, fn=fn)


rows = []
for seed in range(N_SEEDS):
    ytr = train[TARGET]
    yval = val[TARGET].values
    ytest = test[TARGET].values

    sw = damped_sample_weights(train["payment_mode"], ytr)
    hgb = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.06, max_depth=5, random_state=seed,
        early_stopping=True, validation_fraction=0.15,
    )
    hgb.fit(train[feature_cols], ytr, sample_weight=sw)

    val_proba = hgb.predict_proba(val[feature_cols])[:, 1]
    test_proba = hgb.predict_proba(test[feature_cols])[:, 1]

    ppd_rocauc = roc_auc_score(yval[val_ppd_mask], val_proba[val_ppd_mask])
    ppd_prauc = average_precision_score(yval[val_ppd_mask], val_proba[val_ppd_mask])

    # ---- Approach A: pooled calibrator (current shipped approach) --------
    cal_pooled = IsotonicRegression(out_of_bounds="clip")
    cal_pooled.fit(val_proba, yval)
    test_proba_pooled = cal_pooled.predict(test_proba)
    pred_pooled = (test_proba_pooled >= CLOSED_FORM_THRESHOLD).astype(int)
    res_a = evaluate(pred_pooled, ytest, test_cod_mask, test_ppd_mask)

    # ---- Approach B: per-segment calibrators ------------------------------
    cal_cod = IsotonicRegression(out_of_bounds="clip")
    cal_cod.fit(val_proba[val_cod_mask], yval[val_cod_mask])
    cal_ppd = IsotonicRegression(out_of_bounds="clip")
    cal_ppd.fit(val_proba[val_ppd_mask], yval[val_ppd_mask])

    test_proba_segcal = np.empty_like(test_proba)
    test_proba_segcal[test_cod_mask] = cal_cod.predict(test_proba[test_cod_mask])
    test_proba_segcal[test_ppd_mask] = cal_ppd.predict(test_proba[test_ppd_mask])
    pred_segcal = (test_proba_segcal >= CLOSED_FORM_THRESHOLD).astype(int)
    res_b = evaluate(pred_segcal, ytest, test_cod_mask, test_ppd_mask)

    rows.append(dict(
        seed=seed, val_ppd_rocauc=ppd_rocauc, val_ppd_prauc=ppd_prauc,
        pooled_cost=res_a["cost"], pooled_cod_recall=res_a["cod_recall"], pooled_ppd_recall=res_a["ppd_recall"],
        pooled_fp=res_a["fp"], pooled_tp=res_a["tp"], pooled_fn=res_a["fn"],
        segcal_cost=res_b["cost"], segcal_cod_recall=res_b["cod_recall"], segcal_ppd_recall=res_b["ppd_recall"],
        segcal_fp=res_b["fp"], segcal_tp=res_b["tp"], segcal_fn=res_b["fn"],
    ))
    print(f"[seed {seed}] val Prepaid ROC-AUC={ppd_rocauc:.3f}  |  "
          f"pooled: ppd_recall={res_a['ppd_recall']:.3f} cost=Rs{res_a['cost']:,.0f}  |  "
          f"segcal: ppd_recall={res_b['ppd_recall']:.3f} cost=Rs{res_b['cost']:,.0f}  |  "
          f"paired diff (segcal-pooled) ppd_recall={res_b['ppd_recall']-res_a['ppd_recall']:+.3f}")

results = pd.DataFrame(rows)

print("\n================ RAW AVERAGES ACROSS SEEDS (high inter-seed noise) ================")
for col in ["pooled_ppd_recall", "segcal_ppd_recall", "pooled_cod_recall", "segcal_cod_recall",
            "pooled_cost", "segcal_cost"]:
    print(f"{col:24s} mean={results[col].mean():.4f}  std={results[col].std():.4f}")

print("\n================ PAIRED DIFFERENCES (segcal - pooled, SAME model each seed) ============")
diff_ppd_recall = results["segcal_ppd_recall"] - results["pooled_ppd_recall"]
diff_cod_recall = results["segcal_cod_recall"] - results["pooled_cod_recall"]
diff_cost = results["segcal_cost"] - results["pooled_cost"]
for name, diff in [("Prepaid recall", diff_ppd_recall), ("COD recall", diff_cod_recall), ("Cost (Rs)", diff_cost)]:
    mean, std = diff.mean(), diff.std()
    se = std / np.sqrt(N_SEEDS)
    print(f"{name:16s} mean diff={mean:+.4f}  std={std:.4f}  SE={se:.4f}  "
          f"(positive-in-every-seed: {(diff > 0).sum()}/{N_SEEDS})")

print(f"\nNOTE: with n={N_SEEDS} seeds, this is a directional signal, not a hypothesis-test-grade")
print("result. If the paired diff is consistently positive across most/all seeds AND")
print("larger than its SE, that's real evidence worth adopting. A mixed sign pattern")
print("(some seeds up, some down) means this is inside the noise band -- same conclusion")
print("structure as the segment-threshold experiment that was rejected.")

out_path = str(Path(__file__).resolve().parent / "segment_calibration_test_results.csv")
results.to_csv(out_path, index=False)
print(f"\nSaved per-seed results to {out_path}")