import json
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from sklearn.calibration import calibration_curve

OUT = "C:/Numair/Coding/Razorpay/outputs"

val_proba = np.load(f"{OUT}/val_proba_hgb.npy")
val_y = np.load(f"{OUT}/val_y.npy")
test_proba = np.load(f"{OUT}/test_proba.npy")
test_y = np.load(f"{OUT}/test_y.npy")
summary = json.load(open(f"{OUT}/summary.json"))
calibrator = joblib.load(f"{OUT}/calibrator.joblib")
val_proba_calibrated = calibrator.predict(val_proba)   # NEW
test_proba_calibrated = calibrator.predict(test_proba)

COST_FN, COST_FP = summary["cost_fn"], summary["cost_fp"]
best_t = summary["best_threshold"]

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

# --- 1. Precision-Recall curve (test set) ---
prec, rec, th = precision_recall_curve(test_y, test_proba)
axes[0].plot(rec, prec, color="#2563eb", lw=2)
axes[0].scatter([summary["test_recall"]], [summary["test_precision"]],
                color="#dc2626", zorder=5, s=60,
                label=f"chosen threshold={best_t:.2f}")
axes[0].set_xlabel("Recall")
axes[0].set_ylabel("Precision")
axes[0].set_title(f"Precision-Recall Curve (Test)\nPR-AUC={summary['test_prauc']:.3f}")
axes[0].legend()
axes[0].grid(alpha=0.3)

# --- 2. Cost vs threshold (val set, the curve threshold was chosen from) ---
thresholds = np.linspace(0.05, 0.95, 181)
costs = []
for t in thresholds:
    pred = (val_proba_calibrated >= t).astype(int)
    fp = ((pred == 1) & (val_y == 0)).sum()
    fn = ((pred == 0) & (val_y == 1)).sum()
    costs.append(fp * COST_FP + fn * COST_FN)
costs = np.array(costs)
axes[1].plot(thresholds, costs, color="#059669", lw=2)
axes[1].axvline(best_t, color="#dc2626", linestyle="--", label=f"optimal t={best_t:.2f}")
axes[1].set_xlabel("Decision threshold")
axes[1].set_ylabel("Total cost (₹) on validation set")
axes[1].set_title(f"Cost vs Threshold\n(FN=₹{COST_FN:.0f}, FP=₹{COST_FP:.0f})")
axes[1].legend()
axes[1].grid(alpha=0.3)

# --- 3. Calibration curve (test set): raw vs. isotonic-calibrated ---
# n_test=10,000, ~1,638 positives -- enough for quantile bins to be
# genuinely informative here, unlike the fraud artifact's 75-positive case.
frac_pos_raw, mean_pred_raw = calibration_curve(test_y, test_proba, n_bins=10, strategy="quantile")
frac_pos_cal, mean_pred_cal = calibration_curve(test_y, test_proba_calibrated, n_bins=10, strategy="quantile")
axes[2].plot([0, 1], [0, 1], "k--", alpha=0.4, label="perfectly calibrated")
axes[2].plot(mean_pred_raw, frac_pos_raw, marker="o", color="#94a3b8", label="raw HGB")
axes[2].plot(mean_pred_cal, frac_pos_cal, marker="o", color="#7c3aed", label="isotonic-calibrated")
axes[2].set_xlabel("Mean predicted return probability")
axes[2].set_ylabel("Observed return rate")
axes[2].set_title("Calibration (Test)")
axes[2].legend()
axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUT}/evaluation_plots.png", dpi=150)
print(f"Saved {OUT}/evaluation_plots.png")