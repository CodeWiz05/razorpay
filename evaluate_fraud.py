import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from pathlib import Path

OUT = str(Path(__file__).resolve().parent / "outputs")

val_proba_calibrated = np.load(f"{OUT}/val_proba_fraud_calibrated.npy")
val_y = np.load(f"{OUT}/val_y_fraud.npy")
test_proba = np.load(f"{OUT}/test_proba_fraud.npy")
test_y = np.load(f"{OUT}/test_y_fraud.npy")
summary = json.load(open(f"{OUT}/summary_fraud.json"))

COST_FN, COST_FP = summary["cost_fn"], summary["cost_fp"]
best_t = summary["best_threshold"]

# No third (calibration) panel: reliability diagram isn't informative at
# this base rate/n_pos -- see train_fraud.py. Brier scores are in summary_fraud.json.
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

# --- 1. Precision-Recall curve (test set) ---
prec, rec, th = precision_recall_curve(test_y, test_proba)
axes[0].plot(rec, prec, color="#2563eb", lw=2)
axes[0].scatter([summary["test_recall"]], [summary["test_precision"]],
                color="#dc2626", zorder=5, s=60,
                label=f"chosen threshold={best_t:.2f}")
axes[0].set_xlabel("Recall")
axes[0].set_ylabel("Precision")
axes[0].set_title(f"Precision-Recall Curve (Test)\nPR-AUC={summary['test_prauc']:.3f}  (n_pos={int(summary['tp']+summary['fn'])}, small base)")
axes[0].legend()
axes[0].grid(alpha=0.3)

# --- 2. Cost vs threshold (val set, calibrated scale -- matches train_fraud.py) ---
thresholds = np.linspace(0.01, 0.99, 197)
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
axes[1].set_ylabel("Total cost ($, placeholder units) on validation set")
axes[1].set_title(f"Cost vs Threshold\n(FN=${COST_FN:.0f}, FP=${COST_FP:.0f} -- placeholders, not INR)")
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUT}/evaluation_plots_fraud.png", dpi=150)
print(f"Saved {OUT}/evaluation_plots_fraud.png")