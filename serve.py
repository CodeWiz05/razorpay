"""
serve.py
========
FastAPI scoring service for the primary Return-Risk Scorer.

DEFENSE-ONLY: never expose raw score / calibrated score / exact threshold
to checkout-facing traffic -- an adversarial caller could probe repeatedly
and back out the decision boundary. Two response shapes, one endpoint,
controlled by `reviewer_view` (default False = checkout-facing, coarse
only). Full reasoning in README Section 4 [defense-only-statement] --
do not change this without reading that section first.

REASON CODES: importance-ranked, direction-aware (train.py's
reason_code_reference.json), not SHAP -- see README Section 2
[methodology].
"""
import json
from typing import Literal, Optional

import joblib
import pandas as pd
from fastapi import FastAPI, Query
from pydantic import BaseModel, Field

# Adjust if train.py's OUT differs from this relative "outputs/" folder.
OUT = "outputs"

model = joblib.load(f"{OUT}/model.joblib")
calibrator = joblib.load(f"{OUT}/calibrator.joblib")
feature_cols = json.load(open(f"{OUT}/feature_columns.json"))
reason_reference = json.load(open(f"{OUT}/reason_code_reference.json"))
summary = json.load(open(f"{OUT}/summary.json"))
THRESHOLD = summary["best_threshold"]

# Duplicated from generate_data.py (not imported, so serve.py runs without
# data-generation deps) -- keep in sync if that file's constants change.
APPAREL_CATEGORIES = {"Fashion", "Footwear"}

# Direction (high/low = elevated risk) is fixed from reason_code_reference.json
# at training time -- if a retrain flips a feature's direction, check that
# file's "direction" field if a label looks wrong.
# MIN_IMPORTANCE is a judgment call based on the real gap in permutation
# importance for this dataset, not an arbitrary round number -- revisit if
# retraining shows a different spread (see reason_code_reference.json).
MIN_IMPORTANCE = 0.005
LABELS = {
    "price": "Low order value",           # direction="low" in reference data
    "discount_pct": "High discount percentage",  # direction="high"
    "delivery_days": "Long delivery window",     # direction="high"
    "is_apparel": "Apparel/footwear category",
    "customer_prior_orders": "Limited order history",
    "customer_past_return_rate": "Customer's past return rate",
    "payment_mode_COD": "Cash-on-delivery order",
    "payment_mode_Prepaid": "Prepaid order",
    "pincode_tier_Tier3": "Tier-3 delivery area",
    "pincode_tier_Tier2": "Tier-2 delivery area",
    "pincode_tier_Tier1": "Tier-1 delivery area",
    "category_Fashion": "Fashion category",
    "category_Footwear": "Footwear category",
    "category_Electronics": "Electronics category",
    "category_Beauty": "Beauty category",
    "category_Home": "Home category",
    "category_Grocery": "Grocery category",
    "is_bracketed": "Multi-size/color bracket order",
    "size_variant_count": "Multiple size/color variants ordered together",
    "is_festive": "Order placed during a festive sale period",
    "product_past_return_rate": "This product's own history of returns",
}

app = FastAPI(title="Return-Risk Scorer")


class Order(BaseModel):
    category: Literal["Fashion", "Footwear", "Electronics", "Beauty", "Home", "Grocery"]
    price: float = Field(gt=0)
    discount_pct: float = Field(ge=0, le=100)
    payment_mode: Literal["COD", "Prepaid"]
    pincode_tier: Literal["Tier1", "Tier2", "Tier3"]
    delivery_days: float = Field(gt=0)
    customer_prior_orders: int = Field(ge=0)
    customer_past_return_rate: float = Field(ge=0, le=1)
    is_bracketed: bool = Field(default=False)
    size_variant_count: int = Field(default=1, ge=1)
    order_date: Optional[str] = Field(default=None, description="ISO date; omit to assume non-festive")
    product_past_return_rate: float = Field(default=0.16, ge=0, le=1)  # cold-start prior, matches generate_data.py

# Duplicated from generate_data.py, same reasoning as APPAREL_CATEGORIES above
FESTIVE_WINDOWS = [
    ("2025-08-10", "2025-08-20"), ("2025-10-01", "2025-10-25"),
    ("2025-12-20", "2025-12-31"), ("2026-01-20", "2026-01-30"),
    ("2026-07-01", "2026-07-15"), ("2026-08-10", "2026-08-20"),
]

def compute_is_festive(order_date_str):
    if not order_date_str:
        return 0
    d = pd.Timestamp(order_date_str)
    return int(any(pd.Timestamp(s) <= d <= pd.Timestamp(e) for s, e in FESTIVE_WINDOWS))

def build_feature_row(order: Order) -> pd.DataFrame:
    """Builds a single-row feature vector matching training-time encoding.

    Reindexes against feature_columns.json (saved at training time) so a
    single row's dummy columns match the full training column set exactly.
    is_apparel is derived server-side from category, not accepted as input,
    so a caller can't submit an inconsistent combination.
    """
    is_apparel = int(order.category in APPAREL_CATEGORIES)
    row = {
        "price": order.price,
        "discount_pct": order.discount_pct,
        "delivery_days": order.delivery_days,
        "is_apparel": is_apparel,
        "customer_prior_orders": order.customer_prior_orders,
        "customer_past_return_rate": order.customer_past_return_rate,
        f"category_{order.category}": 1,
        f"payment_mode_{order.payment_mode}": 1,
        f"pincode_tier_{order.pincode_tier}": 1,
        "is_bracketed": int(order.is_bracketed),
        "size_variant_count": order.size_variant_count,
        "is_festive": compute_is_festive(order.order_date),
        "product_past_return_rate": order.product_past_return_rate,
    }
    df = pd.DataFrame([row])
    return df.reindex(columns=feature_cols, fill_value=0)


def get_reason_codes(feature_row: pd.DataFrame, top_k: int = 3) -> list:
    """Importance-ranked, direction-aware reason codes for one order.

    Ranks by global permutation importance, checks whether this order's
    value sits on the risk-elevating side of the train-set midpoint for
    that feature. Coarse and defensible by design, not a literal
    contribution score -- see module docstring.
    """
    ranked = sorted(reason_reference.items(), key=lambda kv: -kv[1]["importance"])
    codes = []
    for feat, info in ranked:
        if info["importance"] < MIN_IMPORTANCE:
            continue
        if feat not in feature_row.columns:
            continue
        value = feature_row[feat].iloc[0]
        midpoint = (info["mean_returned"] + info["mean_not_returned"]) / 2
        elevated = (value > midpoint) if info["direction"] == "high" else (value < midpoint)
        if elevated:
            codes.append(LABELS.get(feat, feat))
        if len(codes) == top_k:
            break
    return codes


@app.post("/score")
def score_order(order: Order, reviewer_view: bool = Query(False)):
    feature_row = build_feature_row(order)
    raw_score = float(model.predict_proba(feature_row)[:, 1][0])
    calibrated_score = float(calibrator.predict([raw_score])[0])

    # Decision uses the CALIBRATED score against THRESHOLD -- train.py now
    # calibrates BEFORE threshold selection (fix applied [date]; see README
    # methodology section), so best_threshold in summary.json is on the
    # calibrated scale. Comparing raw_score against it would silently
    # reintroduce the exact ordering bug that fix closed.
    flagged = calibrated_score >= THRESHOLD
    decision = "review" if flagged else "approve"
    reason_codes = get_reason_codes(feature_row) if (flagged or reviewer_view) else []

    if reviewer_view:
        return {
            "score": round(raw_score, 4),
            "calibrated_score": round(calibrated_score, 4),
            "decision": decision,
            "threshold_used": THRESHOLD,
            "reason_codes": reason_codes,
        }
    # Default / checkout-facing response: no raw probabilities, no threshold.
    return {"decision": decision, "reason_codes": reason_codes}


@app.get("/health")
def health():
    return {"status": "ok"}