"""
serve.py
========
FastAPI scoring service for the primary Return-Risk Scorer.

SECURITY / DEFENSE-ONLY NOTE (read before deploying):
  This project's own scope explicitly excludes "anything that would let an
  end user infer the exact decision threshold or reverse-engineer which
  features trigger a flag" (see PROJECT_CONTEXT.md). A single endpoint that
  always returns the raw score, calibrated score, and exact threshold to
  WHOEVER is placing an order would let an adversarial caller send repeated
  probe orders and back out the exact decision boundary -- exactly what the
  buildathon's rules disqualify.

  This service resolves that with two response shapes from the same
  endpoint, controlled by `reviewer_view`:
    - reviewer_view=False (default): what a live checkout backend would
      see. No raw score, no threshold, no calibrated probability -- just a
      coarse decision and coarse reason codes.
    - reviewer_view=True: the fuller response (score, calibrated_score,
      threshold_used, reason_codes) intended ONLY for the internal
      Streamlit reviewer demo (Day 5-6), not for checkout-facing traffic.
      In a real deployment this flag would sit behind internal
      authentication, not be a client-settable query parameter -- it is
      left open here for demo convenience given the buildathon timeline.
      State this trade-off explicitly in the README; don't let it be
      discovered for the first time during a panel question.

REASON CODES: computed from an importance-ranked, direction-aware lookup
(train.py's reason_code_reference.json), NOT SHAP. This avoids an
unverified library dependency (SHAP's TreeExplainer support for
HistGradientBoostingClassifier was not verified before writing this code)
and keeps reason codes at the same coarse level as FICO's long-standing
credit-score "adverse action" codes -- informative without exposing exact
model internals.
"""
import json
from typing import Literal, Optional

import joblib
import pandas as pd
from fastapi import FastAPI, Query
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Match this to wherever your train.py actually writes outputs/. Your own
# console output implies a relative "outputs/" folder -- adjust if your
# local setup differs (same path caveat as everywhere else in this project).
# ---------------------------------------------------------------------------
OUT = "outputs"

model = joblib.load(f"{OUT}/model.joblib")
calibrator = joblib.load(f"{OUT}/calibrator.joblib")
feature_cols = json.load(open(f"{OUT}/feature_columns.json"))
reason_reference = json.load(open(f"{OUT}/reason_code_reference.json"))
summary = json.load(open(f"{OUT}/summary.json"))
THRESHOLD = summary["best_threshold"]

# Keep in sync with generate_data.py's constants -- would move to a shared
# config file once real data replaces the synthetic generator. Duplicated
# here deliberately rather than imported, to keep serve.py runnable without
# needing generate_data.py's data-generation dependencies at serving time.
APPAREL_CATEGORIES = {"Fashion", "Footwear"}

# Human-readable labels for reason codes. Direction (does a HIGH or LOW
# value of this feature indicate elevated risk) was computed once from
# training data -- see reason_code_reference.json. These labels already
# assume that direction matches what generate_data.py's documented
# generative model intends. If a rerun of train.py flips a feature's
# direction, the corresponding label below may read oddly -- check
# reason_code_reference.json's "direction" field if a reason code looks
# wrong during testing.

# Real signal features in this data cluster at importance >= ~0.0067
# (price) and up; everything below that (customer_prior_orders at 0.0004,
# individual category dummies other than is_apparel, minority pincode
# tiers, payment_mode_Prepaid) is near-zero or even negative -- see
# reason_code_reference.json. Reporting those as "reasons" would present
# noise as explanation. This cutoff is a judgment call based on the actual
# gap in your data, not an arbitrary round number -- revisit if you retrain
# on different data and the importance spread looks different.
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

    NOTE: pd.get_dummies on a single row would only create dummy columns
    for THAT row's own category/payment_mode/tier -- it has no way to know
    about the other categories the model was trained on. Reindexing against
    feature_columns.json (saved at training time) fills every missing dummy
    with 0, guaranteeing the exact column set and order the model expects.
    is_apparel is derived server-side from category rather than accepted as
    a raw input field, so a caller can't submit an inconsistent combination
    (e.g. category="Fashion" with is_apparel=0).
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
    }
    df = pd.DataFrame([row])
    return df.reindex(columns=feature_cols, fill_value=0)


def get_reason_codes(feature_row: pd.DataFrame, top_k: int = 3) -> list:
    """Importance-ranked, direction-aware reason codes for one order.

    Ranks features by GLOBAL permutation importance (computed once at
    training time), then checks whether THIS order's value for each feature
    sits on the risk-elevating side of that feature's train-set midpoint
    (the midpoint between average values among returned vs. non-returned
    training orders). Returns up to top_k matching labels, most important
    first. This is a coarse, defensible explanation -- not a literal
    contribution score, by design (see module docstring).
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