"""
generate_data.py
=================
Generates a SYNTHETIC but realistically-calibrated India D2C/e-commerce
order dataset with a `returned` label, for building a return-risk scorer.

WHY SYNTHETIC: No freely-downloadable, auth-free public dataset exists with
India-specific COD/prepaid + return-flag granularity. Rather than force-fit
a foreign dataset (Olist/UCI Online Retail) that lacks the COD signal which
dominates Indian return/RTO behavior, we generate data from a documented
generative model calibrated to publicly reported industry return-rate
ranges. This keeps the *evaluation methodology* honest even though the
*data* is synthetic -- swap this script for a real extract later and
everything downstream (features.py, train.py, evaluate.py) is unchanged.

CALIBRATION ASSUMPTIONS (documented so judges/reviewers can inspect them):
  - Overall return rate target: ~15-18% (industry-cited range for Indian
    fashion/apparel-heavy D2C, lower for electronics/grocery)
  - COD orders return/RTO at roughly 2.5-3x the rate of prepaid orders
    (COD removes the "already paid" commitment device)
  - Apparel/footwear (size-variant categories) return at a higher base
    rate than electronics/grocery/beauty (fit issues)
  - Heavier discounting correlates with higher return rate (impulse buys,
    lower perceived commitment)
  - Repeat customers with a history of returns are more likely to return
    again (serial returner effect) -- this feature MUST be computed as an
    expanding/rolling stat using only PRIOR orders to avoid leakage
  - Tier-3 pincodes have slightly higher RTO due to logistics friction

This is a design choice, not a claim about real-world exact rates -- state
this plainly in your writeup.
"""
import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)
N_ORDERS = 60_000
START_DATE = pd.Timestamp("2025-08-01")
END_DATE = pd.Timestamp("2026-08-01")  # 12 months of order history

CATEGORIES = ["Fashion", "Footwear", "Electronics", "Beauty", "Home", "Grocery"]
CATEGORY_PROBS = [0.30, 0.15, 0.15, 0.15, 0.15, 0.10]
APPAREL_CATS = {"Fashion", "Footwear"}
CATEGORY_BASE_PRICE = {
    "Fashion": 900, "Footwear": 1500, "Electronics": 4500,
    "Beauty": 700, "Home": 1800, "Grocery": 400,
}
TIERS = ["Tier1", "Tier2", "Tier3"]
TIER_PROBS = [0.45, 0.35, 0.20]

N_CUSTOMERS = 22_000


def generate():
    customer_ids = RNG.integers(0, N_CUSTOMERS, size=N_ORDERS)
    order_dates = pd.to_datetime(
        START_DATE.value
        + RNG.integers(0, (END_DATE - START_DATE).value, size=N_ORDERS)
    ).sort_values()

    categories = RNG.choice(CATEGORIES, size=N_ORDERS, p=CATEGORY_PROBS)
    is_apparel = np.isin(categories, list(APPAREL_CATS)).astype(int)

    base_price = np.array([CATEGORY_BASE_PRICE[c] for c in categories])
    price = np.round(base_price * RNG.lognormal(mean=0, sigma=0.5, size=N_ORDERS), 2)

    discount_pct = np.clip(RNG.beta(2, 6, size=N_ORDERS) * 100, 0, 80).round(1)

    # COD probability is higher for lower price, apparel, and tier-3 pincodes
    tier = RNG.choice(TIERS, size=N_ORDERS, p=TIER_PROBS)
    tier3 = (tier == "Tier3").astype(int)
    cod_logit = (
        -0.5
        + 0.9 * is_apparel
        + 0.7 * tier3
        - 0.00015 * price
        + 0.01 * discount_pct
    )
    cod_prob = 1 / (1 + np.exp(-cod_logit))
    payment_mode = np.where(RNG.random(N_ORDERS) < cod_prob, "COD", "Prepaid")

    delivery_days = np.clip(
        RNG.normal(loc=np.where(tier == "Tier1", 2.5, np.where(tier == "Tier2", 4, 6)),
                    scale=1.2, size=N_ORDERS), 1, 15
    ).round(0)

    df = pd.DataFrame({
        "order_id": np.arange(N_ORDERS),
        "customer_id": customer_ids,
        "order_date": order_dates.values,
        "category": categories,
        "is_apparel": is_apparel,
        "price": price,
        "discount_pct": discount_pct,
        "payment_mode": payment_mode,
        "pincode_tier": tier,
        "delivery_days": delivery_days,
    }).sort_values("order_date").reset_index(drop=True)

    # ---- Expanding, leakage-safe "customer past return rate" ----
    # Computed further below once we know returns; for now placeholder.

    # ---- Generate ground-truth return probability (the "true" data-
    #      generating process; the model never sees this directly) ----
    is_cod = (df["payment_mode"] == "COD").astype(int)
    tier3_flag = (df["pincode_tier"] == "Tier3").astype(int)

    logit = (
        -3.0
        + 1.15 * is_cod
        + 0.55 * df["is_apparel"]
        + 0.012 * df["discount_pct"]
        + 0.25 * tier3_flag
        + 0.06 * (df["delivery_days"] - 3).clip(lower=0)
        - 0.00004 * df["price"]
    )
    base_prob = 1 / (1 + np.exp(-logit))

    # Sequentially assign returns per customer so we can build a genuine
    # expanding "past return rate" feature afterward without leakage.
    returned = np.zeros(N_ORDERS, dtype=int)
    cust_return_counts = {}
    cust_order_counts = {}
    past_return_rate = np.zeros(N_ORDERS)
    cust_order_seq = np.zeros(N_ORDERS, dtype=int)

    for i in range(N_ORDERS):
        cid = df.at[i, "customer_id"]
        prior_orders = cust_order_counts.get(cid, 0)
        prior_returns = cust_return_counts.get(cid, 0)
        prr = (prior_returns / prior_orders) if prior_orders >= 2 else 0.12  # cold-start prior
        past_return_rate[i] = prr
        cust_order_seq[i] = prior_orders

        # serial-returner effect layered on top of base_prob
        p = np.clip(base_prob[i] + 0.25 * (prr - 0.15), 0.01, 0.95)
        r = int(RNG.random() < p)
        returned[i] = r

        cust_order_counts[cid] = prior_orders + 1
        cust_return_counts[cid] = prior_returns + r

    df["customer_prior_orders"] = cust_order_seq
    df["customer_past_return_rate"] = past_return_rate.round(3)
    df["returned"] = returned

    return df


if __name__ == "__main__":
    df = generate()
    out_path = "/home/claude/return_risk/data/orders.csv"
    df.to_csv(out_path, index=False)
    print(f"Generated {len(df):,} orders -> {out_path}")
    print(f"Overall return rate: {df['returned'].mean():.2%}")
    print(df.groupby("payment_mode")["returned"].mean())
    print(df.groupby("is_apparel")["returned"].mean())