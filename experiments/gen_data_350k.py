"""
generate_data_350k.py
=======================
SCALING EXPERIMENT: does a larger prepaid-positive pool let the Prepaid
model actually learn something, or is the signal itself too thin
regardless of volume?

Scaled to 350,000 orders. Customer pool scaled proportionally (22,000 ->
~128,000) to preserve the same ~2.73 orders/customer/year ratio as the
production 60k dataset -- NOT scaled naively, which would have given
customers unrealistically deep purchase histories within the same
12-month window (the trap flagged earlier: leaving N_CUSTOMERS fixed
while growing N_ORDERS would silently distort customer_past_return_rate).

COEFFICIENTS: identical to the current production generate_data.py
(post COD-ratio recalibration, post price/delivery COD-specific split).
Deliberately unchanged, so this experiment isolates "does more data
help" from "did we also change the generative model."

OUTPUT: writes to data_orders_350k.csv, a SEPARATE file. Does not touch
the production data_orders.csv or generate_data.py.

NOTE ON SCALE CHOICE: 350,000 was chosen as a stress-test scale to check
robustness, not calibrated against a specific named real-world benchmark
-- stated plainly here so it isn't mistaken for a sourced figure the way
the COD ratio or price bands were.
"""
import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)
N_ORDERS = 350_000
N_CUSTOMERS = 128_000  # scaled proportionally: 22,000 * (350,000/60,000)
START_DATE = pd.Timestamp("2025-08-01")
END_DATE = pd.Timestamp("2026-08-01")  # SAME 12-month window, not extended

CATEGORIES = ["Fashion", "Footwear", "Electronics", "Beauty", "Home", "Grocery"]
CATEGORY_PROBS = [0.30, 0.15, 0.15, 0.15, 0.15, 0.10]
APPAREL_CATS = {"Fashion", "Footwear"}
CATEGORY_BASE_PRICE = {
    "Fashion": 900, "Footwear": 1500, "Electronics": 4500,
    "Beauty": 700, "Home": 1800, "Grocery": 400,
}
TIERS = ["Tier1", "Tier2", "Tier3"]
TIER_PROBS = [0.45, 0.35, 0.20]


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

    tier = RNG.choice(TIERS, size=N_ORDERS, p=TIER_PROBS)
    tier3 = (tier == "Tier3").astype(int)
    cod_logit = (
        -0.5 + 0.9 * is_apparel + 0.7 * tier3 - 0.00015 * price + 0.01 * discount_pct
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

    is_cod = (df["payment_mode"] == "COD").astype(int)
    tier3_flag = (df["pincode_tier"] == "Tier3").astype(int)

    # IDENTICAL to production generate_data.py -- delivery + price terms
    # split into generic (all orders, unverified-for-prepaid) + COD-specific
    # (Shipway-calibrated) components.
    delay = (df["delivery_days"] - 3).clip(lower=0)
    delivery_term = 0.004 * delay + is_cod * 0.11 * delay

    hump = (1 - (df["price"] - 750).abs() / 450).clip(lower=0)
    price_term = 0.015 * hump + is_cod * 0.22 * hump

    logit = (
        -3.781
        + 1.90 * is_cod
        + 0.55 * df["is_apparel"]
        + 0.012 * df["discount_pct"]
        + 0.25 * tier3_flag
        + delivery_term
        + price_term
    )
    base_prob = 1 / (1 + np.exp(-logit))

    returned = np.zeros(N_ORDERS, dtype=int)
    cust_return_counts = {}
    cust_order_counts = {}
    past_return_rate = np.zeros(N_ORDERS)
    cust_order_seq = np.zeros(N_ORDERS, dtype=int)

    for i in range(N_ORDERS):
        cid = df.at[i, "customer_id"]
        prior_orders = cust_order_counts.get(cid, 0)
        prior_returns = cust_return_counts.get(cid, 0)
        prr = (prior_returns / prior_orders) if prior_orders >= 2 else 0.12
        past_return_rate[i] = prr
        cust_order_seq[i] = prior_orders

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
    out_path = "C:/Numair/Coding/Razorpay/data_orders_350k.csv"
    df.to_csv(out_path, index=False)
    print(f"Generated {len(df):,} orders -> {out_path}")
    print(f"Customers: {N_CUSTOMERS:,}  (orders/customer = {N_ORDERS/N_CUSTOMERS:.2f}, "
          f"vs production's {60000/22000:.2f})")
    print(f"Overall return rate: {df['returned'].mean():.2%}")
    print(df.groupby("payment_mode")["returned"].mean())
    print(f"\nPrepaid positive count (full data): {df[df.payment_mode=='Prepaid']['returned'].sum():,}")
    print(f"COD positive count (full data):      {df[df.payment_mode=='COD']['returned'].sum():,}")