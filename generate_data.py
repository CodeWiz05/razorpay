"""
generate_data.py
=================
Generates a SYNTHETIC but realistically-calibrated India D2C/e-commerce
order dataset with a `returned` label, for building a return-risk scorer.

WHY SYNTHETIC: no auth-free public dataset exists with India-specific
COD/prepaid + return-flag granularity. See README Section 3 [data-
calibration--citations] for the full citation trail behind every
assumption below; swap this script for a real extract later and
everything downstream (features.py, train.py, evaluate.py) is unchanged.
This is a design choice, not a claim about real-world exact rates.
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

# BRACKETING FEATURE: apparel x prepaid interaction (COD buyers reject at
# the doorstep without pre-committing; prepaid buyers pre-pay, making
# multi-size ordering-and-returning the rational hedge). See README
# Section 3.2 [bracketing-feature] for the calibration and citation trail.
BRACKET_INTERCEPT = -4.5
BRACKET_APPAREL_COEF = 0.3
BRACKET_PREPAID_COEF = 0.3
BRACKET_INTERACTION_COEF = 4.43
BRACKET_RETURN_BOOST = 1.1
COD_COEF = 2.75
N_LISTINGS_PER_CATEGORY = 80  # documented assumption, not cited -- see README Section 3.10
LISTING_QUALITY_SIGMA = 0.8   # documented assumption, not cited -- see README Section 3.10


def generate():
    customer_ids = RNG.integers(0, N_CUSTOMERS, size=N_ORDERS)
    order_dates = pd.to_datetime(
        START_DATE.value
        + RNG.integers(0, (END_DATE - START_DATE).value, size=N_ORDERS)
    )

    categories = RNG.choice(CATEGORIES, size=N_ORDERS, p=CATEGORY_PROBS)
    is_apparel = np.isin(categories, list(APPAREL_CATS)).astype(int)
        # ---- LISTING-LEVEL IDENTITY: SKU + hidden quality/fit effect ----
    # Category alone can't capture within-category return-rate variance
    # (sizing, material, photo accuracy). Modeled as a hidden per-listing
    # logit offset the model NEVER sees directly -- only the expanding
    # historical return rate computed below (same architecture as
    # customer_past_return_rate, but with no feedback loop into base_prob:
    # this is a pure proxy of an already-latent cause, not a second causal
    # channel). See README Section 3.10 [listing-level-heterogeneity].
    listing_num = RNG.integers(0, N_LISTINGS_PER_CATEGORY, size=N_ORDERS)
    product_id = pd.Series(categories).astype(str) + "_" + pd.Series(listing_num).astype(str)

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
        "product_id": product_id.values,
        "price": price,
        "discount_pct": discount_pct,
        "payment_mode": payment_mode,
        "pincode_tier": tier,
        "delivery_days": delivery_days,
    }).sort_values("order_date").reset_index(drop=True)

    # ---- Expanding, leakage-safe "customer past return rate" ----
    # Computed further below once we know returns; for now placeholder.

    # ---- Generate ground-truth return probability (the "true" data-
    #generating process; the model never sees this directly) ----
    is_cod = (df["payment_mode"] == "COD").astype(int)
    is_prepaid = (df["payment_mode"] == "Prepaid").astype(int)
    tier3_flag = (df["pincode_tier"] == "Tier3").astype(int)

    # CATEGORY RISK: graded per-category term (replaces binary is_apparel).
    # Electronics/Home/Beauty miss their cited floor by 3-5pts (documented,
    # unresolved -- shared-intercept correction didn't converge). See
    # README Section 3.3 [category-level-return-rates] for the full
    # citation trail and the diagnosed cause. is_apparel is retained as a
    # column for reason-code readability but no longer drives the
    # generative probability directly.
    CATEGORY_RISK_TERM = {
        "Grocery": 0.000, "Electronics": 0.663, "Home": 0.900,
        "Footwear": 0.588, "Beauty": 0.765, "Fashion": 1.179,
    }
    category_term = df["category"].map(CATEGORY_RISK_TERM)

    # Interaction term dominates by design -- apparel-alone and prepaid-alone
    # main effects are deliberately small so COD/non-apparel rarely trigger this.
    is_apparel_arr = df["is_apparel"].values
    unique_listings = df["product_id"].unique()
    listing_quality_effect = pd.Series(
        RNG.normal(0, LISTING_QUALITY_SIGMA, size=len(unique_listings)),
        index=unique_listings,
    )
    df["_listing_quality_effect"] = df["product_id"].map(listing_quality_effect)
    bracket_logit = (
        BRACKET_INTERCEPT
        + BRACKET_APPAREL_COEF * is_apparel_arr
        + BRACKET_PREPAID_COEF * is_prepaid
        + BRACKET_INTERACTION_COEF * is_apparel_arr * is_prepaid
    )
    bracket_prob = 1 / (1 + np.exp(-bracket_logit))
    is_bracketed = (RNG.random(N_ORDERS) < bracket_prob).astype(int)
    df["is_bracketed"] = is_bracketed
    df["size_variant_count"] = np.where(is_bracketed == 1, RNG.integers(2, 5, size=N_ORDERS), 1)

    # FESTIVE-PERIOD EFFECT: general (not COD-specific) uplift during known
    # Indian sale windows, conservatively calibrated below TrackVid's cited
    # upper bound. See README Section 3.9 [festive-period-effect] for the
    # citation and calibration result.
    FESTIVE_WINDOWS = [
        ("2025-08-10", "2025-08-20"), ("2025-10-01", "2025-10-25"),
        ("2025-12-20", "2025-12-31"), ("2026-01-20", "2026-01-30"),
        ("2026-07-01", "2026-07-15"), ("2026-08-10", "2026-08-20"),
    ]
    is_festive = pd.Series(False, index=df.index)
    for start, end in FESTIVE_WINDOWS:
        is_festive |= (df["order_date"] >= start) & (df["order_date"] <= end)
    df["is_festive"] = is_festive.astype(int)

    delay = (df["delivery_days"] - 3).clip(lower=0)
    delivery_term = 0.004 * delay + is_cod * 0.11 * delay

    hump = (1 - (df["price"] - 750).abs() / 450).clip(lower=0)
    price_term = 0.015 * hump + is_cod * 0.22 * hump

    logit = (
        -4.95
        + COD_COEF * is_cod
        + category_term
        - 0.001 * df["discount_pct"]
        + 0.25 * tier3_flag
        + delivery_term
        + price_term
        + 0.30 * df["is_festive"]
        + BRACKET_RETURN_BOOST * is_bracketed
        + df["_listing_quality_effect"]
    )
    base_prob = 1 / (1 + np.exp(-logit))

    # Sequentially assign returns per customer so we can build a genuine
    # expanding "past return rate" feature afterward without leakage.
    returned = np.zeros(N_ORDERS, dtype=int)
    cust_return_counts = {}
    cust_order_counts = {}
    past_return_rate = np.zeros(N_ORDERS)
    cust_order_seq = np.zeros(N_ORDERS, dtype=int)

    prod_return_counts = {}
    prod_order_counts = {}
    product_past_return_rate = np.zeros(N_ORDERS)

    for i in range(N_ORDERS):
        cid = df.at[i, "customer_id"]
        pid = df.at[i, "product_id"]

        prior_orders = cust_order_counts.get(cid, 0)
        prior_returns = cust_return_counts.get(cid, 0)
        prr = (prior_returns / prior_orders) if prior_orders >= 2 else 0.12
        past_return_rate[i] = prr
        cust_order_seq[i] = prior_orders

        # Product history: cold-start prior = overall target rate (~0.16),
        # min 3 prior orders before trusting the listing's own rate --
        # stricter than the customer threshold (2) since a single early
        # return on a low-volume SKU is noisier signal than on a customer.
        prod_prior_orders = prod_order_counts.get(pid, 0)
        prod_prior_returns = prod_return_counts.get(pid, 0)
        product_past_return_rate[i] = (
            prod_prior_returns / prod_prior_orders if prod_prior_orders >= 3 else 0.16
        )

        p = np.clip(base_prob[i] + 0.25 * (prr - 0.15), 0.01, 0.95)
        r = int(RNG.random() < p)
        returned[i] = r

        cust_order_counts[cid] = prior_orders + 1
        cust_return_counts[cid] = prior_returns + r
        prod_order_counts[pid] = prod_prior_orders + 1
        prod_return_counts[pid] = prod_prior_returns + r

    df["customer_prior_orders"] = cust_order_seq
    df["customer_past_return_rate"] = past_return_rate.round(3)
    df["product_past_return_rate"] = product_past_return_rate.round(3)
    df["returned"] = returned
    df = df.drop(columns=["_listing_quality_effect"])
    return df


if __name__ == "__main__":
    from pathlib import Path
    df = generate()
    out_path = str(Path(__file__).resolve().parent / "data_orders.csv")
    df.to_csv(out_path, index=False)
    print(f"Generated {len(df):,} orders -> {out_path}")
    print(f"Overall return rate: {df['returned'].mean():.2%}")
    print(df.groupby("payment_mode")["returned"].mean())
    print(df.groupby("is_apparel")["returned"].mean())