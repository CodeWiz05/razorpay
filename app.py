"""
app.py
======
Streamlit reviewer demo for the Return-Risk Scorer -- this is what the
pitch video's live-demo segment shows.

RUN ORDER MATTERS -- two processes, two terminals:
    Terminal 1:  uvicorn serve:app --reload --port 8000
    Terminal 2:  streamlit run app.py

WHY TWO SIDE-BY-SIDE VIEWS: serve.py deliberately returns a different
response shape depending on `reviewer_view` (see serve.py's docstring for
the defense-only reasoning -- a checkout-facing caller never sees the raw
score or exact threshold, only a coarse decision). This demo shows BOTH
side by side specifically so a reviewer can see that distinction working
live, rather than just reading about it.
"""
import requests
import streamlit as st

API_URL = "http://localhost:8000/score"

st.set_page_config(page_title="Return-Risk Scorer", layout="wide")
st.title("Return-Risk Scorer -- Reviewer Demo")
st.caption(
    "Scores a synthetic order for return/RTO risk at time of placement. "
    "This is a verifier, not an auto-blocker -- see the defense-only note at the bottom."
)

CATEGORIES = ["Fashion", "Footwear", "Electronics", "Beauty", "Home", "Grocery"]
PAYMENT_MODES = ["COD", "Prepaid"]
PINCODE_TIERS = ["Tier1", "Tier2", "Tier3"]

# Preset example orders -- for a live demo/recording, filling 8 form fields
# on camera burns time and risks fumbling. Adjust these values if you find
# better examples once exploring reason_code_reference.json further.
PRESETS = {
    "-- Fill manually --": None,
    "High-risk example (COD, apparel, high discount, Tier3)": {
        "category": "Fashion", "price": 899.0, "discount_pct": 60.0,
        "payment_mode": "COD", "pincode_tier": "Tier3", "delivery_days": 6.0,
        "customer_prior_orders": 2, "customer_past_return_rate": 0.4,
    },
    "Low-risk example (Prepaid, electronics, no discount, Tier1)": {
        "category": "Electronics", "price": 4500.0, "discount_pct": 5.0,
        "payment_mode": "Prepaid", "pincode_tier": "Tier1", "delivery_days": 2.0,
        "customer_prior_orders": 5, "customer_past_return_rate": 0.0,
    },
}

preset_choice = st.selectbox("Load an example order", list(PRESETS.keys()))
preset = PRESETS[preset_choice]

st.subheader("Order details")
col1, col2 = st.columns(2)

# NOTE: every widget's `key` includes preset_choice. Without this, switching
# the preset dropdown would NOT update these fields -- Streamlit only uses
# `value=` the first time a widget is created and trusts its own state after
# that. Tying the key to preset_choice forces a fresh widget (and therefore
# a fresh default) whenever a different preset is picked, while still
# letting you freely hand-edit a field without it resetting on unrelated
# reruns.
with col1:
    category = st.selectbox(
        "Category", CATEGORIES,
        index=CATEGORIES.index(preset["category"]) if preset else 0,
        key=f"category_{preset_choice}",
    )
    price = st.number_input(
        "Price (Rs.)", min_value=1.0,
        value=preset["price"] if preset else 1000.0,
        key=f"price_{preset_choice}",
    )
    discount_pct = st.slider(
        "Discount %", 0.0, 100.0,
        preset["discount_pct"] if preset else 10.0,
        key=f"discount_{preset_choice}",
    )
    payment_mode = st.selectbox(
        "Payment mode", PAYMENT_MODES,
        index=PAYMENT_MODES.index(preset["payment_mode"]) if preset else 0,
        key=f"payment_{preset_choice}",
    )

with col2:
    pincode_tier = st.selectbox(
        "Pincode tier", PINCODE_TIERS,
        index=PINCODE_TIERS.index(preset["pincode_tier"]) if preset else 0,
        key=f"tier_{preset_choice}",
    )
    delivery_days = st.number_input(
        "Delivery days", min_value=1.0,
        value=preset["delivery_days"] if preset else 3.0,
        key=f"delivery_{preset_choice}",
    )
    customer_prior_orders = st.number_input(
        "Customer's prior orders", min_value=0, step=1,
        value=preset["customer_prior_orders"] if preset else 0,
        key=f"prior_orders_{preset_choice}",
    )
    customer_past_return_rate = st.slider(
        "Customer's past return rate", 0.0, 1.0,
        preset["customer_past_return_rate"] if preset else 0.12,
        key=f"past_rate_{preset_choice}",
    )

order_payload = {
    "category": category,
    "price": price,
    "discount_pct": discount_pct,
    "payment_mode": payment_mode,
    "pincode_tier": pincode_tier,
    "delivery_days": delivery_days,
    "customer_prior_orders": customer_prior_orders,
    "customer_past_return_rate": customer_past_return_rate,
}

if st.button("Score this order", type="primary"):
    try:
        checkout_resp = requests.post(API_URL, params={"reviewer_view": False}, json=order_payload, timeout=5)
        reviewer_resp = requests.post(API_URL, params={"reviewer_view": True}, json=order_payload, timeout=5)
        checkout_resp.raise_for_status()
        reviewer_resp.raise_for_status()
    except requests.exceptions.ConnectionError:
        st.error(
            "Can't reach the scoring API. Is it running?\n\n"
            "Start it in a separate terminal with:\n"
            "`uvicorn serve:app --reload --port 8000`"
        )
        st.stop()
    except requests.exceptions.RequestException as e:
        st.error(f"Request failed: {e}")
        if e.response is not None:
            st.code(e.response.text)
        st.stop()

    checkout_data = checkout_resp.json()
    reviewer_data = reviewer_resp.json()

    st.divider()
    view_col1, view_col2 = st.columns(2)

    with view_col1:
        st.subheader("What the checkout backend sees")
        st.caption("Default response -- no score, no threshold. See serve.py's defense-only note.")
        decision = checkout_data["decision"]
        if decision == "review":
            st.warning(f"Decision: **{decision.upper()}**")
        else:
            st.success(f"Decision: **{decision.upper()}**")
        if checkout_data["reason_codes"]:
            st.write("Reason codes:")
            for code in checkout_data["reason_codes"]:
                st.write(f"- {code}")
        else:
            st.write("No reason codes (order not flagged).")

    with view_col2:
        st.subheader("What the internal reviewer sees")
        st.caption("reviewer_view=true -- full transparency, for evaluation/judging only.")
        m1, m2, m3 = st.columns(3)
        m1.metric("Raw score", reviewer_data["score"])
        m2.metric("Calibrated score", reviewer_data["calibrated_score"])
        m3.metric("Threshold used", reviewer_data["threshold_used"])
        st.write(f"Decision: **{reviewer_data['decision'].upper()}**")
        if reviewer_data["reason_codes"]:
            st.write("Reason codes:")
            for code in reviewer_data["reason_codes"]:
                st.write(f"- {code}")
        else:
            st.write("No reason codes.")

st.divider()
with st.expander("Defense-only statement"):
    st.write(
        "This system is a verifier, not an auto-blocker. It never auto-rejects "
        "an order, and the checkout-facing response above deliberately withholds "
        "the exact score and threshold, so a caller can't probe the API to "
        "reverse-engineer the decision boundary. Any production use requires a "
        "human or downstream policy layer between a score and an action."
    )