import streamlit as st
import os
import sys

# 🔥 Add project root to Python path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from modules.farm_buy_matcher import AgriMatchingEngine

st.set_page_config(page_title="AgriLink AI - Matching Engine", layout="wide")

st.title("🤝 AgriLink AI — Smart Buyer-Farmer Matching")

engine = AgriMatchingEngine()
# ─── Mode Selection ───────────────────────────────────────

mode = st.radio(
    "Select Mode",
    ["👨‍🌾 Farmer → Find Buyers", "🏢 Buyer → Find Farmers"]
)

# ─────────────────────────────────────────────────────────
# 👨‍🌾 FARMER VIEW
# ─────────────────────────────────────────────────────────

if mode == "👨‍🌾 Farmer → Find Buyers":

    st.subheader("📦 Farmer Details")

    col1, col2 = st.columns(2)

    with col1:
        crop = st.text_input("Crop", "wheat")
        qty = st.number_input("Quantity (kg)", 100, 10000, 800)

    with col2:
        lat = st.number_input("Latitude", value=26.46)
        lon = st.number_input("Longitude", value=79.51)

    market_price = st.number_input("Current Market Price (₹/kg)", value=24.5)

    if st.button("🔍 Find Best Buyers"):

        results = engine.get_best_buyers_for_farmer(
            crop, qty, lat, lon, market_price
        )

        st.success("✅ Top Matches Found")

        for r in results[:5]:
            with st.container():
                st.markdown("---")
                st.subheader(f"🏢 {r['buyer_name']}")

                col1, col2, col3 = st.columns(3)

                col1.metric("Match Score", f"{r['score']}/100")
                col2.metric("Offered Price", f"₹{r['offered_price']}/kg")
                col3.metric("Distance", f"{r['distance_km']} km")

# ─────────────────────────────────────────────────────────
# 🏢 BUYER VIEW
# ─────────────────────────────────────────────────────────

else:

    st.subheader("🏢 Buyer Details")

    buyer_id = st.text_input("Buyer ID", "B001")

    st.markdown("### 📊 Market Prices (for comparison)")

    wheat_price = st.number_input("Wheat Price", value=24.5)
    tomato_price = st.number_input("Tomato Price", value=18.0)
    onion_price = st.number_input("Onion Price", value=22.0)

    if st.button("🔍 Find Best Farmers"):

        current_prices = {
            "wheat": wheat_price,
            "tomato": tomato_price,
            "onion": onion_price
        }

        results = engine.get_best_farmers_for_buyer(
            buyer_id, current_prices
        )

        st.success("✅ Top Farmer Matches Found")

        for r in results[:5]:
            with st.container():
                st.markdown("---")
                st.subheader(f"👨‍🌾 {r['farmer_name']}")

                col1, col2, col3 = st.columns(3)

                col1.metric("Match Score", f"{r['score']}/100")
                col2.metric("Quantity", f"{r['quantity']} kg")
                col3.metric("Distance", f"{r['distance_km']} km")

                st.caption(f"Crop: {r['crop']}")

# ─── Footer ─────────────────────────────────────────────

st.markdown("---")
st.markdown("🚀 Powered by AgriLink AI Matching Engine")