import streamlit as st
import sys
import os

# 🔥 FORCE add project root to Python path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from modules.price_predictor import predict_price_realtime

st.set_page_config(page_title="AgriLink AI - Price Predictor", layout="centered")

st.title("🌾 AgriLink AI — Real-Time Price Predictor")
st.markdown("Predict mandi prices using real-world agricultural data")

# ─── Input Form ─────────────────────────────────────────────

st.subheader("📥 Enter Crop Details")

state = st.text_input("State", placeholder="e.g., Maharashtra")
district = st.text_input("District", placeholder="e.g., Nashik")
market = st.text_input("Market", placeholder="e.g., Lasalgaon")
commodity = st.text_input("Commodity (Crop)", placeholder="e.g., Onion")
variety = st.text_input("Variety", placeholder="e.g., Red")
month = st.slider("Month", 1, 12, 6)

# ─── Predict Button ─────────────────────────────────────────

if st.button("🔍 Predict Price"):
    if not all([state, district, market, commodity, variety]):
        st.warning("⚠️ Please fill all fields")
    else:
        try:
            result = predict_price_realtime(
                state, district, market, commodity, variety, month
            )

            # ─── Output Display ─────────────────────────────

            st.success("✅ Prediction Successful!")

            st.metric(
                label="💰 Predicted Price",
                value=f"₹{result['predicted_price']}/kg"
            )

            col1, col2 = st.columns(2)

            with col1:
                st.info(f"📉 Min Price: ₹{result['range']['min']}/kg")

            with col2:
                st.info(f"📈 Max Price: ₹{result['range']['max']}/kg")

            # Visualization
            st.subheader("📊 Price Range Visualization")
            st.bar_chart({
                "Price": [
                    result['range']['min'],
                    result['predicted_price'],
                    result['range']['max']
                ]
            })

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

# ─── Footer ─────────────────────────────────────────────

st.markdown("---")
st.markdown("Built with ❤️ for farmers using real mandi data")