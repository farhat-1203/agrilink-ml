import streamlit as st
import sys
import os

# 🔥 Add project root to Python path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from modules.demand_forecaster import predict_demand_forecast

st.set_page_config(page_title="AgriLink AI - Demand Forecast", layout="centered")

st.title("📊 AgriLink AI — Demand Forecasting")
st.markdown("Predict future mandi demand using real-world Agmarknet data")

# ─── Input Section ─────────────────────────────────────────

st.subheader("📥 Enter Details")

crop = st.text_input("Crop Name", placeholder="e.g., Wheat, Tomato, Onion")
days = st.slider("Forecast Days", min_value=3, max_value=14, value=7)

# ─── Predict Button ────────────────────────────────────────

if st.button("📈 Forecast Demand"):
    if not crop:
        st.warning("⚠️ Please enter a crop name")
    else:
        result = predict_demand_forecast(crop, days)

        if "error" in result:
            st.error(f"❌ {result['error']}")
        else:
            st.success("✅ Forecast Generated Successfully!")

            # ─── Trend Display ────────────────────────────
            trend = result["trend"]

            if trend == "Rising":
                st.metric("📈 Demand Trend", trend, delta="Increasing")
            elif trend == "Falling":
                st.metric("📉 Demand Trend", trend, delta="Decreasing")
            else:
                st.metric("➡️ Demand Trend", trend, delta="Stable")

            # ─── Forecast Table ───────────────────────────
            st.subheader("📅 Daily Forecast")

            forecast_df = {
                "Date": [d["date"] for d in result["forecast"]],
                "Day": [d["day"] for d in result["forecast"]],
                "Demand (Quintals)": [d["demand_estimate"] for d in result["forecast"]],
            }

            st.table(forecast_df)

            # ─── Chart Visualization ─────────────────────
            st.subheader("📊 Demand Trend Chart")

            st.line_chart(forecast_df["Demand (Quintals)"])

            # ─── Metadata ────────────────────────────────
            st.markdown("---")
            st.caption(f"📌 Model: {result['metadata']['algorithm']}")
            st.caption(f"📅 Last Data Point: {result['metadata']['last_historical_date']}")

# ─── Footer ────────────────────────────────────────────────

st.markdown("---")
st.markdown("Built with ❤️ using real Agmarknet mandi data")