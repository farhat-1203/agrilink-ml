"""
AgriLink AI — Streamlit Demo Interface
Run: streamlit run app.py
"""
import sys, os, warnings, io, base64
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw
from datetime import date, timedelta

# ── ML modules ────────────────────────────────────────────────────────────────
from modules.price_predictor  import predict_price
from modules.demand_forecaster import forecast_demand
from modules.image_analyzer    import analyze_image
from modules.nlp_parser        import parse_voice_input
from modules.buyer_matcher     import match_buyers

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="AgriLink AI",
    page_icon="�",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background: #f5f7f9; }
[data-testid="stSidebar"] { background: #2c3e50; }
[data-testid="stSidebar"] * { color: #ecf0f1 !important; }
[data-testid="stSidebar"] .stRadio label { color: #ecf0f1 !important; font-size: 15px; font-weight: 500; }
[data-testid="stSidebar"] h2 { color: #ffffff !important; }

/* Dropdown/Select styling */
.stSelectbox label { color: #1a1a1a !important; font-weight: 500; font-size: 14px; }
.stSelectbox > div > div { background-color: white !important; color: #1a1a1a !important; border: 1px solid #d0d0d0 !important; }
.stSelectbox [data-baseweb="select"] { background-color: white !important; }
.stSelectbox [data-baseweb="select"] > div { color: #1a1a1a !important; font-weight: 500; }
.stSelectbox option { background-color: white !important; color: #1a1a1a !important; }
div[data-baseweb="select"] > div { background-color: white !important; color: #1a1a1a !important; }
div[data-baseweb="popover"] { background-color: white !important; }
ul[role="listbox"] { background-color: white !important; }
ul[role="listbox"] li { color: #1a1a1a !important; background-color: white !important; }
ul[role="listbox"] li:hover { background-color: #e8f5e9 !important; color: #1a1a1a !important; }
ul[role="listbox"] li[aria-selected="true"] { background-color: #27ae60 !important; color: white !important; }

/* Number input styling */
.stNumberInput label { color: #1a1a1a !important; font-weight: 500; font-size: 14px; }
.stNumberInput input { color: #1a1a1a !important; background-color: white !important; border: 1px solid #d0d0d0 !important; }

/* Text input styling */
.stTextInput label, .stTextArea label { color: #1a1a1a !important; font-weight: 500; font-size: 14px; }
.stTextInput input, .stTextArea textarea { color: #1a1a1a !important; background-color: white !important; border: 1px solid #d0d0d0 !important; }

/* Slider styling */
.stSlider label { color: #1a1a1a !important; font-weight: 500; font-size: 14px; }
.stSlider [data-baseweb="slider"] { color: #27ae60 !important; }

.metric-card {
    background: white; border-radius: 12px; padding: 18px 20px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1); margin-bottom: 12px;
    border-left: 4px solid #27ae60;
}
.metric-card h2 { color: #1a1a1a !important; margin: 0; font-size: 32px; }
.metric-card p { color: #555555 !important; margin: 4px 0 0; font-size: 14px; }
.grade-A { background: #d4edda; border-radius:8px; padding:12px 18px; border-left:4px solid #28a745; }
.grade-A h2 { color: #155724 !important; font-weight: 600; }
.grade-B { background: #fff3cd; border-radius:8px; padding:12px 18px; border-left:4px solid #ffc107; }
.grade-B h2 { color: #856404 !important; font-weight: 600; }
.grade-C { background: #f8d7da; border-radius:8px; padding:12px 18px; border-left:4px solid #dc3545; }
.grade-C h2 { color: #721c24 !important; font-weight: 600; }
.buyer-card {
    background: white; border-radius: 10px; padding: 16px 20px;
    box-shadow: 0 2px 6px rgba(0,0,0,0.1); margin-bottom: 12px;
    border-left: 4px solid #27ae60;
}
.buyer-card span { color: #1a1a1a !important; }
.buyer-card div { color: #444444 !important; }
.tag { background:#27ae60; color:#ffffff; border-radius:20px;
       padding:4px 12px; font-size:12px; margin-right:6px; font-weight: 500; }
.hero { background: linear-gradient(135deg,#27ae60,#2ecc71);
        color:white; border-radius:16px; padding:28px 32px; margin-bottom:24px; box-shadow: 0 4px 12px rgba(0,0,0,0.15); }
h1,h2,h3 { color: #1a1a1a !important; font-weight: 600; }
h1 { font-size: 2.5rem; }
h2 { font-size: 1.8rem; }
h3 { font-size: 1.4rem; }
.stButton>button { background:#27ae60; color:white; border:none;
                   border-radius:8px; font-weight:600; padding: 0.5rem 1rem; }
.stButton>button:hover { background:#229954; }
p, span, div { color: #2c3e50; }
.stMarkdown { color: #2c3e50; }
label { color: #1a1a1a !important; font-weight: 500; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## AgriLink AI")
    st.markdown("*Empowering farmers with AI*")
    st.markdown("---")
    page = st.radio("Navigate", [
        "Home",
        "Price Prediction",
        "Demand Forecast",
        "Image Quality",
        "Voice Parser",
        "Buyer Matching",
        "Full Pipeline",
    ])
    st.markdown("---")
    st.markdown("**Supported Crops**")
    crops_list = ["Tomato","Onion","Potato","Cabbage","Carrot","Wheat","Rice"]
    for c in crops_list:
        st.markdown(f"<span class='tag'>{c}</span>", unsafe_allow_html=True)
    st.markdown("---")
    st.caption("Model: RandomForest · R²=0.84\nForecast: Holt-Winters ETS\nNLP: Regex + Vocab Engine")

# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════
CROPS   = ["tomato","onion","potato","cabbage","carrot","wheat","rice","sugarcane"]
SEASONS = ["kharif","rabi","zaid"]
MARKETS = ["Mumbai","Pune","Nashik","Nagpur","Aurangabad","Kolhapur"]
GRADES  = ["A","B","C"]
CROP_EMOJI = {}

def emoji(crop): return ""

def gauge_chart(value, title, max_val=100, color="#2e7d32"):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={"text": title, "font": {"size": 14}},
        gauge={
            "axis": {"range": [0, max_val]},
            "bar": {"color": color},
            "steps": [
                {"range": [0, max_val*0.4],  "color": "#ffebee"},
                {"range": [max_val*0.4, max_val*0.7], "color": "#fff8e1"},
                {"range": [max_val*0.7, max_val], "color": "#e8f5e9"},
            ],
            "threshold": {"line": {"color": color, "width": 3}, "value": value},
        },
        number={"suffix": f"/{max_val}", "font": {"size": 22}},
    ))
    fig.update_layout(height=200, margin=dict(t=40, b=10, l=20, r=20),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    return fig

def make_synthetic_image(color_rgb, add_spots=False, size=240):
    """Create a synthetic crop image for demo purposes."""
    img = Image.new("RGB", (size, size), (248, 245, 238))
    draw = ImageDraw.Draw(img)
    r = size // 2 - 20
    cx, cy = size // 2, size // 2
    draw.ellipse([cx-r, cy-r, cx+r, cy+r], fill=color_rgb,
                 outline=tuple(max(0,c-30) for c in color_rgb), width=2)
    if add_spots:
        for ox, oy in [(-30,-20),(10,25),(-10,-35),(35,10)]:
            draw.ellipse([cx+ox-8, cy+oy-8, cx+ox+8, cy+oy+8], fill=(70,50,35))
    return img


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: HOME
# ══════════════════════════════════════════════════════════════════════════════
if page == "Home":
    st.markdown("""
    <div class="hero">
        <h1 style="color:white;margin:0">AgriLink AI</h1>
        <p style="color:#ffffff;font-size:17px;margin:8px 0 0;opacity:0.95">
        AI-powered marketplace connecting farmers directly with buyers.<br>
        Price transparency · Demand forecasting · Quality estimation · Smart matching
        </p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown('<div class="metric-card"><h2 style="margin:0;color:#1a1a1a">₹2.80</h2><p style="color:#555555;margin:4px 0 0">Avg price error (MAE)</p></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="metric-card"><h2 style="margin:0;color:#1a1a1a">0.84</h2><p style="color:#555555;margin:4px 0 0">Price model R² score</p></div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="metric-card"><h2 style="margin:0;color:#1a1a1a">7 days</h2><p style="color:#555555;margin:4px 0 0">Demand forecast horizon</p></div>', unsafe_allow_html=True)
    with c4:
        st.markdown('<div class="metric-card"><h2 style="margin:0;color:#1a1a1a">5 modules</h2><p style="color:#555555;margin:4px 0 0">AI capabilities live</p></div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### What each module does")
    modules = [
        ("Price Prediction",  "RandomForest on 3,000 samples. Inputs: crop, season, market, quality, rainfall. Returns price/kg with confidence range."),
        ("Demand Forecast",   "Holt-Winters Exponential Smoothing on 3-year daily history. Returns 7-day chart-ready JSON with trend direction."),
        ("Image Quality",     "OpenCV HSV + saturation analysis. Grades A/B/C, freshness score 0–100, shelf life in days."),
        ("Voice Parser",      "Regex + vocabulary engine supporting English + Hindi/Marathi transliteration. Converts speech to structured listing JSON."),
        ("Buyer Matching",    "4-factor scoring: price fairness (35%), quantity fit (25%), distance (25%), demand urgency (15%). Returns ranked buyer list."),
        ("Full Pipeline",     "Voice → parse → price → match in a single call. The star demo endpoint for zero-friction farmer onboarding."),
    ]
    for i in range(0, len(modules), 2):
        cols = st.columns(2)
        for j, col in enumerate(cols):
            if i+j < len(modules):
                name, desc = modules[i+j]
                col.markdown(f"**{name}**\n\n{desc}")

    st.markdown("---")
    st.info("Select a module from the sidebar to start testing")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: PRICE PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Price Prediction":
    st.title("Price Prediction")
    st.caption("RandomForest · 200 trees · R²=0.84 · MAE=₹2.80/kg")

    with st.form("price_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            crop    = st.selectbox("Crop", CROPS, format_func=lambda x: x.title())
            season  = st.selectbox("Season", SEASONS, format_func=str.title)
        with c2:
            market  = st.selectbox("Market", MARKETS)
            quality = st.selectbox("Quality Grade", GRADES,
                                   help="A = Premium, B = Standard, C = Below standard")
        with c3:
            qty     = st.number_input("Quantity (quintals)", 1.0, 200.0, 5.0, 1.0)
            rain    = st.slider("Rainfall (mm)", 10, 250, 80)
            days    = st.slider("Days to market", 0, 7, 1)
        submitted = st.form_submit_button("Predict Price", use_container_width=True)

    if submitted:
        with st.spinner("Running RandomForest..."):
            result = predict_price(crop, season, market, quality, qty, float(rain), days)

        price = result["predicted_price_per_kg"]
        lo    = result["price_range"]["low"]
        hi    = result["price_range"]["high"]
        conf  = result["confidence"]

        c1, c2, c3 = st.columns(3)
        c1.metric("Predicted Price", f"₹{price}/kg", delta=f"±₹{round((hi-lo)/2,2)} range")
        c2.metric("Price Range", f"₹{lo} – ₹{hi}")
        c3.metric("Confidence", f"{round(conf*100,1)}%")

        # Waterfall-style range bar
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=["Low estimate", "Predicted price", "High estimate"],
            y=[lo, price, hi],
            marker_color=["#ef9a9a", "#2e7d32", "#a5d6a7"],
            text=[f"₹{lo}", f"₹{price}", f"₹{hi}"],
            textposition="outside",
        ))
        fig.update_layout(
            title=f"Price estimate for {crop.title()} — {market} ({season.title()})",
            yaxis_title="Price (₹/kg)",
            showlegend=False, height=320,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)

        # Multi-quality comparison
        st.markdown("#### Price across all quality grades")
        comp_rows = []
        for g in GRADES:
            r2 = predict_price(crop, season, market, g, qty, float(rain), days)
            comp_rows.append({"Grade": g, "Price (₹/kg)": r2["predicted_price_per_kg"]})
        df_comp = pd.DataFrame(comp_rows)
        fig2 = px.bar(df_comp, x="Grade", y="Price (₹/kg)",
                      color="Grade",
                      color_discrete_map={"A":"#2e7d32","B":"#f9a825","C":"#c62828"},
                      text="Price (₹/kg)")
        fig2.update_traces(texttemplate="₹%{text}", textposition="outside")
        fig2.update_layout(height=260, showlegend=False,
                           paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig2, use_container_width=True)

        with st.expander("Raw API response"):
            st.json(result)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: DEMAND FORECAST
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Demand Forecast":
    st.title("Demand Forecast")
    st.caption("Holt-Winters Exponential Smoothing · Weekly seasonality · 3-year training history")

    c1, c2 = st.columns([2, 1])
    with c1:
        crop  = st.selectbox("Crop", CROPS, format_func=lambda x: x.title())
    with c2:
        steps = st.slider("Forecast days", 3, 14, 7)

    if st.button("Run Forecast", use_container_width=True):
        with st.spinner("Fitting Holt-Winters model..."):
            result = forecast_demand(crop, steps)

        # Trend badge
        c1, c2, c3 = st.columns(3)
        c1.metric("Trend", result['trend'].title())
        c2.metric("Slope", f"{result['trend_slope_per_day']:+.1f} qtl/day")
        c3.metric("Method", result["forecast_method"].split()[0])

        # Main forecast chart
        df = pd.DataFrame(result["daily_forecast"])
        df["date"] = pd.to_datetime(df["date"])
        avg = df["forecast_quintals"].mean()

        fig = go.Figure()
        fig.add_hline(y=avg, line_dash="dot", line_color="#9e9e9e",
                      annotation_text=f"Avg {avg:.0f} qtl", annotation_position="right")
        fig.add_trace(go.Scatter(
            x=df["date"], y=df["forecast_quintals"],
            mode="lines+markers+text",
            line=dict(color="#2e7d32", width=3),
            marker=dict(size=9, color="#2e7d32"),
            text=[f"{v:.0f}" for v in df["forecast_quintals"]],
            textposition="top center",
            fill="tozeroy", fillcolor="rgba(46,125,50,0.08)",
            name="Forecast",
        ))
        fig.update_layout(
            title=f"{crop.title()} demand forecast — next {steps} days",
            xaxis_title="Date", yaxis_title="Demand (quintals)",
            height=360, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(tickformat="%b %d"),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Day-by-day table with colored bars
        st.markdown("#### Day-by-day breakdown")
        df_show = df.copy()
        df_show["date"] = df_show["date"].dt.strftime("%a, %b %d")
        df_show.columns = ["Date", "Day", "Forecast (qtl)"]
        df_show = df_show[["Date", "Day", "Forecast (qtl)"]]
        st.dataframe(
            df_show.style.background_gradient(subset=["Forecast (qtl)"], cmap="Greens"),
            use_container_width=True, hide_index=True,
        )

        # Compare all 3 crops
        st.markdown("#### Compare crops side by side")
        with st.spinner("Fetching all crop forecasts..."):
            fig2 = go.Figure()
            colors = {"tomato":"#e53935","onion":"#8d6e63","potato":"#fdd835"}
            for c_name in ["tomato","onion","potato"]:
                r2 = forecast_demand(c_name, steps)
                df2 = pd.DataFrame(r2["daily_forecast"])
                fig2.add_trace(go.Scatter(
                    x=df2["date"], y=df2["forecast_quintals"],
                    mode="lines+markers",
                    line=dict(color=colors[c_name], width=2),
                    name=c_name.title(),
                ))
            fig2.update_layout(
                title="Demand comparison — Tomato / Onion / Potato",
                xaxis_title="Date", yaxis_title="Quintals",
                height=320, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig2, use_container_width=True)

        with st.expander("Raw API response"):
            st.json(result)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: IMAGE QUALITY
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Image Quality":
    st.title("Image Quality Analysis")
    st.caption("OpenCV HSV + saturation heuristic · Grades A/B/C · Shelf-life estimation")

    tab1, tab2 = st.tabs(["Upload Image", "Use Demo Image"])

    def show_quality_result(result, img: Image.Image):
        grade = result["quality_grade"]
        css   = {"A":"grade-A","B":"grade-B","C":"grade-C"}[grade]

        c1, c2 = st.columns([1, 2])
        with c1:
            st.image(img, caption="Analyzed image", use_container_width=True)

        with c2:
            st.markdown(f"""
            <div class="{css}">
                <h2 style="margin:0">Grade {grade} — {result['grade_label']}</h2>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("---")

            m1, m2, m3 = st.columns(3)
            m1.metric("Freshness", f"{result['freshness_score']}/100")
            m2.metric("Defect Score", f"{result['defect_score']}/100")
            m3.metric("Shelf Life", f"{result['shelf_life_days']} days")

            st.info(f"Recommendation: {result['recommendation']}")

            # Radar / bar chart for scores
            scores = {
                "Freshness":    result["freshness_score"],
                "Defect-free":  result["defect_score"],
                "Overall":      result["overall_score"],
            }
            fig = go.Figure(go.Bar(
                x=list(scores.keys()), y=list(scores.values()),
                marker_color=["#2e7d32" if v>=70 else "#f9a825" if v>=45 else "#c62828"
                              for v in scores.values()],
                text=[f"{v:.1f}" for v in scores.values()],
                textposition="outside",
            ))
            fig.update_layout(
                yaxis=dict(range=[0,110]), height=220,
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(t=10,b=10),
            )
            st.plotly_chart(fig, use_container_width=True)

        with st.expander("Raw API response"):
            st.json(result)

    with tab1:
        crop_up = st.selectbox("Crop type", CROPS, key="img_crop",
                               format_func=lambda x: x.title())
        uploaded = st.file_uploader("Upload crop image (JPG / PNG / WebP)",
                                    type=["jpg","jpeg","png","webp"])
        if uploaded:
            img_bytes = uploaded.read()
            pil_img   = Image.open(io.BytesIO(img_bytes))
            with st.spinner("Analyzing image..."):
                result = analyze_image(img_bytes, crop_up)
            show_quality_result(result, pil_img)

    with tab2:
        st.markdown("Choose a demo scenario to simulate different crop conditions:")
        demo_col1, demo_col2 = st.columns(2)
        with demo_col1:
            demo_crop = st.selectbox("Crop", ["tomato","onion","potato","cabbage"], key="demo_crop",
                                     format_func=lambda x: x.title())
        with demo_col2:
            demo_cond = st.selectbox("Condition", [
                "Fresh (vibrant color)",
                "Slightly stale (dull color)",
                "Rotting (spots + discolored)",
            ])

        color_map = {
            "tomato":  {"Fresh (vibrant color)":(210,38,30),
                        "Slightly stale (dull color)":(160,80,60),
                        "Rotting (spots + discolored)":(115,75,55)},
            "onion":   {"Fresh (vibrant color)":(195,130,65),
                        "Slightly stale (dull color)":(155,105,60),
                        "Rotting (spots + discolored)":(110,80,50)},
            "potato":  {"Fresh (vibrant color)":(210,175,100),
                        "Slightly stale (dull color)":(170,140,90),
                        "Rotting (spots + discolored)":(120,100,65)},
            "cabbage": {"Fresh (vibrant color)":(55,165,50),
                        "Slightly stale (dull color)":(90,140,70),
                        "Rotting (spots + discolored)":(85,110,65)},
        }
        add_spots = "Rotting" in demo_cond
        color     = color_map[demo_crop][demo_cond]
        demo_img  = make_synthetic_image(color, add_spots=add_spots)

        if st.button("Analyze Demo Image", use_container_width=True):
            buf = io.BytesIO()
            demo_img.save(buf, "JPEG")
            img_bytes = buf.getvalue()
            with st.spinner("Analyzing..."):
                result = analyze_image(img_bytes, demo_crop)
            show_quality_result(result, demo_img)
        else:
            st.image(demo_img, caption="Preview (click Analyze to run)", width=220)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: VOICE PARSER
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Voice Parser":
    st.title("Voice / Text Parser")
    st.caption("Regex + multilingual vocabulary engine · English + Hindi/Marathi transliteration")

    st.markdown("Type (or paste transcribed speech) in the box below:")

    examples = [
        "2 quintal tomatoes fresh selling tomorrow",
        "50 kg onion grade A nashik market price 25 rupees",
        "aloo 3 bag good quality pune market today",
        "मेरे पास 5 क्विंटल प्याज है",
        "3 quintal cabbage standard quality nagpur at 14 rupees",
        "carrot 100 kg fresh grade A kolhapur tomorrow",
    ]
    ex_choice = st.selectbox("Try an example", ["(type your own)"] + examples)
    default_text = "" if ex_choice == "(type your own)" else ex_choice
    text_input = st.text_area("Farmer utterance", value=default_text, height=80,
                               placeholder="e.g. 2 quintal tomatoes fresh selling tomorrow nashik")

    if st.button("Parse", use_container_width=True) and text_input.strip():
        with st.spinner("Parsing..."):
            result = parse_voice_input(text_input)

        conf_pct = int(result["confidence"] * 100)
        conf_color = "#2e7d32" if conf_pct >= 60 else "#f9a825" if conf_pct >= 40 else "#c62828"

        # Confidence bar
        st.markdown(f"""
        <div style="background:#f5f5f5;border-radius:8px;padding:4px 8px;margin-bottom:12px">
            <div style="background:{conf_color};width:{conf_pct}%;border-radius:6px;
                        padding:6px 12px;color:white;font-weight:600;font-size:14px">
                Parsing confidence: {conf_pct}%
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Parsed fields
        st.markdown("#### Extracted Fields")
        fields = {
            "Crop":           result.get("crop") or "—",
            "Quantity":       result.get("quantity_display") or "—",
            "Quantity (kg)":  f"{result['quantity_kg']} kg" if result.get("quantity_kg") else "—",
            "Quality Grade":  result.get("quality_grade") or "—",
            "Available Date": result.get("available_date") or "—",
            "Target Market":  result.get("target_market") or "—",
            "Asking Price":   f"₹{result['asking_price_per_kg']}/kg" if result.get("asking_price_per_kg") else "—",
        }
        c1, c2 = st.columns(2)
        items = list(fields.items())
        for i, (k, v) in enumerate(items):
            col = c1 if i % 2 == 0 else c2
            status = "[✓]" if v != "—" else "[?]"
            col.markdown(f"{status} **{k}:** {v}")

        # Missing fields warning
        if result["missing_fields"]:
            st.warning(f"Could not extract: **{', '.join(result['missing_fields'])}**. "
                       f"Ask the farmer to mention these.")
        else:
            st.success("All fields extracted successfully!")

        # Token debug
        with st.expander("Parsing tokens (debug)"):
            st.json(result["parsed_tokens"])
        with st.expander("Raw API response"):
            st.json(result)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: BUYER MATCHING
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Buyer Matching":
    st.title("Buyer Matching")
    st.caption("4-factor weighted scoring: price (35%) · quantity (25%) · distance (25%) · demand (15%)")

    with st.form("match_form"):
        c1, c2 = st.columns(2)
        with c1:
            crop     = st.selectbox("Crop", CROPS, format_func=lambda x: x.title())
            qty_kg   = st.number_input("Quantity (kg)", 50.0, 50000.0, 200.0, 50.0)
            mkt_price = st.number_input("Current market price (₹/kg)", 5.0, 200.0, 18.5, 0.5)
        with c2:
            lat = st.number_input("Farmer latitude",  17.0, 22.0, 19.99, 0.01)
            lon = st.number_input("Farmer longitude", 72.0, 80.0, 73.78, 0.01)
            top_n = st.slider("Show top N buyers", 1, 10, 5)
        submitted = st.form_submit_button("Find Best Buyers", use_container_width=True)

    if submitted:
        with st.spinner("Scoring buyers..."):
            result = match_buyers(crop, qty_kg, lat, lon, mkt_price, top_n)

        st.success(f"Found **{result['total_buyers_evaluated']}** buyers evaluated · "
                   f"Showing top **{len(result['matches'])}**")

        # Score breakdown chart
        if result["matches"]:
            df_scores = pd.DataFrame([
                {
                    "Buyer": m["name"].split()[0] + " " + m["name"].split()[-1],
                    "Price": m["score_breakdown"]["price_score"],
                    "Quantity": m["score_breakdown"]["quantity_score"],
                    "Distance": m["score_breakdown"]["distance_score"],
                    "Demand": m["score_breakdown"]["demand_score"],
                    "Total": m["match_score"],
                }
                for m in result["matches"]
            ])

            fig = go.Figure()
            for factor, color in [("Price","#2e7d32"),("Quantity","#1565c0"),
                                   ("Distance","#f57f17"),("Demand","#6a1b9a")]:
                fig.add_trace(go.Bar(
                    name=factor, x=df_scores["Buyer"], y=df_scores[factor],
                    marker_color=color, text=df_scores[factor].round(1),
                    textposition="inside",
                ))
            fig.update_layout(
                barmode="stack", title="Score breakdown by buyer",
                yaxis_title="Score (max 100)", height=320,
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                legend=dict(orientation="h", y=1.1),
            )
            st.plotly_chart(fig, use_container_width=True)

        # Buyer cards
        st.markdown("#### Ranked Buyers")
        rank_icons = ["#1","#2","#3","#4","#5","#6","#7","#8","#9","#10"]
        for i, m in enumerate(result["matches"]):
            label_color = {"Excellent":"#27ae60","Good":"#f39c12","Fair":"#e74c3c"}[m["match_label"]]
            st.markdown(f"""
            <div class="buyer-card">
                <div style="display:flex;justify-content:space-between;align-items:center">
                    <span style="font-size:18px;font-weight:700;color:#2c3e50">{rank_icons[i]} {m['name']}</span>
                    <span style="background:{label_color};color:white;border-radius:20px;
                                 padding:4px 14px;font-size:13px;font-weight:600">
                        {m['match_label']} · {m['match_score']}
                    </span>
                </div>
                <div style="color:#34495e;margin-top:8px;font-size:14px;line-height:1.6">
                    Location: {m['location']} | Distance: {m['distance_km']} km | Price: ₹{m['offered_price_per_kg']}/kg<br>
                    Capacity: {m['min_qty_kg']}–{m['max_qty_kg']} kg | Payment: {m['payment_days']} days | Rating: {m['buyer_rating']}
                </div>
            </div>
            """, unsafe_allow_html=True)

        with st.expander("Raw API response"):
            st.json(result)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: FULL PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Full Pipeline":
    st.title("Full Voice → Price → Match Pipeline")
    st.caption("One call: parse farmer speech → predict price → rank buyers")

    st.markdown("""
    <div style="background:#d4edda;border-radius:10px;padding:14px 18px;margin-bottom:16px;border-left:4px solid #28a745;color:#155724">
    This is the <strong>star demo endpoint</strong> — a zero-friction flow designed for farmers with low literacy.<br>
    Speak → transcribe (via device STT) → paste here → get instant price + buyers.
    </div>
    """, unsafe_allow_html=True)

    pipeline_examples = [
        "3 quintal fresh onion nashik tomorrow price 22 rupees",
        "2 quintal tomatoes fresh selling tomorrow",
        "potato 500 kg grade B pune market",
        "cabbage 100 kg good quality nagpur today at 14 rupees",
        "carrot 200 kg premium nashik market kal",
    ]
    ex = st.selectbox("Try an example", ["(type your own)"] + pipeline_examples)
    default = "" if ex == "(type your own)" else ex
    voice_text = st.text_area("Farmer voice input (transcribed)", value=default, height=70)

    # Farmer GPS (simplified)
    with st.expander("Farmer location (optional)"):
        gc1, gc2 = st.columns(2)
        farmer_lat = gc1.number_input("Latitude",  17.0, 22.0, 19.5, 0.01)
        farmer_lon = gc2.number_input("Longitude", 72.0, 80.0, 73.8, 0.01)

    if st.button("Run Full Pipeline", use_container_width=True) and voice_text.strip():

        col_step, col_result = st.columns([1, 2])

        with col_step:
            # Animated step indicators
            s1 = st.empty(); s2 = st.empty(); s3 = st.empty()

            s1.markdown("[●] **Step 1:** Parsing voice...")
            with st.spinner(""):
                parsed = parse_voice_input(voice_text)
            s1.markdown("[✓] **Step 1:** Voice parsed")

            if not parsed["crop"] or not parsed["quantity_kg"]:
                st.error(f"Could not extract: {parsed['missing_fields']}\n\nPlease re-phrase.")
                st.stop()

            s2.markdown("[●] **Step 2:** Predicting price...")
            price_res = predict_price(
                parsed["crop"],
                "kharif",
                parsed["target_market"] or "Nashik",
                parsed["quality_grade"] or "B",
                (parsed["quantity_kg"] or 100) / 100,
                80.0, 1,
            )
            s2.markdown("[✓] **Step 2:** Price predicted")

            s3.markdown("[●] **Step 3:** Matching buyers...")
            buyer_res = match_buyers(
                parsed["crop"],
                parsed["quantity_kg"],
                farmer_lat, farmer_lon,
                price_res["predicted_price_per_kg"],
                top_n=3,
            )
            s3.markdown("[✓] **Step 3:** Buyers ranked")

        with col_result:
            crop_e = ""

            # Summary banner
            best = buyer_res["best_match"]
            st.markdown(f"""
            <div style="background:linear-gradient(135deg,#1b5e20,#388e3c);
                        color:white;border-radius:14px;padding:20px 24px;margin-bottom:16px">
                <h3 style="color:white;margin:0">{crop_e} {parsed['crop'].title()} — {parsed['quantity_display']}</h3>
                <p style="color:#c8e6c9;margin:6px 0 0;font-size:15px">
                    Market price: <strong>₹{price_res['predicted_price_per_kg']}/kg</strong> &nbsp;·&nbsp;
                    Best offer: <strong>₹{best['offered_price_per_kg']}/kg</strong> from {best['name']}
                </p>
            </div>
            """, unsafe_allow_html=True)

            # Three side-by-side result cards
            r1, r2, r3 = st.columns(3)
            r1.metric("Parsed crop",    parsed['crop'].title())
            r2.metric("Market price",   f"₹{price_res['predicted_price_per_kg']}/kg",
                      delta=f"conf {round(price_res['confidence']*100)}%")
            r3.metric("Best buyer offer", f"₹{best['offered_price_per_kg']}/kg",
                      delta=f"{best['distance_km']} km away")

            # Mini demand chart
            with st.spinner("Loading demand forecast..."):
                forecast = forecast_demand(parsed["crop"], 7)
            df_f = pd.DataFrame(forecast["daily_forecast"])
            fig = go.Figure(go.Scatter(
                x=df_f["day"].str[:3], y=df_f["forecast_quintals"],
                mode="lines+markers", fill="tozeroy",
                line=dict(color="#2e7d32", width=2),
                marker=dict(size=6),
            ))
            fig.update_layout(
                title=f"7-day demand for {parsed['crop'].title()}",
                height=200, margin=dict(t=35,b=20,l=30,r=10),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig, use_container_width=True)

        # Buyer results
        st.markdown("#### Top Matched Buyers")
        rank_icons = ["#1","#2","#3"]
        bc1, bc2, bc3 = st.columns(3)
        for i, (m, col) in enumerate(zip(buyer_res["matches"], [bc1, bc2, bc3])):
            col.markdown(f"""
            <div class="buyer-card" style="height:140px">
                <div style="font-weight:700;font-size:15px;color:#2c3e50">{rank_icons[i]} {m['name']}</div>
                <div style="color:#34495e;font-size:13px;margin-top:6px;line-height:1.6">
                    Location: {m['location']}<br>
                    Price: ₹{m['offered_price_per_kg']}/kg<br>
                    Distance: {m['distance_km']} km | Rating: {m['buyer_rating']}<br>
                    Payment: {m['payment_days']} days
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Full parsed details
        with st.expander("Full parsed details"):
            st.markdown("**Parsed listing:**"); st.json(parsed)
            st.markdown("**Price prediction:**"); st.json(price_res)
            st.markdown("**Buyer matches:**"); st.json(buyer_res)