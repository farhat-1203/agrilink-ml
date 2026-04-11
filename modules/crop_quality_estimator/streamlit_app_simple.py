"""Simplified Crop Quality Estimator - Analyze Image Only"""
import io
import sys
import os
import warnings
import time

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from PIL import Image, ImageDraw

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Crop Quality Estimator",
    page_icon="🌿",
    layout="wide",
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background: #f5f7f2; }
.grade-A { background:#e8f5e9; border-left:5px solid #2e7d32; border-radius:10px; padding:14px 18px; }
.grade-B { background:#fff8e1; border-left:5px solid #f9a825; border-radius:10px; padding:14px 18px; }
.grade-C { background:#fce4ec; border-left:5px solid #c62828; border-radius:10px; padding:14px 18px; }
h1,h2,h3 { color: #1b5e20; }
.stButton>button { background:#2e7d32!important; color:white!important; }
</style>
""", unsafe_allow_html=True)

# ── Load inference engine ──────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading AI model…")
def load_engine():
    from modules.inference import InferenceEngine
    from pathlib import Path
    ckpt = "models/best_model.pt"
    if Path(ckpt).exists():
        eng = InferenceEngine(checkpoint_path=ckpt, device="cpu")
    else:
        eng = InferenceEngine(heuristic_only=True)
    return eng

ENGINE = load_engine()

# ── Helpers ────────────────────────────────────────────────────────────────────
GRADE_CSS = {"A": "grade-A", "B": "grade-B", "C": "grade-C"}
GRADE_ICON = {"A": "✅", "B": "⚠️", "C": "🔴"}
URGENCY_COLOR = {"low": "#2e7d32", "medium": "#f57f17", "high": "#c62828"}
URGENCY_ICON = {"low": "🟢", "medium": "🟡", "high": "🔴"}

def pil_to_bytes(img: Image.Image, fmt="JPEG") -> bytes:
    buf = io.BytesIO()
    img.save(buf, fmt)
    return buf.getvalue()

def make_demo_image(color: tuple, add_spots=False, size=300) -> Image.Image:
    img = Image.new("RGB", (size, size), (242, 246, 238))
    draw = ImageDraw.Draw(img)
    r = size // 2 - 24
    cx, cy = size // 2, size // 2
    draw.ellipse([cx-r, cy-r, cx+r, cy+r], fill=color,
                 outline=tuple(max(0, c-35) for c in color), width=3)
    if add_spots:
        import random
        rng = random.Random(42)
        for _ in range(rng.randint(5, 9)):
            ox = rng.randint(-r//2, r//2)
            oy = rng.randint(-r//2, r//2)
            sr = rng.randint(8, 16)
            draw.ellipse([cx+ox-sr, cy+oy-sr, cx+ox+sr, cy+oy+sr],
                         fill=(70, 48, 28))
    return img

def run_inference(image_bytes: bytes) -> dict:
    return ENGINE.predict(image_bytes)

def render_result(result: dict, img: Image.Image):
    grade = result["quality_grade"]
    urgency = result["urgency_level"]
    css = GRADE_CSS[grade]
    ug_color = URGENCY_COLOR[urgency]
    
    st.markdown(f"""
    <div class="{css}">
        <div style="display:flex; justify-content:space-between; align-items:center">
            <div>
                <h2 style="margin:0">{GRADE_ICON[grade]} Grade {grade} — {result['grade_label']}</h2>
                <p style="margin:4px 0 0; color:#555; font-size:14px">{result['summary']}</p>
            </div>
            <div style="text-align:right">
                <span style="background:{ug_color};color:white;border-radius:20px;padding:4px 14px;font-size:13px;font-weight:600">
                    {URGENCY_ICON[urgency]} {urgency.upper()}
                </span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Freshness Score", f"{result['freshness_score']}/100")
    m2.metric("Shelf Life", f"{result['shelf_life_days']} days")
    m3.metric("Disease Conf.", f"{result['disease_confidence']}%")
    m4.metric("Market", result["market_tier"].split("/")[0].strip())
    
    st.markdown("---")
    left, right = st.columns([1, 1])
    
    with left:
        st.image(img, caption="Analyzed image", use_container_width=True)
        st.markdown("#### 🦠 Disease Detection")
        dis_icon = "🔴" if result["is_diseased"] else "🟢"
        st.markdown(f"{dis_icon} **{result['disease_label']}** *(confidence: {result['disease_confidence']}%)*")
        
        if result["top3_predictions"]:
            st.markdown("**Top 3 predictions:**")
            for pred in result["top3_predictions"]:
                lbl = pred["label"].replace("___", " — ").replace("_", " ")
                pct = pred["confidence"]
                st.progress(int(pct), text=f"{lbl} ({pct:.1f}%)")
    
    with right:
        st.markdown("#### 📡 Freshness Signals")
        signals = result["freshness_signals"]
        sig_names = list(signals.keys())
        sig_vals = [round(v * 100, 1) for v in signals.values()]
        
        fig_radar = go.Figure(go.Scatterpolar(
            r=sig_vals + [sig_vals[0]],
            theta=sig_names + [sig_names[0]],
            fill="toself",
            line=dict(color="#2e7d32", width=2),
            fillcolor="rgba(46,125,50,0.15)",
        ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            height=280, margin=dict(t=20, b=20, l=40, r=40),
            paper_bgcolor="rgba(0,0,0,0)", showlegend=False,
        )
        st.plotly_chart(fig_radar, use_container_width=True)
        
        fig_bar = go.Figure(go.Bar(
            x=sig_names, y=sig_vals,
            marker_color=["#2e7d32" if v >= 60 else "#f9a825" if v >= 35 else "#c62828" for v in sig_vals],
            text=[f"{v:.0f}" for v in sig_vals],
            textposition="outside",
        ))
        fig_bar.update_layout(
            yaxis=dict(range=[0, 115], title="Score (0–100)"),
            height=240, margin=dict(t=10, b=30, l=10, r=10),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    
    with st.expander("📦 Full JSON response"):
        st.json(result)

# ══════════════════════════════════════════════════════════════════════════════
# MAIN PAGE
# ══════════════════════════════════════════════════════════════════════════════
st.title("🌿 Crop Quality Estimator")
st.caption("Upload a real photo or choose a demo scenario")

tab_upload, tab_demo = st.tabs(["📁 Upload Image", "🎨 Demo Scenarios"])

# ── Upload tab ─────────────────────────────────────────────────────────────────
with tab_upload:
    uploaded = st.file_uploader("Upload crop image (JPEG / PNG / WebP)",
                                 type=["jpg", "jpeg", "png", "webp"])
    if uploaded:
        raw = uploaded.read()
        pil = Image.open(io.BytesIO(raw)).convert("RGB")
        
        col_prev, col_btn = st.columns([3, 1])
        with col_prev:
            st.image(pil, caption="Uploaded image", width=220)
        with col_btn:
            st.markdown("<br><br>", unsafe_allow_html=True)
            run_btn = st.button("🔍 Analyze", use_container_width=True)
        
        if run_btn:
            with st.spinner("Running AI pipeline..."):
                result = run_inference(raw)
            st.markdown("---")
            render_result(result, pil)

# ── Demo tab ───────────────────────────────────────────────────────────────────
with tab_demo:
    st.markdown("Choose a pre-built scenario to test without a real photo:")
    
    DEMO_SCENARIOS = {
        "🍅 Fresh Tomato": {"color": (210, 40, 32), "spots": False},
        "🍅 Diseased Tomato": {"color": (155, 75, 55), "spots": True},
        "🧅 Fresh Onion": {"color": (210, 155, 70), "spots": False},
        "🥔 Rotting Potato": {"color": (115, 82, 52), "spots": True},
        "🥬 Healthy Cabbage": {"color": (52, 168, 50), "spots": False},
        "🥕 Fresh Carrot": {"color": (220, 110, 30), "spots": False},
    }
    
    chosen = st.selectbox("Select scenario", list(DEMO_SCENARIOS.keys()))
    cfg = DEMO_SCENARIOS[chosen]
    demo_img = make_demo_image(cfg["color"], cfg["spots"])
    
    col_img, col_info = st.columns([1, 2])
    with col_img:
        st.image(demo_img, caption=chosen, width=200)
    with col_info:
        st.markdown(f"**Spots visible:** {'Yes' if cfg['spots'] else 'No'}")
    
    if st.button("🔍 Analyze Demo Image", use_container_width=True):
        img_bytes = pil_to_bytes(demo_img)
        with st.spinner("Running AI pipeline..."):
            result = run_inference(img_bytes)
        st.markdown("---")
        render_result(result, demo_img)
