"""
Malaria Detection - Streamlit App
===================================
AI-powered blood smear analysis.
Run with: streamlit run app.py
"""

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import time
import datetime

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Malaria Detection System",
    page_icon="🦟",
    layout="centered"
)

# ─────────────────────────────────────────────
#  CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Inter:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background-color: #0d1117;
        color: #e6edf3;
    }

    .main-header {
        text-align: center;
        padding: 2rem 0 1rem;
    }

    .badge {
        display: inline-block;
        background: linear-gradient(90deg, #238636, #2ea043);
        color: white;
        padding: 4px 16px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 2px;
        margin-bottom: 12px;
        font-family: 'Space Mono', monospace;
    }

    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #e6edf3;
        margin: 0;
    }

    .subtitle {
        color: #8b949e;
        font-size: 0.95rem;
        margin-top: 8px;
    }

    .stat-container {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        margin-bottom: 1rem;
    }

    .stat-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #58a6ff;
        font-family: 'Space Mono', monospace;
    }

    .stat-label {
        font-size: 0.72rem;
        color: #8b949e;
        margin-top: 4px;
        letter-spacing: 0.5px;
    }

    .result-infected {
        background: rgba(248, 81, 73, 0.1);
        border: 1px solid rgba(248, 81, 73, 0.4);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
    }

    .result-healthy {
        background: rgba(46, 160, 67, 0.1);
        border: 1px solid rgba(46, 160, 67, 0.4);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
    }

    .result-title {
        font-size: 1.4rem;
        font-weight: 700;
        margin: 0.5rem 0;
    }

    .result-infected .result-title { color: #f85149; }
    .result-healthy  .result-title { color: #2ea043; }

    .result-meta {
        color: #8b949e;
        font-size: 0.85rem;
        font-family: 'Space Mono', monospace;
    }

    .history-item {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 0.75rem 1rem;
        margin-bottom: 0.5rem;
        display: flex;
        justify-content: space-between;
        font-size: 0.85rem;
    }

    .stButton > button {
        background: linear-gradient(90deg, #1f6feb, #388bfd) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        padding: 0.6rem 2rem !important;
        width: 100% !important;
        font-size: 1rem !important;
    }

    .divider {
        border: none;
        border-top: 1px solid #21262d;
        margin: 1.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  SESSION STATE
# ─────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []
if "total_latency" not in st.session_state:
    st.session_state.total_latency = 0

# ─────────────────────────────────────────────
#  LOAD MODEL (cached so it only loads once)
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    from keras.layers import Dense

    class PatchedDense(Dense):
        def __init__(self, *args, **kwargs):
            kwargs.pop('quantization_config', None)
            super().__init__(*args, **kwargs)

    model = tf.keras.models.load_model(
        'malaria_model_final.h5',
        custom_objects={'Dense': PatchedDense},
        compile=False
    )
    return model

IMG_SIZE = (128, 128)

# ─────────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <div class="badge">⚡ 5G ENABLED</div>
    <div class="main-title">🦟 Malaria Detection System</div>
    <div class="subtitle">AI-powered blood smear analysis · MobileNetV2</div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  STATS BAR
# ─────────────────────────────────────────────
total = len(st.session_state.history)
avg_latency = round(st.session_state.total_latency / total) if total > 0 else 0

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown('<div class="stat-container"><div class="stat-value">94.3%</div><div class="stat-label">MODEL ACCURACY</div></div>', unsafe_allow_html=True)
with col2:
    st.markdown('<div class="stat-container"><div class="stat-value">0.9846</div><div class="stat-label">AUC-ROC SCORE</div></div>', unsafe_allow_html=True)
with col3:
    st.markdown(f'<div class="stat-container"><div class="stat-value">{total}</div><div class="stat-label">TOTAL PREDICTIONS</div></div>', unsafe_allow_html=True)
with col4:
    latency_display = f"{avg_latency}ms" if total > 0 else "—"
    st.markdown(f'<div class="stat-container"><div class="stat-value">{latency_display}</div><div class="stat-label">AVG LATENCY</div></div>', unsafe_allow_html=True)

st.markdown('<hr class="divider">', unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  UPLOAD + PREDICT
# ─────────────────────────────────────────────
st.markdown("#### 🔬 Upload Blood Smear Image")
uploaded_file = st.file_uploader(
    "Choose a cell image (PNG or JPG)",
    type=["png", "jpg", "jpeg"],
    label_visibility="collapsed"
)

if uploaded_file:
    col_img, col_info = st.columns([1, 2])
    with col_img:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="Uploaded image", use_container_width=True)
    with col_info:
        st.markdown(f"""
        **File:** `{uploaded_file.name}`  
        **Size:** `{img.size[0]} × {img.size[1]} px`  
        **Format:** `{uploaded_file.type}`
        """)
        st.markdown(" ")
        analyze = st.button("🚀 Analyze via 5G Network")

    if analyze:
        model = load_model()

        with st.spinner("Transmitting over 5G network... Running AI analysis..."):
            start = time.time()

            img_resized = img.resize(IMG_SIZE)
            img_array = np.array(img_resized) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            prob = float(model.predict(img_array, verbose=0)[0][0])
            latency_ms = round((time.time() - start) * 1000)

        prediction = "Parasitized" if prob > 0.5 else "Uninfected"
        confidence = prob if prob > 0.5 else 1 - prob
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")

        st.session_state.history.insert(0, {
            "prediction": prediction,
            "confidence": f"{confidence:.1%}",
            "latency_ms": latency_ms,
            "timestamp": timestamp,
        })
        st.session_state.total_latency += latency_ms

        st.markdown('<hr class="divider">', unsafe_allow_html=True)

        if prediction == "Parasitized":
            st.markdown(f"""
            <div class="result-infected">
                <div style="font-size:3rem">🦟</div>
                <div class="result-title">Malaria Detected — Parasitized</div>
                <div class="result-meta">Confidence: {confidence:.1%} &nbsp;·&nbsp; Latency: {latency_ms}ms &nbsp;·&nbsp; {timestamp}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-healthy">
                <div style="font-size:3rem">✅</div>
                <div class="result-title">No Malaria — Uninfected</div>
                <div class="result-meta">Confidence: {confidence:.1%} &nbsp;·&nbsp; Latency: {latency_ms}ms &nbsp;·&nbsp; {timestamp}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown(" ")
        st.progress(confidence, text=f"Confidence: {confidence:.1%}")

        st.rerun()

# ─────────────────────────────────────────────
#  PREDICTION HISTORY
# ─────────────────────────────────────────────
st.markdown('<hr class="divider">', unsafe_allow_html=True)
st.markdown("#### 📋 Prediction History")

if not st.session_state.history:
    st.markdown('<p style="color:#8b949e; font-size:0.9rem;">No predictions yet. Upload an image to begin.</p>', unsafe_allow_html=True)
else:
    for entry in st.session_state.history:
        is_infected = entry["prediction"] == "Parasitized"
        dot_color = "#f85149" if is_infected else "#2ea043"
        label = "🦟 Parasitized" if is_infected else "✅ Uninfected"
        st.markdown(f"""
        <div class="history-item">
            <span>
                <span style="display:inline-block;width:10px;height:10px;border-radius:50%;
                      background:{dot_color};margin-right:8px;vertical-align:middle;"></span>
                <strong>{label}</strong>
            </span>
            <span style="color:#8b949e;">{entry['confidence']} confidence</span>
            <span style="color:#8b949e;font-family:'Space Mono',monospace;">{entry['latency_ms']}ms</span>
            <span style="color:#6e7681;">{entry['timestamp']}</span>
        </div>
        """, unsafe_allow_html=True)