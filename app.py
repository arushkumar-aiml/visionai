"""
VisionAI — Smart Image Classifier
Streamlit web app with AntigravityAI explanations and classification history.

Built by Arush Kumar & Ayushi Shukla | github.com/arushkumar-aiml/visionai
Usage: streamlit run app.py
"""

import os
import io
import json
import base64
import datetime
import numpy as np
import streamlit as st
from PIL import Image

# ── Page config (must be first Streamlit call) ─────────────────────────────
st.set_page_config(
    page_title="VisionAI — Smart Image Classifier",
    page_icon="🔭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Lazy imports (avoid hard crash if tf not installed yet) ────────────────
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

from ai_explainer import AntigravityAI

# ── Constants ──────────────────────────────────────────────────────────────
MODEL_PATH = "models/visionai_model.keras"
CLASS_NAMES_PATH = "models/class_names.json"
IMG_SIZE = (224, 224)
MAX_HISTORY = 10

# ── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* ── Global ─────────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* ── Header gradient ────────────────────────── */
    .visionai-header {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        border-radius: 16px;
        padding: 2rem 2.5rem;
        margin-bottom: 1.5rem;
        color: white;
        text-align: center;
    }
    .visionai-header h1 { font-size: 2.8rem; font-weight: 700; margin: 0; letter-spacing: -1px; }
    .visionai-header p  { font-size: 1.05rem; opacity: 0.8; margin: 0.4rem 0 0; }

    /* ── Cards ──────────────────────────────────── */
    .result-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid rgba(100,130,255,0.3);
        border-radius: 14px;
        padding: 1.5rem;
        color: white;
        margin-bottom: 1rem;
    }
    .ai-card {
        background: linear-gradient(135deg, #0d2137 0%, #1a3a5c 100%);
        border: 1px solid rgba(0,200,255,0.3);
        border-radius: 14px;
        padding: 1.5rem;
        color: white;
        margin-top: 1rem;
    }
    .ai-card h3 { color: #00d4ff; margin-top: 0; }

    /* ── Confidence bar ─────────────────────────── */
    .conf-bar-wrap { background: rgba(255,255,255,0.1); border-radius: 99px; height: 14px; width: 100%; }
    .conf-bar-fill { height: 14px; border-radius: 99px;
                     background: linear-gradient(90deg, #6c63ff, #00d4ff); }

    /* ── History sidebar items ──────────────────── */
    .hist-item {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 10px;
        padding: 0.6rem 0.8rem;
        margin-bottom: 0.5rem;
        font-size: 0.82rem;
    }
    .hist-class { font-weight: 700; color: #00d4ff; }
    .hist-conf  { color: #aaa; }

    /* ── Upload zone ────────────────────────────── */
    .upload-hint {
        text-align: center;
        padding: 1.5rem;
        color: #888;
        border: 2px dashed #444;
        border-radius: 12px;
        margin-bottom: 1rem;
    }

    /* ── Badges ─────────────────────────────────── */
    .badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 99px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 6px;
    }
    .badge-ai  { background: rgba(0,212,255,0.2); color: #00d4ff; border: 1px solid rgba(0,212,255,0.4); }
    .badge-tf  { background: rgba(255,130,50,0.2); color: #ff8232; border: 1px solid rgba(255,130,50,0.4); }

    /* ── Footer ─────────────────────────────────── */
    .footer { text-align: center; color: #555; font-size: 0.78rem; margin-top: 3rem; }

    /* Hide Streamlit default elements */
    #MainMenu, footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Helper functions ───────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading VisionAI model...")
def load_model_and_classes():
    """Load the trained Keras model and class names. Cached across reruns."""
    if not TF_AVAILABLE:
        return None, []

    if not os.path.exists(MODEL_PATH):
        return None, []

    class_names = []
    if os.path.exists(CLASS_NAMES_PATH):
        with open(CLASS_NAMES_PATH) as f:
            class_names = json.load(f)

    model = tf.keras.models.load_model(MODEL_PATH)
    return model, class_names


def preprocess_image(pil_image: Image.Image) -> np.ndarray:
    """Resize and normalise a PIL image for model inference."""
    img = pil_image.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32)
    # MobileNetV2 preprocessing: scale to [-1, 1]
    arr = (arr / 127.5) - 1.0
    return np.expand_dims(arr, axis=0)


def image_to_b64(pil_image: Image.Image, max_size: int = 80) -> str:
    """Thumbnail a PIL image and return base64 string for sidebar display."""
    thumb = pil_image.copy()
    thumb.thumbnail((max_size, max_size))
    buf = io.BytesIO()
    thumb.save(buf, format="JPEG", quality=80)
    return base64.b64encode(buf.getvalue()).decode()


def confidence_color(conf: float) -> str:
    if conf >= 0.85:
        return "#00e676"
    elif conf >= 0.65:
        return "#ffeb3b"
    elif conf >= 0.45:
        return "#ff9800"
    else:
        return "#f44336"


def init_session_state():
    if "history" not in st.session_state:
        st.session_state.history = []
    if "api_key" not in st.session_state:
        st.session_state.api_key = os.environ.get("ANTHROPIC_API_KEY", "")


def add_to_history(pil_image, predicted_class, confidence, explanation_data):
    """Prepend a classification result to session history (max 10 items)."""
    entry = {
        "timestamp": datetime.datetime.now().strftime("%H:%M:%S"),
        "image_b64": image_to_b64(pil_image),
        "predicted_class": predicted_class,
        "confidence": confidence,
        "explanation": explanation_data.get("explanation", ""),
        "fun_fact": explanation_data.get("fun_fact", ""),
        "confidence_label": explanation_data.get("confidence_label", ""),
    }
    st.session_state.history.insert(0, entry)
    if len(st.session_state.history) > MAX_HISTORY:
        st.session_state.history = st.session_state.history[:MAX_HISTORY]


# ── Sidebar ────────────────────────────────────────────────────────────────

def render_sidebar():
    with st.sidebar:
        st.markdown("## ⚙️ Settings")

        # API key input
        api_key_input = st.text_input(
            "Anthropic API Key",
            value=st.session_state.api_key,
            type="password",
            placeholder="sk-ant-...",
            help="Required for AntigravityAI explanations. Get yours at console.anthropic.com",
        )
        if api_key_input != st.session_state.api_key:
            st.session_state.api_key = api_key_input

        if st.session_state.api_key:
            st.success("✅ AntigravityAI active")
        else:
            st.warning("⚠️ No API key — basic results only")

        st.divider()

        # Classification history
        st.markdown("## 🕘 History")

        if not st.session_state.history:
            st.caption("No classifications yet. Upload an image to start!")
        else:
            if st.button("🗑️ Clear History", use_container_width=True):
                st.session_state.history = []
                st.rerun()

            for i, entry in enumerate(st.session_state.history):
                conf_pct = round(entry["confidence"] * 100, 1)
                color = confidence_color(entry["confidence"])

                st.markdown(
                    f"""
                    <div class="hist-item">
                        <div style="display:flex; align-items:center; gap:10px;">
                            <img src="data:image/jpeg;base64,{entry['image_b64']}"
                                 style="width:44px;height:44px;border-radius:8px;object-fit:cover;">
                            <div>
                                <div class="hist-class">{entry['predicted_class'].replace('_',' ').title()}</div>
                                <div class="hist-conf">{conf_pct}% confidence &nbsp; <span style="color:{color}">●</span></div>
                                <div style="color:#666;font-size:0.7rem;">{entry['timestamp']}</div>
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Expandable details
                with st.expander(f"💬 AI Explanation #{len(st.session_state.history) - i}", expanded=False):
                    if entry["explanation"]:
                        st.write(entry["explanation"])
                    if entry["fun_fact"]:
                        st.info(entry["fun_fact"])

        st.divider()
        st.markdown(
            "<div style='font-size:0.75rem;color:#555;text-align:center;'>"
            "Built by <b>Arush Kumar</b> & <b>Ayushi Shukla</b><br>"
            "<a href='https://github.com/arushkumar-aiml/visionai' style='color:#6c63ff;'>"
            "github.com/arushkumar-aiml/visionai</a></div>",
            unsafe_allow_html=True,
        )


# ── Main app ───────────────────────────────────────────────────────────────

def main():
    init_session_state()
    render_sidebar()

    # ── Header ──────────────────────────────────────────────────────────────
    st.markdown(
        """
        <div class="visionai-header">
            <h1>🔭 VisionAI</h1>
            <p>Smart Image Classifier powered by TensorFlow + AntigravityAI (Claude)</p>
            <div style="margin-top:0.8rem;">
                <span class="badge badge-tf">TensorFlow</span>
                <span class="badge badge-ai">AntigravityAI</span>
                <span class="badge badge-ai">Claude API</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Load model ───────────────────────────────────────────────────────────
    model, class_names = load_model_and_classes()

    model_loaded = model is not None and len(class_names) > 0

    if not TF_AVAILABLE:
        st.error("❌ TensorFlow is not installed. Run: `pip install tensorflow`")
        return

    if not model_loaded:
        st.warning(
            "⚠️ **No trained model found.** "
            "Train your model first by running `python train.py`, "
            "then refresh this page. "
            "Make sure `models/visionai_model.h5` and `models/class_names.json` exist."
        )

    # ── Main layout ──────────────────────────────────────────────────────────
    col_upload, col_result = st.columns([1, 1], gap="large")

    with col_upload:
        st.markdown("### 📤 Upload an Image")
        uploaded_file = st.file_uploader(
            "Choose an image",
            type=["jpg", "jpeg", "png", "webp", "bmp"],
            label_visibility="collapsed",
        )

        if uploaded_file:
            pil_image = Image.open(uploaded_file)
            st.image(pil_image, caption="Uploaded Image", use_container_width=True)
            st.caption(
                f"📐 {pil_image.width} × {pil_image.height} px  |  "
                f"🎨 {pil_image.mode}  |  📁 {uploaded_file.name}"
            )
        else:
            st.markdown(
                '<div class="upload-hint">'
                "🖼️ Drag & drop or click to upload<br>"
                "<small>Supports JPG, PNG, WEBP, BMP</small>"
                "</div>",
                unsafe_allow_html=True,
            )

    with col_result:
        st.markdown("### 🎯 Classification Result")

        if not uploaded_file:
            st.info("Upload an image on the left to see results here.")
            return

        if not model_loaded:
            st.error("Model not loaded. Please train first.")
            return

        # ── Inference ────────────────────────────────────────────────────────
        with st.spinner("🧠 Analysing image..."):
            try:
                processed = preprocess_image(pil_image)
                predictions = model.predict(processed, verbose=0)[0]
                pred_idx = int(np.argmax(predictions))
                confidence = float(predictions[pred_idx])
                predicted_class = class_names[pred_idx]

                # Top-3 predictions
                top3_idx = np.argsort(predictions)[::-1][:3]
                top3 = [(class_names[i], float(predictions[i])) for i in top3_idx]

            except Exception as e:
                st.error(f"❌ Prediction failed: {e}")
                return

        # ── Result card ──────────────────────────────────────────────────────
        conf_pct = round(confidence * 100, 1)
        color = confidence_color(confidence)
        display_name = predicted_class.replace("_", " ").title()

        st.markdown(
            f"""
            <div class="result-card">
                <div style="font-size:0.85rem;color:#aaa;margin-bottom:4px;">DETECTED</div>
                <div style="font-size:2rem;font-weight:700;">{display_name}</div>
                <div style="font-size:1.1rem;color:{color};margin:6px 0 12px;">
                    {conf_pct}% confidence
                </div>
                <div class="conf-bar-wrap">
                    <div class="conf-bar-fill" style="width:{conf_pct}%;background:linear-gradient(90deg,{color},{color}88);"></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ── Top-3 breakdown ──────────────────────────────────────────────────
        with st.expander("📊 Top-3 Predictions", expanded=False):
            for cls, prob in top3:
                p = round(prob * 100, 1)
                c = confidence_color(prob)
                st.markdown(
                    f"""
                    <div style="margin-bottom:8px;">
                        <div style="display:flex;justify-content:space-between;margin-bottom:3px;">
                            <span>{cls.replace('_',' ').title()}</span>
                            <span style="color:{c};font-weight:600;">{p}%</span>
                        </div>
                        <div class="conf-bar-wrap">
                            <div class="conf-bar-fill" style="width:{p}%;height:8px;background:{c}55;"></div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        # ── AntigravityAI explanation ─────────────────────────────────────────
        st.markdown("### 🤖 AntigravityAI Explanation")

        with st.spinner("✨ Generating AI explanation..."):
            ai = AntigravityAI(api_key=st.session_state.api_key)
            explanation_data = ai.explain(predicted_class, confidence)

        if explanation_data["success"]:
            badge = '<span class="badge badge-ai">Powered by Claude</span>'
        else:
            badge = '<span class="badge" style="background:rgba(255,150,0,0.2);color:#ffaa00;border:1px solid rgba(255,150,0,0.4);">Basic Mode</span>'

        st.markdown(
            f"""
            <div class="ai-card">
                <h3>💡 About: {display_name} &nbsp; {badge}</h3>
                <p style="line-height:1.7;">{explanation_data['explanation']}</p>
                <div style="background:rgba(0,212,255,0.08);border-left:3px solid #00d4ff;
                            padding:0.7rem 1rem;border-radius:0 8px 8px 0;margin:0.8rem 0;">
                    🎉 <em>{explanation_data['fun_fact']}</em>
                </div>
                <div style="color:#aaa;font-size:0.85rem;margin-top:0.6rem;">
                    📊 <strong>Confidence:</strong> {explanation_data['confidence_label']}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if not explanation_data["success"] and not st.session_state.api_key:
            st.caption("💡 Add your Anthropic API key in the sidebar for full AI explanations.")

        # ── Save to history ───────────────────────────────────────────────────
        add_to_history(pil_image, predicted_class, confidence, explanation_data)

    # ── Classes info ─────────────────────────────────────────────────────────
    if model_loaded:
        with st.expander(f"📚 Model Info — {len(class_names)} Classes", expanded=False):
            cols = st.columns(min(len(class_names), 5))
            for i, cls in enumerate(class_names):
                with cols[i % len(cols)]:
                    st.markdown(
                        f"<div style='background:rgba(108,99,255,0.15);border:1px solid rgba(108,99,255,0.4);"
                        f"border-radius:8px;padding:6px 12px;text-align:center;margin:3px;font-size:0.85rem;'>"
                        f"{cls.replace('_',' ').title()}</div>",
                        unsafe_allow_html=True,
                    )

    # ── Footer ────────────────────────────────────────────────────────────────
    st.markdown(
        "<div class='footer'>"
        "Built by <strong>Arush Kumar</strong> &amp; <strong>Ayushi Shukla</strong> &nbsp;|&nbsp; "
        "<a href='https://github.com/arushkumar-aiml/visionai' style='color:#6c63ff;'>"
        "github.com/arushkumar-aiml/visionai</a>"
        "</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()