import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from preprocessing import preprocess_image
from predict import predict_image, classes

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Garbage Classification",
    page_icon="♻️",
    layout="centered",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

/* ── Root palette ── */
:root {
    --bg:        #0d0f14;
    --surface:   #161b25;
    --border:    #262d3d;
    --accent:    #3bffa0;
    --accent2:   #00d4ff;
    --muted:     #6b7a99;
    --text:      #e8edf5;
    --danger:    #ff5c7a;
    --radius:    14px;
}

/* ── Global reset ── */
html, body, [class*="css"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif !important;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2.5rem !important; max-width: 720px !important; }

/* ── Hero header ── */
.hero {
    text-align: center;
    padding: 2.8rem 1.5rem 2rem;
    background: linear-gradient(135deg, #111827 0%, #0d1520 100%);
    border: 1px solid var(--border);
    border-radius: 20px;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -60px; left: -60px;
    width: 220px; height: 220px;
    background: radial-gradient(circle, rgba(59,255,160,0.12) 0%, transparent 70%);
    pointer-events: none;
}
.hero::after {
    content: '';
    position: absolute;
    bottom: -40px; right: -40px;
    width: 180px; height: 180px;
    background: radial-gradient(circle, rgba(0,212,255,0.10) 0%, transparent 70%);
    pointer-events: none;
}
.hero-icon { font-size: 3rem; margin-bottom: 0.4rem; }
.hero-title {
    font-family: 'Space Mono', monospace !important;
    font-size: 1.9rem !important;
    font-weight: 700 !important;
    letter-spacing: -0.5px;
    background: linear-gradient(90deg, var(--accent), var(--accent2));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 !important;
}
.hero-sub {
    color: var(--muted) !important;
    font-size: 0.95rem;
    margin-top: 0.4rem;
}

/* ── Section label ── */
.section-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 0.5rem;
}

/* ── Model cards ── */
.model-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 12px;
    margin-bottom: 1.5rem;
}
.model-card {
    background: var(--surface);
    border: 1.5px solid var(--border);
    border-radius: var(--radius);
    padding: 1.1rem 1.2rem;
    cursor: pointer;
    transition: all 0.2s ease;
    position: relative;
}
.model-card.selected {
    border-color: var(--accent);
    background: rgba(59,255,160,0.07);
}
.model-card:hover { border-color: #3a4560; }
.model-card-title {
    font-weight: 600;
    font-size: 0.95rem;
    color: var(--text);
}
.model-card-desc {
    font-size: 0.78rem;
    color: var(--muted);
    margin-top: 3px;
}
.model-badge {
    position: absolute;
    top: 10px; right: 10px;
    font-size: 0.65rem;
    font-family: 'Space Mono', monospace;
    letter-spacing: 1px;
    text-transform: uppercase;
    padding: 2px 7px;
    border-radius: 20px;
}
.badge-fast  { background: rgba(59,255,160,0.15); color: var(--accent); }
.badge-smart { background: rgba(0,212,255,0.15);  color: var(--accent2); }

/* ── Upload zone ── */
[data-testid="stFileUploader"] {
    background: var(--surface) !important;
    border: 1.5px dashed var(--border) !important;
    border-radius: var(--radius) !important;
    padding: 1rem !important;
    transition: border-color 0.2s;
}
[data-testid="stFileUploader"]:hover { border-color: var(--accent) !important; }
[data-testid="stFileUploader"] label { color: var(--muted) !important; }

/* ── Uploaded image ── */
[data-testid="stImage"] img {
    border-radius: var(--radius) !important;
    border: 1px solid var(--border) !important;
}

/* ── Predict button ── */
[data-testid="stButton"] > button {
    width: 100%;
    background: linear-gradient(135deg, #1a9e6b, #0a7fa0) !important;
    color: #fff !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.85rem !important;
    font-weight: 700 !important;
    letter-spacing: 1.5px !important;
    text-transform: uppercase !important;
    border: none !important;
    border-radius: var(--radius) !important;
    padding: 0.75rem 1.5rem !important;
    margin-top: 0.8rem;
    transition: opacity 0.2s, transform 0.15s !important;
}
[data-testid="stButton"] > button:hover {
    opacity: 0.88 !important;
    transform: translateY(-1px) !important;
}

/* ── Result cards ── */
.result-row { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin: 1.2rem 0; }
.result-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.1rem 1.3rem;
}
.result-card-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 0.4rem;
}
.result-card-value {
    font-size: 1.35rem;
    font-weight: 600;
    color: var(--accent);
}
.result-card-value.conf { color: var(--accent2); }

/* ── Selectbox hidden (we use custom cards) ── */
[data-testid="stSelectbox"] { display: none !important; }

/* ── Spinner ── */
[data-testid="stSpinner"] { color: var(--accent) !important; }

/* ── Alert tweaks ── */
[data-testid="stAlert"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
}

/* ── Divider ── */
hr { border-color: var(--border) !important; margin: 1.5rem 0 !important; }
</style>
""", unsafe_allow_html=True)


# ── Model name mapping ─────────────────────────────────────────────────────────
MODEL_OPTIONS = {
    "⚡ Fast & Lightweight": "CNN",
    "🎯 Accurate & Advanced": "MobileNetV2",
}
MODEL_BADGES = {
    "⚡ Fast & Lightweight": ("badge-fast",  "FAST"),
    "🎯 Accurate & Advanced": ("badge-smart", "SMART"),
}

# ── Session state ──────────────────────────────────────────────────────────────
if "selected_model_label" not in st.session_state:
    st.session_state.selected_model_label = list(MODEL_OPTIONS.keys())[0]

# ── Hero ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-icon">♻️</div>
    <div class="hero-title">Garbage Classifier</div>
    <div class="hero-sub">Upload an image — get an instant recycling verdict</div>
</div>
""", unsafe_allow_html=True)


# ── Model selector (custom cards via buttons) ──────────────────────────────────
st.markdown('<div class="section-label">Choose a model</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
for col, label in zip([col1, col2], MODEL_OPTIONS.keys()):
    badge_cls, badge_txt = MODEL_BADGES[label]
    selected = st.session_state.selected_model_label == label
    border   = "#3bffa0" if selected else "#262d3d"
    bg       = "rgba(59,255,160,0.07)" if selected else "#161b25"
    with col:
        st.markdown(f"""
        <div class="model-card {'selected' if selected else ''}"
             style="border-color:{border}; background:{bg};">
            <span class="model-badge {badge_cls}">{badge_txt}</span>
            <div class="model-card-title">{label}</div>
            <div class="model-card-desc">
                {"Low latency, great for quick scans" if "Fast" in label
                 else "Higher accuracy, best for precision"}
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button(label, key=f"btn_{label}", use_container_width=True):
            st.session_state.selected_model_label = label
            st.rerun()

selected_model_internal = MODEL_OPTIONS[st.session_state.selected_model_label]

# Hidden selectbox keeps predict_image happy if it reads from session
st.selectbox("model_hidden", list(MODEL_OPTIONS.values()),
             index=list(MODEL_OPTIONS.values()).index(selected_model_internal),
             key="_model_select")

st.markdown("<hr>", unsafe_allow_html=True)

# ── File uploader ──────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">Upload image</div>', unsafe_allow_html=True)
uploaded_file = st.file_uploader(
    "",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed",
)

# ── Prediction flow ────────────────────────────────────────────────────────────
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="", use_container_width=True)

    processed_image = preprocess_image(image, model_name=selected_model_internal)

    if st.button("▶  RUN CLASSIFICATION"):
        with st.spinner("Analyzing image…"):
            predicted_class, confidence, probabilities = predict_image(
                processed_image,
                selected_model_internal,
            )

        # Result cards
        st.markdown(f"""
        <div class="result-row">
            <div class="result-card">
                <div class="result-card-label">Predicted class</div>
                <div class="result-card-value">{predicted_class}</div>
            </div>
            <div class="result-card">
                <div class="result-card-label">Confidence</div>
                <div class="result-card-value conf">{confidence:.1%}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Chart
        st.markdown('<div class="section-label" style="margin-top:1.2rem;">Class probabilities</div>',
                    unsafe_allow_html=True)

        accent_green = "#3bffa0"
        accent_blue  = "#00d4ff"
        bg_surface   = "#161b25"
        border_color = "#262d3d"

        top_idx = int(np.argmax(probabilities))
        bar_colors = [accent_green if i == top_idx else "#2a3450"
                      for i in range(len(classes))]

        fig, ax = plt.subplots(figsize=(7, 3.2))
        fig.patch.set_facecolor(bg_surface)
        ax.set_facecolor(bg_surface)

        bars = ax.bar(classes, probabilities, color=bar_colors,
                      width=0.55, zorder=3, edgecolor="none")

        # Subtle grid
        ax.yaxis.grid(True, color=border_color, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        ax.spines[["top","right","left","bottom"]].set_visible(False)
        ax.tick_params(colors="#6b7a99", labelsize=8)
        ax.set_ylabel("Probability", color="#6b7a99", fontsize=8, labelpad=8)
        plt.xticks(rotation=35, ha="right", color="#6b7a99", fontsize=8)
        ax.set_ylim(0, max(probabilities) * 1.18)

        # Value label on top bar
        ax.text(top_idx, probabilities[top_idx] + max(probabilities) * 0.03,
                f"{probabilities[top_idx]:.0%}",
                ha="center", va="bottom", color=accent_green,
                fontsize=8.5, fontweight="bold",
                fontfamily="monospace")

        legend_patch = mpatches.Patch(color=accent_green, label=f"Top: {predicted_class}")
        ax.legend(handles=[legend_patch], facecolor=bg_surface,
                  edgecolor=border_color, labelcolor="#e8edf5", fontsize=8)

        plt.tight_layout(pad=0.5)
        st.pyplot(fig)
        plt.close(fig)
