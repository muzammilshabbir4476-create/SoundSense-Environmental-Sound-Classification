import json
import os
import html

import librosa
import numpy as np
import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
import streamlit.components.v1 as components

st.set_page_config(
    page_title="SoundSense | AI Audio Intelligence",
    page_icon="🔊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------- DESIGN ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

:root {
    --bg: #080d19;
    --panel: #111a2c;
    --line: #24314b;
    --muted: #98a8c4;
    --blue: #60a5fa;
    --cyan: #67e8f9;
}

.stApp {
    background:
        radial-gradient(ellipse at 10% 0%, #172b4d 0%, transparent 38%),
        radial-gradient(ellipse at 90% 12%, #15254a 0%, transparent 32%),
        var(--bg);
    color: #f4f7ff;
    font-family: 'DM Sans', sans-serif;
}

.block-container {
    max-width: 1160px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

h1, h2, h3 {
    font-family: 'Space Grotesk', sans-serif !important;
    color: #f4f7ff !important;
}

header[data-testid="stHeader"] {
    background: transparent;
}

.brand {
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 700;
    font-size: 1.55rem;
    letter-spacing: -1px;
}

.brand span { color: var(--cyan); }

.pill {
    display: inline-block;
    border: 1px solid #315078;
    border-radius: 30px;
    padding: 6px 12px;
    color: var(--cyan);
    background: #10243b;
    font-size: .78rem;
    letter-spacing: .4px;
}

.hero {
    padding: 42px 0 25px 0;
}

.hero h1 {
    font-size: clamp(2.4rem, 5vw, 4.4rem) !important;
    line-height: 1.06;
    letter-spacing: -2.5px;
    margin: 18px 0;
}

.gradient {
    background: linear-gradient(90deg, #60a5fa, #67e8f9);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.sub {
    color: #aab8d0;
    font-size: 1.05rem;
    max-width: 650px;
    line-height: 1.8;
}

.panel {
    background: linear-gradient(145deg, #141f34, #0e1728);
    border: 1px solid var(--line);
    border-radius: 20px;
    padding: 24px;
    margin: 8px 0 18px 0;
}

.stat {
    background: #111a2c;
    border: 1px solid var(--line);
    border-radius: 16px;
    padding: 18px;
    min-height: 100px;
}

.stat-label {
    color: var(--muted);
    font-size: .83rem;
    margin-bottom: 8px;
}

.stat-value {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.65rem;
    font-weight: 700;
    color: #f4f7ff;
}


/* Premium audio upload area */
div[data-testid="stFileUploader"] {
    background: linear-gradient(135deg, #101e34, #11172a);
    border: 1px dashed #60a5fa;
    border-radius: 18px;
    padding: 18px;
    transition: all 0.25s ease;
}

div[data-testid="stFileUploader"]:hover {
    border-color: #67e8f9;
    background: linear-gradient(135deg, #152844, #111a30);
}

div[data-testid="stFileUploader"] section {
    background: transparent;
    border: none;
}

div[data-testid="stFileUploader"] button {
    background: #1d4ed8 !important;
    color: #ffffff !important;
    border: 1px solid #60a5fa !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
}

div[data-testid="stFileUploader"] button:hover {
    background: #2563eb !important;
    color: #ffffff !important;
}

div[data-testid="stFileUploader"] small {
    color: #b8c7df !important;
}


div.stButton > button {
    width: 100%;
    min-height: 48px;
    border-radius: 12px;
    border: 1px solid #60a5fa;
    background: linear-gradient(90deg, #2563eb, #0891b2);
    color: white;
    font-weight: 700;
    transition: .2s ease;
}

div.stButton > button:hover {
    border-color: #a5f3fc;
    color: white;
    filter: brightness(1.12);
}

div[data-testid="stMetric"] {
    background: #111a2c;
    border: 1px solid var(--line);
    border-radius: 15px;
    padding: 16px;
}

div[data-testid="stMetricLabel"] { color: var(--muted); }
div[data-testid="stMetricValue"] { color: #67e8f9; }

hr { border-color: var(--line); }

footer { visibility: hidden; }

@media (max-width: 700px) {
    .hero { padding-top: 25px; }
    .panel { padding: 16px; }
}
</style>
""", unsafe_allow_html=True)

# ---------- HEADER ----------
header_left, header_right = st.columns([3, 1])

with header_left:
    st.markdown(
        '<div class="brand">🔊 Sound<span>Sense</span></div>',
        unsafe_allow_html=True
    )

with header_right:
    st.markdown(
        '<div style="text-align:right;padding-top:8px">'
        '<span class="pill">● AI SYSTEM</span></div>',
        unsafe_allow_html=True
    )

st.markdown("""
<div class="hero">
    <span class="pill">ENVIRONMENTAL AUDIO INTELLIGENCE</span>
    <h1>Hear the world.<br><span class="gradient">Understand every sound.</span></h1>
    <p class="sub">
        Upload an audio clip and let our YAMNet-powered deep learning
        model identify the environmental sound in seconds.
    </p>
</div>
""", unsafe_allow_html=True)

# ---------- PROJECT STATS ----------
c1, c2, c3 = st.columns(3)

for col, label, value in [
    (c1, "TEST ACCURACY", "90%"),
    (c2, "SOUND CATEGORIES", "10"),
    (c3, "FEATURE EXTRACTOR", "YAMNet"),
]:
    with col:
        st.markdown(
            f'<div class="stat"><div class="stat-label">{label}</div>'
            f'<div class="stat-value">{value}</div></div>',
            unsafe_allow_html=True
        )

st.write("")

# ---------- MODEL FILES ----------
MODEL_DIR = "model"
MODEL_PATH = os.path.join(
    MODEL_DIR, "soundsense_yamnet_classifier.keras"
)
NORM_PATH = os.path.join(
    MODEL_DIR, "soundsense_normalization.npz"
)
CATEGORIES_PATH = os.path.join(
    MODEL_DIR, "soundsense_categories.json"
)

@st.cache_resource
def load_models():
    model = tf.keras.models.load_model(MODEL_PATH)
    norm = np.load(NORM_PATH)

    with open(CATEGORIES_PATH, "r") as f:
        categories = json.load(f)

    yamnet = hub.load("https://tfhub.dev/google/yamnet/1")

    return model, norm["mean"], norm["std"], categories, yamnet

try:
    model, mean, std, categories, yamnet = load_models()
except Exception as e:
    st.error("Model load nahi ho saka. Please check model files and dependencies.")
    st.exception(e)
    st.stop()

st.markdown(
    '<span class="pill">✓ MODEL READY</span>',
    unsafe_allow_html=True
)

# ---------- MAIN WORKSPACE ----------
left, right = st.columns([1.05, 0.95], gap="large")

with left:
    st.markdown("### 🎧 Audio workspace")
    st.markdown(
        '<div class="panel"><h3>Upload your audio</h3>'
        '<p style="color:#98a8c4">Choose a recording to begin classification. '
        'WAV, MP3, FLAC and OGG are supported.</p></div>',
        unsafe_allow_html=True
    )

    audio_file = st.file_uploader(
        "Select audio file",
        type=["wav", "mp3", "flac", "ogg"],
        label_visibility="collapsed"
    )

    
if audio_file is not None:
    audio_bytes = audio_file.getvalue()
    st.audio(audio_bytes)
    
        st.markdown(
            f'<div class="panel"><div class="stat-label">SELECTED FILE</div>'
            f'<b>{html.escape(audio_file.name)}</b>'
            f'<p style="color:#98a8c4;margin-bottom:0">'
            f'{audio_file.size / 1024:.1f} KB</p></div>',
            unsafe_allow_html=True
        )

    classify = st.button(
        "✦  Analyze audio",
        type="primary",
        disabled=audio_file is None,
        use_container_width=True
    )

with right:
    st.markdown("### ✨ How it works")

    for number, title, description in [
        ("01", "Audio input", "Your recording is loaded as mono audio at 16 kHz."),
        ("02", "Feature extraction", "Pretrained YAMNet converts audio into 1024 features."),
        ("03", "AI classification", "The trained neural network ranks 10 sound categories."),
    ]:
        st.markdown(
            f'<div class="panel"><span class="pill">{number}</span>'
            f'<h3 style="margin-top:14px">{title}</h3>'
            f'<p style="color:#98a8c4;margin-bottom:0">{description}</p></div>',
            unsafe_allow_html=True
        )

# ---------- PREDICTION ----------
if audio_file is not None and classify:
    with st.spinner("Listening to audio and analyzing sound patterns..."):
        try:
            audio, sr = librosa.load(
                audio_file, sr=16000, mono=True
            )

            if len(audio) == 0:
                st.error("Audio file is empty.")
                st.stop()

            _, frame_embeddings, _ = yamnet(
                tf.convert_to_tensor(audio, dtype=tf.float32)
            )

            embedding = tf.reduce_mean(
                frame_embeddings, axis=0
            ).numpy()

            embedding = (embedding - mean) / (std + 1e-8)

            probabilities = model.predict(
                embedding.reshape(1, 1024), verbose=0
            )[0]

            best = int(np.argmax(probabilities))
            ranking = np.argsort(probabilities)[::-1][:3]

            st.markdown("---")
            st.markdown(
                '<div id="prediction-results"></div>',
                unsafe_allow_html=True
            )
            st.markdown("## 🎯 Analysis results")

            # Scroll down after the results heading has been rendered.
            components.html(
                """
                <script>
                const scrollToResults = () => {
                    try {
                        const doc = window.parent.document;
                        const target = doc.getElementById("prediction-results");
                        if (target) {
                            target.scrollIntoView({
                                behavior: "smooth",
                                block: "start"
                            });
                        }
                    } catch (e) {
                        // Keep the prediction usable if browser access is restricted.
                    }
                };
                setTimeout(scrollToResults, 500);
                setTimeout(scrollToResults, 1200);
                </script>
                """,
                height=0,
            )

            result_left, result_right = st.columns([1, 1])

            with result_left:
                st.markdown(
                    '<div class="panel"><div class="stat-label">'
                    'DETECTED SOUND</div>'
                    f'<div class="stat-value" style="font-size:2rem">'
                    f'{html.escape(categories[best].replace("_", " ").title())}'
                    '</div><p style="color:#98a8c4">'
                    'Most likely category predicted by the model</p></div>',
                    unsafe_allow_html=True
                )

            with result_right:
                st.markdown(
                    '<div class="panel"><div class="stat-label">'
                    'MODEL CONFIDENCE</div>'
                    f'<div class="stat-value" style="font-size:2.5rem">'
                    f'{probabilities[best] * 100:.2f}%</div>'
                    '<p style="color:#98a8c4">Top predicted class score</p></div>',
                    unsafe_allow_html=True
                )

            st.markdown("### Top 3 predictions")

            for idx in ranking:
                score = float(probabilities[idx]) * 100
                label = categories[idx].replace("_", " ").title()

                st.markdown(
                    f'<div style="display:flex;justify-content:space-between;'
                    f'margin-top:16px;margin-bottom:6px">'
                    f'<b>{html.escape(label)}</b>'
                    f'<span style="color:#67e8f9">{score:.2f}%</span></div>',
                    unsafe_allow_html=True
                )
                st.progress(min(max(score / 100, 0.0), 1.0))

            st.caption(
                "Confidence is the model's predicted score, not a guarantee "
                "that the classification is correct. Predictions are limited "
                "to the 10 supported categories."
            )

        except Exception as e:
            st.error(f"Prediction failed: {e}")

# ---------- FOOTER ----------
st.markdown("---")
st.markdown(
    '<div style="text-align:center;color:#98a8c4;padding:12px">'
    '<b style="color:#e2e8f0">SoundSense</b> · Environmental Sound '
    'Classification · Powered by YAMNet + Deep Learning'
    '</div>',
    unsafe_allow_html=True
)
