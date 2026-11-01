
import json
import os

import librosa
import numpy as np
import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub

st.set_page_config(
    page_title="SoundSense",
    page_icon="🔊",
    layout="centered"
)

st.title("🔊 SoundSense")
st.subheader("Environmental Sound Classification")
st.write(
    "Upload an audio recording to identify its environmental sound."
)

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
    st.success("Model loaded successfully.")
except Exception as e:
    st.error(
        "Could not load the model. Check that all three files "
        "exist in the model/ directory and dependencies are installed."
    )
    st.exception(e)
    st.stop()

audio_file = st.file_uploader(
    "Choose an audio file",
    type=["wav", "mp3", "flac", "ogg"]
)

if audio_file is not None:
    st.audio(audio_file)

    if st.button("Classify Sound", type="primary"):
        try:
            audio, sr = librosa.load(
                audio_file, sr=16000, mono=True
            )

            if len(audio) == 0:
                st.error("The uploaded audio file is empty.")
                st.stop()

            _, frame_embeddings, _ = yamnet(audio)
            embedding = tf.reduce_mean(
                frame_embeddings, axis=0
            ).numpy()

            embedding = (embedding - mean) / std
            probabilities = model.predict(
                embedding.reshape(1, 1024), verbose=0
            )[0]

            best = int(np.argmax(probabilities))

            st.success(
                f"Predicted sound: {categories[best]}"
            )
            st.metric(
                "Model confidence",
                f"{probabilities[best] * 100:.2f}%"
            )

            st.write("### Top 3 predictions")
            for idx in np.argsort(probabilities)[-3:][::-1]:
                st.write(
                    f"**{categories[idx]}** — "
                    f"{probabilities[idx] * 100:.2f}%"
                )

            st.caption(
                "Predictions are limited to the 10 classes "
                "supported by this model."
            )

        except Exception as e:
            st.error(f"Prediction failed: {e}")
