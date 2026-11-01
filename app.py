    import json
import os
import html

import librosa
import numpy as np
import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
import streamlit.components.v1 as components

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
