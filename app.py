"""
Streamlit Web Application for AI Speech Emotion Recognition (SER).
Provides interactive audio upload, microphone recording, sample audio testing,
visual waveform & spectrogram display, predicted emotion banner, confidence metrics,
and class probability distribution charts.
"""

import sys
import os
import io
import time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

import streamlit as st

# Configure Streamlit page layout and title
st.set_page_config(
    page_title="AI Speech Emotion Recognition",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern polished design
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.3rem;
        font-weight: 800;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .emotion-card {
        padding: 1.5rem;
        border-radius: 14px;
        background: linear-gradient(135deg, #f0f4ff 0%, #e0e7ff 100%);
        border: 2px solid #818cf8;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .emotion-badge {
        font-size: 2.8rem;
        font-weight: 800;
        letter-spacing: 1px;
        color: #1e1b4b;
    }
    .confidence-meter {
        font-size: 1.25rem;
        font-weight: 600;
        color: #3730a3;
        margin-top: 0.5rem;
    }
    .metric-box {
        background: #ffffff;
        border-radius: 10px;
        padding: 1rem;
        border: 1px solid #e5e7eb;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def load_config_and_modules():
    """Safely import project modules and verify model availability."""
    try:
        import config
        from scripts.predict import predict_emotion, load_artifacts
        from scripts.extract_features import preprocess_audio, extract_mfcc
        import librosa
        return config, predict_emotion, load_artifacts, preprocess_audio, extract_mfcc, librosa
    except Exception as e:
        st.error(f"⚠️ Initialization Error: {e}")
        return None, None, None, None, None, None


config, predict_emotion, load_artifacts, preprocess_audio, extract_mfcc, librosa = load_config_and_modules()


def render_audio_visuals(audio_bytes):
    """Visualizes the audio waveform and Mel-Spectrogram."""
    try:
        y, sr = librosa.load(io.BytesIO(audio_bytes), sr=config.SAMPLE_RATE, mono=True)
        time_axis = np.linspace(0, len(y) / sr, num=len(y))

        fig, axes = plt.subplots(2, 1, figsize=(10, 4.5), sharex=False)
        fig.patch.set_facecolor("#fafafa")

        # 1. Waveform
        axes[0].plot(time_axis, y, color="#2563eb", linewidth=0.8)
        axes[0].set_title("Speech Audio Waveform", fontsize=11, fontweight="bold", pad=6)
        axes[0].set_ylabel("Amplitude", fontsize=9)
        axes[0].grid(True, linestyle=":", alpha=0.6)
        axes[0].set_xlim(0, max(time_axis))

        # 2. Mel-Spectrogram
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=config.N_FFT, hop_length=config.HOP_LENGTH)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        img = librosa.display.specshow(
            mel_spec_db,
            sr=sr,
            hop_length=config.HOP_LENGTH,
            x_axis="time",
            y_axis="mel",
            ax=axes[1],
            cmap="viridis",
        )
        axes[1].set_title("Mel Spectrogram (Frequency Representation)", fontsize=11, fontweight="bold", pad=6)
        axes[1].set_xlabel("Time (seconds)", fontsize=9)
        axes[1].set_ylabel("Hz", fontsize=9)
        fig.colorbar(img, ax=axes[1], format="%+2.0f dB", pad=0.02)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    except Exception as e:
        st.warning(f"Waveform visualizer preview unavailable: {e}")


def main():
    # Header Banner
    st.markdown('<div class="main-title">🎙️ AI Speech Emotion Recognition</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Classify human vocal emotions in real time using Mel-Frequency Cepstral Coefficients (MFCC) & Deep Learning</div>',
        unsafe_allow_html=True,
    )

    # Check Model Existence
    model_exists = config.MODEL_PATH.exists() if config else False

    if not model_exists:
        st.warning(
            "⚠️ **Model checkpoint not found!**\n\n"
            f"The trained model is expected at `{config.MODEL_PATH}`.\n\n"
            "To train the model, open a terminal in the project root and run:\n"
            "```bash\n"
            "python scripts/prepare_dataset.py\n"
            "python scripts/extract_features.py\n"
            "python scripts/train.py\n"
            "```"
        )

    # Sidebar: Input source selection & model status
    st.sidebar.title("🎛️ Audio Controls")

    # Status Pill
    if model_exists:
        st.sidebar.success("✅ Model Ready: CNN + Bi-LSTM")
    else:
        st.sidebar.error("❌ Model Missing")

    input_mode = st.sidebar.radio(
        "Select Audio Source:",
        ["Upload File (WAV/MP3)", "Use Sample Audio", "Record Microphone"],
    )

    audio_bytes = None
    audio_source_label = ""

    # Option 1: Upload File
    if input_mode == "Upload File (WAV/MP3)":
        uploaded_file = st.sidebar.file_uploader(
            "Choose a voice audio file",
            type=["wav", "mp3", "ogg", "flac", "m4a"],
            help="Upload a speech recording (recommended duration: 2-5 seconds)",
        )
        if uploaded_file is not None:
            audio_bytes = uploaded_file.read()
            audio_source_label = f"Uploaded: {uploaded_file.name}"

    # Option 2: Preloaded Sample Audio
    elif input_mode == "Use Sample Audio":
        sample_files = []
        if config.RAW_DATA_DIR.exists():
            sample_files = list(config.RAW_DATA_DIR.rglob("*.wav"))[:20]

        if sample_files:
            sample_options = {f.name: f for f in sample_files}
            selected_sample_name = st.sidebar.selectbox("Choose sample audio clip:", list(sample_options.keys()))
            if selected_sample_name:
                selected_sample_path = sample_options[selected_sample_name]
                with open(selected_sample_path, "rb") as f:
                    audio_bytes = f.read()
                audio_source_label = f"Sample Clip: {selected_sample_name}"
        else:
            st.sidebar.info("No raw sample files found in data/raw/. Please run scripts/prepare_dataset.py.")

    # Option 3: Record Microphone
    elif input_mode == "Record Microphone":
        st.sidebar.markdown("**Live Voice Recording:**")
        # Streamlit 1.30+ audio_input if supported
        try:
            recorded_audio = st.sidebar.audio_input("Record your voice (say a short phrase)")
            if recorded_audio is not None:
                audio_bytes = recorded_audio.read()
                audio_source_label = "Live Voice Recording"
        except Exception:
            st.sidebar.warning("Live audio_input widget requires a modern browser and permissions.")

    # Main Area Layout
    tab1, tab2, tab3 = st.tabs(["📊 Emotion Analysis", "📈 Audio Visualization", "ℹ️ About Model & System"])

    with tab1:
        if audio_bytes is not None:
            col1, col2 = st.columns([1, 1])

            with col1:
                st.subheader("🔊 Audio Player")
                st.audio(audio_bytes, format="audio/wav")
                st.caption(f"Source: {audio_source_label}")

                # Quick file validation
                if len(audio_bytes) < 1000:
                    st.error("Audio recording is too short or empty. Please record/upload a valid speech sample.")
                    return

            if model_exists:
                # Run Emotion Prediction
                with st.spinner("Analyzing speech acoustic features..."):
                    result = predict_emotion(io.BytesIO(audio_bytes))

                if result["success"]:
                    emotion = result["emotion"]
                    confidence = result["confidence"]
                    probabilities = result["probabilities"]

                    emoji = config.EMOTION_EMOJIS.get(emotion.lower(), "🎙️")
                    card_color = config.EMOTION_COLORS.get(emotion.lower(), "#3b82f6")

                    with col2:
                        st.subheader("🎯 Emotion Prediction")
                        st.markdown(
                            f"""
                            <div class="emotion-card" style="border-color: {card_color};">
                                <div style="font-size: 1.1rem; color: #4b5563; font-weight: 600;">DETECTED EMOTION</div>
                                <div class="emotion-badge">{emoji} {emotion}</div>
                                <div class="confidence-meter">Confidence: {confidence:.1f}%</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                    st.markdown("---")
                    st.subheader("📊 Emotion Probabilities Distribution")

                    # Probabilities Bar Chart
                    prob_df = pd.DataFrame(
                        {
                            "Emotion": [f"{config.EMOTION_EMOJIS.get(e.lower(), '')} {e}" for e in probabilities.keys()],
                            "Probability (%)": list(probabilities.values()),
                        }
                    )

                    # Display metrics summary row
                    metric_cols = st.columns(min(4, len(probabilities)))
                    for i, (emo_name, prob_val) in enumerate(list(probabilities.items())[:4]):
                        with metric_cols[i % len(metric_cols)]:
                            emo_emoji = config.EMOTION_EMOJIS.get(emo_name.lower(), "•")
                            st.metric(label=f"{emo_emoji} {emo_name}", value=f"{prob_val:.1f}%")

                    st.bar_chart(
                        prob_df.set_index("Emotion"),
                        color="#4f46e5",
                        use_container_width=True,
                    )
                else:
                    st.error(f"Prediction error: {result['error']}")
            else:
                st.info("Train the model to view emotion predictions here.")
        else:
            st.info("👈 Please select an audio input option from the sidebar (Upload, Sample, or Microphone) to begin emotion analysis.")

    with tab2:
        st.subheader("📈 Audio Signal & Spectral Features")
        if audio_bytes is not None:
            render_audio_visuals(audio_bytes)
        else:
            st.info("Upload or select an audio sample to inspect its waveform and spectrogram.")

    with tab3:
        st.subheader("🧠 System Architecture & Methodology")
        st.markdown(
            """
            ### Pipeline Overview:
            1. **Audio Input**: WAV / MP3 voice clip (speech recording).
            2. **Preprocessing**:
               - Resampled to `22,050 Hz` mono signal.
               - Peak amplitude normalization.
               - Center-cropping / zero-padding to standard `3.0 seconds` (66,150 samples).
            3. **Feature Extraction**:
               - **40 Mel-Frequency Cepstral Coefficients (MFCCs)** per time step.
               - Frame length: 2048, Hop length: 512.
               - Cepstral Mean & Variance Normalization (CMVN) for acoustic robustness.
            4. **Deep Learning Model (CPU Optimized)**:
               - **Conv1D layers (128 & 256 filters)** with Batch Normalization and Dropout for spatial acoustic pattern extraction.
               - **Bidirectional LSTM (64 units)** for temporal context across spoken phonemes.
               - **Dense Classification Head** with Softmax activation.
            5. **Dataset**:
               - Ryerson Audio-Visual Database of Emotional Speech and Song (**RAVDESS**), with automatic support for TESS and EMO-DB.
            """
        )

        if config.CONFUSION_MATRIX_PATH.exists():
            st.markdown("### Model Evaluation Matrix")
            st.image(str(config.CONFUSION_MATRIX_PATH), caption="Test Set Confusion Matrix", use_column_width=True)

        if config.TRAINING_HISTORY_PATH.exists():
            st.markdown("### Training & Validation Curves")
            st.image(str(config.TRAINING_HISTORY_PATH), caption="Training History (Loss & Accuracy)", use_column_width=True)


if __name__ == "__main__":
    main()
