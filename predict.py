"""
Inference & Prediction Module for Speech Emotion Recognition.
Accepts WAV/MP3 audio, extracts MFCC features, loads trained model,
and outputs predicted emotion, confidence score, and class probability distribution.
"""

import sys
import os
import argparse
import joblib
from pathlib import Path
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

import config

# Set UTF-8 encoding for Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Global cache for inference artifacts
_CACHED_MODEL = None
_CACHED_ENCODER = None


def load_artifacts(model_path: Path = None, encoder_path: Path = None):
    """
    Loads and caches the trained Keras model and LabelEncoder.
    Provides clear, human-friendly error messages if models are not yet trained.
    """
    global _CACHED_MODEL, _CACHED_ENCODER

    model_p = Path(model_path) if model_path else config.MODEL_PATH
    encoder_p = Path(encoder_path) if encoder_path else config.LABEL_ENCODER_PATH

    if not model_p.exists():
        raise FileNotFoundError(
            f"[Error] Trained model not found at '{model_p}'.\n"
            "Please train the model first by running: python scripts/train.py"
        )

    if not encoder_p.exists():
        raise FileNotFoundError(
            f"[Error] Label encoder not found at '{encoder_p}'.\n"
            "Please run the training pipeline first: python scripts/train.py"
        )

    if _CACHED_MODEL is None or model_path is not None:
        import tensorflow as tf
        from tensorflow import keras
        _CACHED_MODEL = keras.models.load_model(model_p)

    if _CACHED_ENCODER is None or encoder_path is not None:
        _CACHED_ENCODER = joblib.load(encoder_p)

    return _CACHED_MODEL, _CACHED_ENCODER


def validate_audio_input(audio_input):
    """Validates the input audio file or byte buffer before processing."""
    if audio_input is None:
        raise ValueError("Audio input cannot be None.")

    # If file path (str or Path)
    if isinstance(audio_input, (str, Path)):
        p = Path(audio_input)
        if not p.exists():
            raise FileNotFoundError(f"Audio file '{p}' does not exist.")
        if p.stat().st_size == 0:
            raise ValueError(f"Audio file '{p}' is empty (0 bytes).")
        valid_extensions = {".wav", ".mp3", ".ogg", ".flac", ".m4a"}
        if p.suffix.lower() not in valid_extensions:
            raise ValueError(
                f"Unsupported audio format '{p.suffix}'. Supported formats: {', '.join(sorted(valid_extensions))}"
            )
    elif hasattr(audio_input, "read"):
        # File-like object (BytesIO or Streamlit UploadedFile)
        audio_input.seek(0, os.SEEK_END)
        size = audio_input.tell()
        audio_input.seek(0)
        if size == 0:
            raise ValueError("Uploaded audio file is empty (0 bytes).")


def predict_emotion(audio_input, model=None, encoder=None) -> dict:
    """
    Accepts an audio file path or file-like buffer, extracts features,
    and returns emotion prediction with confidence and probability distribution.

    Parameters:
        audio_input: Path or file-like buffer containing audio.
        model: Optional preloaded Keras model.
        encoder: Optional preloaded LabelEncoder.

    Returns:
        dict:
            - emotion: str (e.g. 'HAPPY')
            - confidence: float (e.g. 91.4)
            - probabilities: dict mapping emotion to percentage
            - success: bool
            - error: str or None
    """
    from scripts.extract_features import extract_audio_features

    try:
        # 1. Validate Input
        validate_audio_input(audio_input)

        # 2. Load Artifacts
        if model is None or encoder is None:
            model, encoder = load_artifacts()

        # 3. Extract Features
        # Output shape: (time_steps, n_mfcc) -> (130, 40)
        features = extract_audio_features(audio_input)

        # 4. Batch Dimension: (1, time_steps, n_mfcc)
        x_input = np.expand_dims(features, axis=0)

        # 5. Model Inference
        probs = model.predict(x_input, verbose=0)[0]

        # 6. Parse Results
        pred_idx = int(np.argmax(probs))
        classes = list(encoder.classes_)
        predicted_class = classes[pred_idx]
        confidence = float(probs[pred_idx] * 100.0)

        # Build full probability dictionary sorted descending
        prob_dict = {}
        for cls_name, prob in zip(classes, probs):
            prob_dict[cls_name.capitalize()] = round(float(prob * 100.0), 2)

        sorted_probs = dict(sorted(prob_dict.items(), key=lambda item: item[1], reverse=True))

        return {
            "emotion": predicted_class.upper(),
            "confidence": round(confidence, 1),
            "probabilities": sorted_probs,
            "success": True,
            "error": None,
        }

    except Exception as e:
        return {
            "emotion": "UNKNOWN",
            "confidence": 0.0,
            "probabilities": {},
            "success": False,
            "error": str(e),
        }


def main():
    parser = argparse.ArgumentParser(description="Classify speech emotion from an audio file")
    parser.add_argument("--file", "-f", type=str, required=True, help="Path to WAV or MP3 audio file")
    args = parser.parse_args()

    result = predict_emotion(args.file)

    if not result["success"]:
        print(f"\n[Error during prediction]: {result['error']}", file=sys.stderr)
        sys.exit(1)

    print("\n" + "=" * 40)
    print(f"Emotion:    {result['emotion']}")
    print(f"Confidence: {result['confidence']}%")
    print("=" * 40)
    print("Probabilities:")
    for emo, prob in result["probabilities"].items():
        bar = "█" * int(prob / 5)
        print(f"  {emo:<10}: {prob:>5.1f}%  {bar}")
    print("=" * 40 + "\n")


if __name__ == "__main__":
    main()
