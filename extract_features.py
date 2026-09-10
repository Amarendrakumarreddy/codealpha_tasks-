"""
Feature Extraction Module for Speech Emotion Recognition.
Extracts 40 MFCC features from preprocessed audio signals.
Provides reusable functions for both training dataset extraction and real-time audio inference.
"""

import sys
import os
import json
from pathlib import Path
import numpy as np
import pandas as pd
import soundfile as sf
import librosa

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


def preprocess_audio(
    file_path_or_buffer,
    target_sr: int = config.SAMPLE_RATE,
    max_samples: int = config.MAX_AUDIO_SAMPLES,
) -> np.ndarray:
    """
    Loads, resamples to target_sr, converts to mono, normalizes amplitude,
    and applies padding or truncation to fixed length.

    Parameters:
        file_path_or_buffer: Path to audio file or file-like object/bytes.
        target_sr: Desired sampling rate (default 22050).
        max_samples: Fixed number of audio samples (default 66150 = 3.0s).

    Returns:
        np.ndarray: 1D float32 audio waveform of shape (max_samples,).
    """
    try:
        # Load audio with librosa (handles mono conversion & resampling)
        y, sr = librosa.load(file_path_or_buffer, sr=target_sr, mono=True)
    except Exception as e:
        # Fallback to soundfile if librosa load faces format quirks
        try:
            data, sr = sf.read(file_path_or_buffer)
            if data.ndim > 1:
                data = np.mean(data, axis=1)  # Mono conversion
            if sr != target_sr:
                y = librosa.resample(data, orig_sr=sr, target_sr=target_sr)
            else:
                y = data
        except Exception as sf_err:
            raise ValueError(f"Could not load audio file: {e} | SoundFile fallback error: {sf_err}")

    # Amplitude normalization (peak normalization)
    max_val = np.max(np.abs(y))
    if max_val > 1e-6:
        y = y / max_val

    # Fixed length handling (padding or truncation)
    current_length = len(y)
    if current_length < max_samples:
        # Pad with zeros equally at beginning and end
        pad_total = max_samples - current_length
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        y = np.pad(y, (pad_left, pad_right), mode="constant")
    elif current_length > max_samples:
        # Center-crop audio
        excess = current_length - max_samples
        start = excess // 2
        y = y[start : start + max_samples]

    return y.astype(np.float32)


def extract_mfcc(
    y: np.ndarray,
    sr: int = config.SAMPLE_RATE,
    n_mfcc: int = config.N_MFCC,
    n_fft: int = config.N_FFT,
    hop_length: int = config.HOP_LENGTH,
    normalize_cmvn: bool = True,
) -> np.ndarray:
    """
    Extracts Mel-Frequency Cepstral Coefficients (MFCC) from a 1D audio waveform.

    Parameters:
        y: 1D audio waveform.
        sr: Audio sampling rate.
        n_mfcc: Number of MFCC coefficients to compute (default 40).
        n_fft: FFT window size (default 2048).
        hop_length: Hop length between frames (default 512).
        normalize_cmvn: Whether to apply Cepstral Mean & Variance Normalization.

    Returns:
        np.ndarray: 2D feature matrix of shape (time_steps, n_mfcc).
    """
    # mfcc shape: (n_mfcc, time_steps)
    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop_length,
    )

    # Transpose to (time_steps, n_mfcc) for sequential modeling
    mfcc = mfcc.T

    # Apply Cepstral Mean and Variance Normalization (CMVN)
    if normalize_cmvn:
        mean = np.mean(mfcc, axis=0, keepdims=True)
        std = np.std(mfcc, axis=0, keepdims=True) + 1e-8
        mfcc = (mfcc - mean) / std

    return mfcc.astype(np.float32)


def extract_audio_features(file_path_or_buffer) -> np.ndarray:
    """
    End-to-end feature extraction for a single audio file or stream.
    Reusable for training, batch evaluation, and real-time Streamlit inference.

    Returns:
        np.ndarray: Preprocessed MFCC feature matrix of shape (time_steps, n_mfcc).
    """
    y = preprocess_audio(file_path_or_buffer)
    mfcc = extract_mfcc(y)
    return mfcc


def extract_all_features(metadata_csv_path=None):
    """
    Extracts features for all audio files registered in dataset_metadata.csv
    and saves X_features.npy and y_labels.npy to data/processed/.
    """
    print("=" * 60)
    print("  STEP 2: AUDIO FEATURE EXTRACTION (MFCC)")
    print("=" * 60)

    csv_path = Path(metadata_csv_path) if metadata_csv_path else config.DATASET_METADATA_PATH
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Metadata CSV not found at {csv_path}. Please run prepare_dataset.py first."
        )

    df = pd.read_csv(csv_path)
    total_files = len(df)
    print(f"Extracting 40 MFCC features for {total_files} audio samples...")
    print(f"Sample Rate: {config.SAMPLE_RATE} Hz | Duration: {config.DURATION}s | Hop: {config.HOP_LENGTH}")

    features_list = []
    labels_list = []
    valid_indices = []

    for i, row in df.iterrows():
        rel_path = row["relative_path"]
        abs_path = config.PROJECT_ROOT / rel_path

        # If file not in project root, check as absolute path
        if not abs_path.exists():
            abs_path = Path(rel_path)

        if not abs_path.exists():
            print(f"[Warning] File not found: {abs_path}, skipping.")
            continue

        try:
            feats = extract_audio_features(abs_path)
            features_list.append(feats)
            labels_list.append(row["emotion"])
            valid_indices.append(i)
        except Exception as e:
            print(f"[Warning] Failed to process {abs_path.name}: {e}")

        if (i + 1) % 200 == 0 or (i + 1) == total_files:
            print(f"  Processed {i + 1}/{total_files} audio files...")

    X = np.array(features_list, dtype=np.float32)
    y = np.array(labels_list)

    print(f"\nFeature extraction complete!")
    print(f"Features array shape (X): {X.shape} (samples, time_steps, n_mfcc)")
    print(f"Labels array shape   (y): {y.shape}")

    # Save to disk
    np.save(config.FEATURES_X_PATH, X)
    np.save(config.LABELS_Y_PATH, y)
    print(f"Saved features to: {config.FEATURES_X_PATH}")
    print(f"Saved labels to:   {config.LABELS_Y_PATH}")

    # Save preprocessor configuration
    config_dict = {
        "sample_rate": config.SAMPLE_RATE,
        "duration": config.DURATION,
        "max_samples": config.MAX_AUDIO_SAMPLES,
        "n_mfcc": config.N_MFCC,
        "n_fft": config.N_FFT,
        "hop_length": config.HOP_LENGTH,
        "time_steps": int(X.shape[1]),
        "classes": sorted(list(set(y))),
    }
    with open(config.PREPROCESSOR_CONFIG_PATH, "w") as f:
        json.dump(config_dict, f, indent=4)
    print(f"Saved preprocessor config to: {config.PREPROCESSOR_CONFIG_PATH}")
    print("=" * 60)
    return X, y


if __name__ == "__main__":
    extract_all_features()
