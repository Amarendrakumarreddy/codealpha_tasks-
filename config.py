"""
Configuration module for Speech Emotion Recognition (SER) Project.
Centralizes paths, audio preprocessing parameters, model architecture settings,
and label mapping rules.
"""

import os
from pathlib import Path

# ==========================================
# Directory Paths (Relative to Project Root)
# ==========================================
PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

# Saved Model & Artefacts Paths
MODEL_PATH = MODELS_DIR / "emotion_model.keras"
LABEL_ENCODER_PATH = MODELS_DIR / "label_encoder.pkl"
SCALER_PATH = MODELS_DIR / "feature_scaler.pkl"
PREPROCESSOR_CONFIG_PATH = MODELS_DIR / "preprocessor_config.json"

# Processed Feature Files
FEATURES_X_PATH = PROCESSED_DATA_DIR / "X_features.npy"
LABELS_Y_PATH = PROCESSED_DATA_DIR / "y_labels.npy"
DATASET_METADATA_PATH = PROCESSED_DATA_DIR / "dataset_metadata.csv"

# Results Output Paths
CONFUSION_MATRIX_PATH = RESULTS_DIR / "confusion_matrix.png"
TRAINING_HISTORY_PATH = RESULTS_DIR / "training_history.png"
CLASSIFICATION_REPORT_PATH = RESULTS_DIR / "classification_report.txt"

# Candidate External Locations for Dataset Discovery
EXTERNAL_SEARCH_PATHS = [
    Path(r"c:\Users\DELL\Downloads\archive (4)"),
    Path(r"c:\Users\DELL\Downloads\archive (4)\audio_speech_actors_01-24"),
    PROJECT_ROOT.parent / "archive (4)",
]

# Ensure required project directories exist
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, RESULTS_DIR, NOTEBOOKS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ==========================================
# Audio Processing Parameters
# ==========================================
SAMPLE_RATE = 22050       # Standard sampling rate for librosa speech analysis
DURATION = 3.0            # Target audio clip duration in seconds
MAX_AUDIO_SAMPLES = int(SAMPLE_RATE * DURATION)  # 66,150 samples

# Feature Extraction Parameters
N_MFCC = 40               # Number of Mel-Frequency Cepstral Coefficients
N_FFT = 2048              # FFT window size
HOP_LENGTH = 512          # Number of samples between successive frames

# Expected 2D MFCC Feature Shape: (time_steps, n_mfcc)
# Time steps = 1 + (MAX_AUDIO_SAMPLES / HOP_LENGTH) approx 130
TIME_STEPS = 1 + (MAX_AUDIO_SAMPLES // HOP_LENGTH)

# ==========================================
# Training Hyperparameters (CPU Optimized)
# ==========================================
BATCH_SIZE = 32
EPOCHS = 45
LEARNING_RATE = 0.001
RANDOM_STATE = 42
TEST_SIZE = 0.15          # 15% Test set
VAL_SIZE = 0.15           # 15% Validation set (from train pool)
EARLY_STOPPING_PATIENCE = 10
REDUCE_LR_PATIENCE = 5
REDUCE_LR_FACTOR = 0.5
MIN_LR = 1e-5

# ==========================================
# Emotion Label Mappings
# ==========================================
# RAVDESS Filename format:
# Modality (01) - Channel (01) - Emotion (01-08) - Intensity (01-02) - Statement (01-02) - Repetition (01-02) - Actor (01-24)
RAVDESS_EMOTIONS = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fear",
    "07": "disgust",
    "08": "surprise",
}

# TESS Emotion keywords in filenames
TESS_EMOTIONS = {
    "angry": "angry",
    "disgust": "disgust",
    "fear": "fear",
    "happy": "happy",
    "neutral": "neutral",
    "ps": "surprise",  # pleasant surprise
    "sad": "sad",
}

# EMO-DB Emotion code letter in filename (6th character)
EMODB_EMOTIONS = {
    "W": "angry",
    "L": "boredom",
    "E": "disgust",
    "A": "fear",
    "F": "happy",
    "T": "sad",
    "N": "neutral",
}

# Emotion visual styling for Streamlit and plots
EMOTION_EMOJIS = {
    "happy": "😊",
    "sad": "😢",
    "angry": "😡",
    "neutral": "😐",
    "calm": "😌",
    "fear": "😨",
    "disgust": "🤢",
    "surprise": "😲",
}

EMOTION_COLORS = {
    "happy": "#FFD700",
    "sad": "#4682B4",
    "angry": "#FF4500",
    "neutral": "#A9A9A9",
    "calm": "#87CEEB",
    "fear": "#9370DB",
    "disgust": "#32CD32",
    "surprise": "#FF69B4",
}
