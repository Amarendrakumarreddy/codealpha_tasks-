"""
Training Pipeline for Speech Emotion Recognition (SER).
Implements:
- Stratified Train / Validation / Test data splitting
- Label encoding and artifact preservation
- CNN + Bidirectional LSTM deep learning architecture
- Callbacks: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
- Training history plotting (Loss & Accuracy)
- Reproducible random seeds
"""

import sys
import os
import json
import joblib
import random
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Set random seeds for reproducibility
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)

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

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks

# Set TensorFlow seed
tf.random.set_seed(SEED)


def build_ser_model(input_shape: tuple, num_classes: int) -> keras.Model:
    """
    Builds a robust, CPU-efficient CNN + Bi-LSTM Deep Learning Model
    for sequential MFCC speech emotion classification.
    """
    model = keras.Sequential(
        [
            layers.Input(shape=input_shape, name="mfcc_input"),
            # Convolutional Block 1
            layers.Conv1D(128, kernel_size=5, strides=1, padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling1D(pool_size=2),
            layers.Dropout(0.3),
            # Convolutional Block 2
            layers.Conv1D(128, kernel_size=5, strides=1, padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling1D(pool_size=2),
            layers.Dropout(0.3),
            # Convolutional Block 3
            layers.Conv1D(256, kernel_size=3, strides=1, padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling1D(pool_size=2),
            layers.Dropout(0.3),
            # Temporal Sequence Modeling
            layers.Bidirectional(layers.LSTM(64, return_sequences=False)),
            layers.Dropout(0.3),
            # Dense Classification Head
            layers.Dense(128, activation="relu"),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            layers.Dense(num_classes, activation="softmax", name="emotion_output"),
        ],
        name="Speech_Emotion_Classifier",
    )

    optimizer = keras.optimizers.Adam(learning_rate=config.LEARNING_RATE)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def plot_training_curves(history, output_path: Path):
    """Plots and saves training and validation Loss and Accuracy graphs."""
    epochs_ran = range(1, len(history.history["loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#fafafa")

    # Accuracy Plot
    axes[0].plot(epochs_ran, history.history["accuracy"], "b-o", label="Training Accuracy", linewidth=2)
    axes[0].plot(epochs_ran, history.history["val_accuracy"], "g-s", label="Validation Accuracy", linewidth=2)
    axes[0].set_title("Model Accuracy Across Epochs", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Epoch", fontsize=11)
    axes[0].set_ylabel("Accuracy", fontsize=11)
    axes[0].grid(True, linestyle="--", alpha=0.6)
    axes[0].legend(loc="lower right")

    # Loss Plot
    axes[1].plot(epochs_ran, history.history["loss"], "r-o", label="Training Loss", linewidth=2)
    axes[1].plot(epochs_ran, history.history["val_loss"], "m-s", label="Validation Loss", linewidth=2)
    axes[1].set_title("Model Loss Across Epochs", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Epoch", fontsize=11)
    axes[1].set_ylabel("Loss", fontsize=11)
    axes[1].grid(True, linestyle="--", alpha=0.6)
    axes[1].legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved training history curves to: {output_path}")


def train_model():
    """Executes the complete training workflow."""
    print("=" * 60)
    print("  STEP 3: MODEL TRAINING (CPU OPTIMIZED)")
    print("=" * 60)

    # 1. Load Features and Labels
    if not config.FEATURES_X_PATH.exists() or not config.LABELS_Y_PATH.exists():
        raise FileNotFoundError(
            "Features or labels not found in data/processed/. Please run extract_features.py first."
        )

    X = np.load(config.FEATURES_X_PATH)
    y_raw = np.load(config.LABELS_Y_PATH)
    print(f"Loaded feature matrix: {X.shape}, labels: {y_raw.shape}")

    # 2. Encode Labels
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y_raw)
    classes = list(encoder.classes_)
    num_classes = len(classes)
    print(f"Detected {num_classes} classes: {classes}")

    # Save Label Encoder
    joblib.dump(encoder, config.LABEL_ENCODER_PATH)
    print(f"Saved label encoder to: {config.LABEL_ENCODER_PATH}")

    # 3. Stratified Train / Test Split (Held-out test set for unbiased evaluation)
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y_encoded,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y_encoded,
    )

    # Stratified Train / Validation Split
    val_ratio = config.VAL_SIZE / (1.0 - config.TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=val_ratio,
        random_state=config.RANDOM_STATE,
        stratify=y_train_val,
    )

    # Save test set partitions for evaluate.py
    np.save(config.PROCESSED_DATA_DIR / "X_test.npy", X_test)
    np.save(config.PROCESSED_DATA_DIR / "y_test.npy", y_test)
    print(f"Partitioned Data -> Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    # 4. Build Model
    input_shape = (X.shape[1], X.shape[2])  # (time_steps, n_mfcc)
    print(f"Input shape: {input_shape} | Output classes: {num_classes}")
    model = build_ser_model(input_shape, num_classes)
    model.summary()

    # 5. Callbacks
    cb_early = callbacks.EarlyStopping(
        monitor="val_loss",
        patience=config.EARLY_STOPPING_PATIENCE,
        restore_best_weights=True,
        verbose=1,
    )
    cb_checkpoint = callbacks.ModelCheckpoint(
        filepath=str(config.MODEL_PATH),
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1,
    )
    cb_reduce_lr = callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=config.REDUCE_LR_FACTOR,
        patience=config.REDUCE_LR_PATIENCE,
        min_lr=config.MIN_LR,
        verbose=1,
    )

    # 6. Fit Model
    print(f"\nStarting training for {config.EPOCHS} epochs with batch size {config.BATCH_SIZE}...")
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        callbacks=[cb_early, cb_checkpoint, cb_reduce_lr],
        verbose=1,
    )

    # Ensure best model is saved
    model.save(config.MODEL_PATH)
    print(f"Final best model saved to: {config.MODEL_PATH}")

    # 7. Plot and Save Training Curves
    plot_training_curves(history, config.TRAINING_HISTORY_PATH)
    print("=" * 60)
    return model, history


if __name__ == "__main__":
    train_model()
