"""
Evaluation Script for Speech Emotion Recognition.
Evaluates the trained model on the held-out test dataset.
Generates:
- Test Accuracy, Precision, Recall, and F1-Scores (Macro & Weighted)
- Normalized & Count Confusion Matrix (results/confusion_matrix.png)
- Detailed Scikit-Learn Classification Report (results/classification_report.txt)
"""

import sys
import os
import joblib
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

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


def plot_confusion_matrix(y_true, y_pred, class_names, output_path: Path):
    """
    Plots a high-quality dual confusion matrix (counts + normalized percentages).
    """
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype("float") / (cm.sum(axis=1)[:, np.newaxis] + 1e-8)

    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor("#ffffff")

    # Annotated labels with count and percentage
    annot_matrix = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot_matrix[i, j] = f"{cm[i, j]}\n({cm_norm[i, j]*100:.1f}%)"

    sns.heatmap(
        cm_norm,
        annot=annot_matrix,
        fmt="",
        cmap="Blues",
        xticklabels=[c.capitalize() for c in class_names],
        yticklabels=[c.capitalize() for c in class_names],
        cbar=True,
        linewidths=1.0,
        linecolor="#f0f0f0",
        ax=ax,
    )

    ax.set_title("Speech Emotion Recognition - Confusion Matrix", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Predicted Emotion", fontsize=12, labelpad=10)
    ax.set_ylabel("True Emotion", fontsize=12, labelpad=10)
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.yticks(rotation=0, fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved confusion matrix heatmap to: {output_path}")


def evaluate_model():
    """Runs evaluation on the test partition and writes reports."""
    print("=" * 60)
    print("  STEP 4: MODEL EVALUATION ON HELD-OUT TEST DATA")
    print("=" * 60)

    # 1. Check artifact existence
    test_x_path = config.PROCESSED_DATA_DIR / "X_test.npy"
    test_y_path = config.PROCESSED_DATA_DIR / "y_test.npy"

    if not test_x_path.exists() or not test_y_path.exists():
        raise FileNotFoundError(
            "Test partition (X_test.npy / y_test.npy) not found. Please run train.py first."
        )

    if not config.MODEL_PATH.exists():
        raise FileNotFoundError(f"Model checkpoint not found at: {config.MODEL_PATH}")

    if not config.LABEL_ENCODER_PATH.exists():
        raise FileNotFoundError(f"Label encoder not found at: {config.LABEL_ENCODER_PATH}")

    # 2. Load Data and Artifacts
    X_test = np.load(test_x_path)
    y_test = np.load(test_y_path)
    encoder = joblib.load(config.LABEL_ENCODER_PATH)
    class_names = list(encoder.classes_)
    model = keras.models.load_model(config.MODEL_PATH)

    print(f"Loaded test dataset: {len(X_test)} samples across {len(class_names)} classes.")
    print(f"Target classes: {class_names}")

    # 3. Model Inference
    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # 4. Calculate Metrics
    accuracy = accuracy_score(y_test, y_pred)
    macro_precision = precision_score(y_test, y_pred, average="macro", zero_division=0)
    weighted_precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    macro_recall = recall_score(y_test, y_pred, average="macro", zero_division=0)
    weighted_recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    report_text = classification_report(
        y_test,
        y_pred,
        target_names=[c.capitalize() for c in class_names],
        digits=4,
        zero_division=0,
    )

    # 5. Format & Save Classification Report
    summary_report = (
        "========================================================================\n"
        "           SPEECH EMOTION RECOGNITION - EVALUATION REPORT\n"
        "========================================================================\n\n"
        f"Evaluated Test Samples: {len(X_test)}\n"
        f"Number of Classes:      {len(class_names)}\n"
        f"Class Labels:           {', '.join(class_names)}\n\n"
        "------------------------------------------------------------------------\n"
        "OVERALL PERFORMANCE METRICS:\n"
        "------------------------------------------------------------------------\n"
        f"Test Accuracy:          {accuracy * 100:.2f}%\n"
        f"Macro Precision:        {macro_precision * 100:.2f}%\n"
        f"Weighted Precision:     {weighted_precision * 100:.2f}%\n"
        f"Macro Recall:           {macro_recall * 100:.2f}%\n"
        f"Weighted Recall:        {weighted_recall * 100:.2f}%\n"
        f"Macro F1-Score:         {macro_f1 * 100:.2f}%\n"
        f"Weighted F1-Score:      {weighted_f1 * 100:.2f}%\n\n"
        "------------------------------------------------------------------------\n"
        "DETAILED PER-CLASS CLASSIFICATION REPORT:\n"
        "------------------------------------------------------------------------\n"
        f"{report_text}\n"
        "========================================================================\n"
    )

    with open(config.CLASSIFICATION_REPORT_PATH, "w") as f:
        f.write(summary_report)

    print(summary_report)
    print(f"Saved full evaluation report to: {config.CLASSIFICATION_REPORT_PATH}")

    # 6. Plot Confusion Matrix
    plot_confusion_matrix(y_test, y_pred, class_names, config.CONFUSION_MATRIX_PATH)
    print("=" * 60)

    return {
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "weighted_precision": weighted_precision,
        "macro_recall": macro_recall,
        "weighted_recall": weighted_recall,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
    }


if __name__ == "__main__":
    evaluate_model()
