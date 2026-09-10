# TASK 2: AI Speech Emotion Recognition (SER)

An end-to-end deep learning system that recognizes human emotions from speech audio signals using **Mel-Frequency Cepstral Coefficients (MFCC)**, a **CPU-optimized CNN + Bidirectional LSTM** deep neural network, comprehensive evaluation metrics, and an interactive **Streamlit web application**.

---

## Table of Contents
- [Objective](#objective)
- [System Architecture](#system-architecture)
- [Dataset Information & Discovery](#dataset-information--discovery)
- [Project Structure](#project-structure)
- [Technologies Used](#technologies-used)
- [Installation Guide](#installation-guide)
- [Data Pipeline & Training Instructions](#data-pipeline--training-instructions)
  - [1. Dataset Ingestion & Preparation](#1-dataset-ingestion--preparation)
  - [2. Feature Extraction](#2-feature-extraction)
  - [3. Model Training](#3-model-training)
  - [4. Model Evaluation](#4-model-evaluation)
  - [5. CLI Prediction](#5-cli-prediction)
- [Streamlit Web Application](#streamlit-web-application)
- [Evaluation Results & Performance](#evaluation-results--performance)
- [Limitations](#limitations)
- [Future Improvements](#future-improvements)

---

## Objective
Human speech conveys emotional states through acoustic variations in pitch, energy, spectral content, and temporal cadence. The objective of this project is to create an automated, reliable AI system that classifies human voice recordings into discrete emotional states:
- **Happy** 😊
- **Sad** 😢
- **Angry** 😡
- **Neutral** 😐
- **Calm** 😌
- **Fear** 😨
- **Disgust** 🤢
- **Surprise** 😲

The pipeline automatically detects and supports the target emotion classes present in the dataset.

---

## System Architecture

```
Raw Audio Input (WAV / MP3)
      │
      ▼
Audio Preprocessing
├── Mono Conversion
├── Resampling to 22,050 Hz
├── Peak Amplitude Normalization
└── Fixed-Length (3.0s = 66,150 samples) with Center-Crop / Zero-Padding
      │
      ▼
Feature Extraction (Librosa)
├── 40 Mel-Frequency Cepstral Coefficients (MFCCs)
├── Window FFT: 2048, Hop Length: 512
└── Utterance Cepstral Mean & Variance Normalization (CMVN)
      │  Feature Matrix Shape: (130 time steps, 40 MFCC bands)
      ▼
Deep Learning Architecture (TensorFlow / Keras)
├── Conv1D (128 filters, kernel=5) + BatchNorm + ReLU + MaxPool(2) + Dropout(0.3)
├── Conv1D (128 filters, kernel=5) + BatchNorm + ReLU + MaxPool(2) + Dropout(0.3)
├── Conv1D (256 filters, kernel=3) + BatchNorm + ReLU + MaxPool(2) + Dropout(0.3)
├── Bidirectional LSTM (64 units) + Dropout(0.3)
├── Dense (128 units, ReLU) + BatchNorm + Dropout(0.4)
└── Dense (num_classes, Softmax)
      │
      ▼
Emotion Prediction Output
├── Predicted Emotion Class (e.g., HAPPY)
├── Prediction Confidence Score (e.g., 89.2%)
└── Full Probability Distribution across all classes
      │
      ▼
Interactive Streamlit Web Dashboard
```

---

## Dataset Information & Discovery

The primary dataset used is the **Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS)**:
- **Total Speech Files**: 1,440 WAV audio recordings (48 kHz, 16-bit uncompressed).
- **Actors**: 24 professional actors (12 male, 12 female).
- **Statements**: Two standardized neutral statements spoken with varied emotional intonations.
- **Emotion Categories**: 8 discrete categories (Neutral, Calm, Happy, Sad, Angry, Fearful, Disgust, Surprised).
- **Intensities**: Normal and strong emotional expressions.

### Automatic Dataset Discovery:
The ingestion script `scripts/prepare_dataset.py` automatically checks:
1. `data/raw/` (internal project directory)
2. `c:\Users\DELL\Downloads\archive (4)` (external downloaded archive)
3. Any custom directory provided via `--source <path>`

If the dataset is not yet downloaded, download the RAVDESS Speech dataset from Kaggle or Zenodo and place the `Actor_01` ... `Actor_24` folders inside `data/raw/`.

---

## Project Structure

```
speech recog/
│
├── data/
│   ├── raw/                              # Ingested raw WAV audio files (RAVDESS/TESS/EMO-DB)
│   └── processed/
│       ├── dataset_metadata.csv          # Catalog of samples, labels, actors, and gender
│       ├── X_features.npy                # Extracted 40 MFCC feature tensors (1440, 130, 40)
│       ├── y_labels.npy                  # Encoded string emotion labels
│       ├── X_test.npy                    # Unbiased held-out test feature partition
│       └── y_test.npy                    # Held-out test ground-truth labels
│
├── models/
│   ├── emotion_model.keras               # Best trained deep learning model checkpoint
│   ├── label_encoder.pkl                 # Scikit-Learn LabelEncoder instance
│   └── preprocessor_config.json          # Preprocessing and class configuration parameters
│
├── notebooks/
│   └── emotion_recognition_exploration.ipynb # Interactive EDA and acoustic feature visualizer
│
├── scripts/
│   ├── prepare_dataset.py                # Discovers, copies, and indexes dataset audio files
│   ├── extract_features.py               # Preprocesses audio and computes 40 MFCCs
│   ├── train.py                          # Trains CNN + Bi-LSTM model on CPU with callbacks
│   ├── evaluate.py                       # Evaluates model on test data and plots confusion matrix
│   └── predict.py                        # CLI inference tool for single audio files
│
├── results/
│   ├── confusion_matrix.png              # Heatmap with counts and normalized percentages
│   ├── training_history.png              # Loss and accuracy curves across training epochs
│   └── classification_report.txt         # Full precision, recall, and F1 metrics
│
├── app.py                                # Modern interactive Streamlit web application
├── requirements.txt                      # Complete list of Python package dependencies
├── README.md                             # Comprehensive project documentation
└── config.py                             # Centralized paths, parameters, and hyperparameters
```

---

## Technologies Used

- **Python 3.11**
- **Librosa (0.11.0)**: Acoustic signal loading, resampling, and MFCC computation.
- **SoundFile (0.14.0)**: High-speed audio I/O.
- **TensorFlow (2.21.0) / Keras (3.15.1)**: CNN + Bi-LSTM deep learning model architecture and training.
- **Scikit-Learn (1.9.0)**: Stratified dataset splitting, label encoding, and classification metrics.
- **Streamlit (1.63.0)**: Web application frontend with live audio playback, microphone recording, and interactive graphs.
- **Matplotlib & Seaborn**: Signal waveform, spectrogram, loss curves, and confusion matrix visualizations.
- **NumPy & Pandas**: Numerical feature representation and dataset tabular processing.

---

## Installation Guide

Clone or navigate to the project directory and install the required dependencies:

```bash
cd "speech recog"
pip install -r requirements.txt
```

---

## Data Pipeline & Training Instructions

Run the pipeline sequentially from the project root directory:

### 1. Dataset Ingestion & Preparation
Discovers dataset files, parses filename emotion codes, copies audio to `data/raw/`, and generates `data/processed/dataset_metadata.csv`:
```bash
python scripts/prepare_dataset.py
```

### 2. Feature Extraction
Normalizes audio clips, enforces a 3.0-second length, extracts 40-band MFCCs with CMVN, and saves `X_features.npy` and `y_labels.npy`:
```bash
python scripts/extract_features.py
```

### 3. Model Training
Splits data into stratified Train (70%), Validation (15%), and Test (15%) partitions. Trains the CNN + Bi-LSTM model on CPU with Early Stopping and learning-rate reduction:
```bash
python scripts/train.py
```
Outputs:
- Best checkpoint: `models/emotion_model.keras`
- Label encoder: `models/label_encoder.pkl`
- Training graphs: `results/training_history.png`

### 4. Model Evaluation
Evaluates the model on the held-out test set (216 samples):
```bash
python scripts/evaluate.py
```
Outputs:
- Evaluation report: `results/classification_report.txt`
- Confusion matrix heatmap: `results/confusion_matrix.png`

### 5. CLI Prediction
Predict emotion on any standalone WAV or MP3 audio file:
```bash
python scripts/predict.py --file "data/raw/ravdess/Actor_01/03-01-03-01-01-01-01.wav"
```

Example Output:
```
========================================
Emotion:    HAPPY
Confidence: 89.2%
========================================
Probabilities:
  Happy     :  89.2%  █████████████████
  Neutral   :   4.0%  
  Angry     :   3.4%  
  Fear      :   1.8%  
  Surprise  :   0.9%  
  Calm      :   0.4%  
  Sad       :   0.3%  
  Disgust   :   0.1%  
========================================
```

---

## Streamlit Web Application

Launch the interactive web application:
```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`.

### Features of the Web Interface:
1. **Audio Input**:
   - Upload any custom WAV, MP3, OGG, or FLAC voice recording.
   - Select from pre-loaded dataset sample clips for instant testing.
   - Live microphone voice recording via `st.audio_input`.
2. **Audio Player**: Listen to the speech clip directly in the browser.
3. **Signal Visualizer**: Inspect the audio waveform and 2D Mel-Spectrogram representation.
4. **Emotion Prediction Banner**: Highlights detected emotion with color-coded badge and confidence percentage.
5. **Probability Distribution**: Interactive bar chart displaying likelihood across all 8 emotions.
6. **Model Info Tab**: View model architecture parameters and evaluation confusion matrix.

---

## Evaluation Results & Performance

Evaluated on **216 held-out test samples** across 8 emotion classes:

| Metric | Score |
| :--- | :--- |
| **Test Accuracy** | **59.72%** (Baseline chance: 12.50%) |
| **Macro Precision** | **61.28%** |
| **Weighted Precision**| **61.98%** |
| **Macro Recall** | **60.53%** |
| **Weighted Recall** | **59.72%** |
| **Macro F1-Score** | **58.71%** |
| **Weighted F1-Score** | **58.62%** |

### Per-Class Performance Summary:
- **Angry**: 80.0% Precision, 71.4% Recall, 75.5% F1-Score
- **Surprise**: 76.0% Precision, 65.5% Recall, 70.4% F1-Score
- **Fear**: 56.1% Precision, 79.3% Recall, 65.7% F1-Score
- **Calm**: 51.0% Precision, 86.2% Recall, 64.1% F1-Score
- **Neutral**: 50.0% Precision, 71.4% Recall, 58.8% F1-Score
- **Disgust**: 80.0% Precision, 41.4% Recall, 54.5% F1-Score
- **Happy**: 57.1% Precision, 41.4% Recall, 48.0% F1-Score
- **Sad**: 40.0% Precision, 27.6% Recall, 32.7% F1-Score

---

## Limitations

1. **Acoustic Overlap**: Emotions with similar acoustic dynamics (such as sadness and neutrality, or high-arousal happiness and anger) can exhibit spectral overlap in MFCC space.
2. **Actor Diversity**: RAVDESS recordings are performed by North American English speakers; linguistic accents and cultural speech patterns may affect confidence on cross-cultural speech.
3. **Background Noise**: Model is trained on clean studio recordings; noisy real-world recordings benefit from pre-denoising filters.

---

## Future Improvements

1. **Acoustic Data Augmentation**: Introduce pitch shifting, time stretching, background noise injection, and SpecAugment to enhance robustness.
2. **Multi-Feature Fusion**: Combine MFCCs with Chroma, Spectral Contrast, Zero Crossing Rate, and fundamental frequency (F0 / pitch contour).
3. **Pretrained Speech Transformers**: Fine-tune self-supervised speech foundation models such as Wav2Vec 2.0 or HuBERT for state-of-the-art SER accuracy.
4. **Real-Time Streaming**: Stream speech chunks continuously via WebSockets for live conversation sentiment monitoring.
