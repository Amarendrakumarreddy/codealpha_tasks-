"""
Dataset Preparation & Ingestion Script.
Discovers audio files from local or external folders, parses emotion labels
from RAVDESS, TESS, or EMO-DB datasets, validates files, copies them into data/raw/,
and produces a dataset metadata CSV file.
"""

import os
import sys
import shutil
import re
import argparse
from pathlib import Path
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

import config


def parse_ravdess_filename(filename: str):
    """
    Parses RAVDESS audio filename metadata.
    Format: Modality-Channel-Emotion-Intensity-Statement-Repetition-Actor.wav
    Example: 03-01-03-01-01-01-01.wav -> Emotion 03 (Happy), Actor 01 (Male)
    """
    stem = Path(filename).stem
    parts = stem.split("-")
    if len(parts) == 7:
        emotion_code = parts[2]
        emotion = config.RAVDESS_EMOTIONS.get(emotion_code, "unknown")
        intensity = "normal" if parts[3] == "01" else "strong"
        actor_id = int(parts[6])
        gender = "female" if actor_id % 2 == 0 else "male"
        return {
            "dataset": "RAVDESS",
            "emotion": emotion,
            "intensity": intensity,
            "actor_id": actor_id,
            "gender": gender,
        }
    return None


def parse_tess_filename(filename: str):
    """
    Parses TESS audio filename metadata.
    Example: OAF_back_angry.wav -> Emotion angry, Female
    """
    stem = Path(filename).stem.lower()
    for key, emotion in config.TESS_EMOTIONS.items():
        if key in stem:
            gender = "female"  # TESS is recorded by two female speakers
            actor_id = 1 if "oaf" in stem else 2
            return {
                "dataset": "TESS",
                "emotion": emotion,
                "intensity": "normal",
                "actor_id": actor_id,
                "gender": gender,
            }
    return None


def parse_emodb_filename(filename: str):
    """
    Parses EMO-DB audio filename metadata.
    Example: 03a01Fa.wav -> 6th char 'F' is Freude (Happy)
    """
    stem = Path(filename).stem
    if len(stem) >= 6:
        code = stem[5].upper()
        if code in config.EMODB_EMOTIONS:
            emotion = config.EMODB_EMOTIONS[code]
            actor_id = stem[:2]
            return {
                "dataset": "EMO-DB",
                "emotion": emotion,
                "intensity": "normal",
                "actor_id": actor_id,
                "gender": "unknown",
            }
    return None


def identify_audio_file(filename: str):
    """Identifies dataset type and extracts metadata from filename."""
    meta = parse_ravdess_filename(filename)
    if meta:
        return meta
    meta = parse_tess_filename(filename)
    if meta:
        return meta
    meta = parse_emodb_filename(filename)
    if meta:
        return meta
    return None


# Set UTF-8 encoding for Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


def discover_dataset(custom_source_dir=None):
    """
    Searches for dataset files in data/raw/, custom_source_dir,
    or configured external search locations.
    """
    # 1. Check data/raw first
    raw_files = list(config.RAW_DATA_DIR.rglob("*.wav")) + list(config.RAW_DATA_DIR.rglob("*.mp3"))
    if len(raw_files) >= 50:
        print(f"[Dataset Discovery] Found {len(raw_files)} audio files directly in {config.RAW_DATA_DIR}")
        return config.RAW_DATA_DIR, raw_files, False

    # 2. Check custom source if provided
    candidates = []
    if custom_source_dir:
        candidates.append(Path(custom_source_dir))

    # 3. Add external search paths
    candidates.extend(config.EXTERNAL_SEARCH_PATHS)

    for candidate in candidates:
        if candidate.exists() and candidate.is_dir():
            # If candidate contains audio_speech_actors_01-24, use only that subfolder to avoid duplicates
            preferred_sub = candidate / "audio_speech_actors_01-24"
            target_search_dir = preferred_sub if preferred_sub.exists() else candidate

            files = list(target_search_dir.rglob("*.wav")) + list(target_search_dir.rglob("*.mp3"))
            if len(files) >= 50:
                print(f"[Dataset Discovery] Found {len(files)} audio files in: {target_search_dir}")
                return target_search_dir, files, True

    return None, [], False


def prepare_dataset(source_dir=None, copy_to_raw=True):
    """
    Prepares dataset metadata, ingests audio files into data/raw/,
    and generates data/processed/dataset_metadata.csv.
    """
    print("=" * 60)
    print("  STEP 1: DATASET INGESTION & METADATA PREPARATION")
    print("=" * 60)

    found_dir, files, is_external = discover_dataset(source_dir)

    if not found_dir or len(files) == 0:
        error_msg = (
            "\n[ERROR] No speech emotion dataset found!\n"
            f"Please place your dataset (RAVDESS, TESS, or EMO-DB audio files) into:\n"
            f"  -> {config.RAW_DATA_DIR}\n\n"
            f"Searched external paths:\n"
            + "\n".join(f"  - {p}" for p in config.EXTERNAL_SEARCH_PATHS)
            + "\n\nThen rerun: python scripts/prepare_dataset.py"
        )
        print(error_msg, file=sys.stderr)
        raise FileNotFoundError("Speech emotion dataset not found.")

    print(f"Total audio files found: {len(files)}")

    metadata_records = []
    seen_filenames = set()
    skipped_files = 0

    target_raw_base = config.RAW_DATA_DIR

    for idx, file_path in enumerate(files):
        filename = file_path.name
        if filename in seen_filenames:
            continue
        seen_filenames.add(filename)

        meta = identify_audio_file(filename)

        if not meta or meta["emotion"] == "unknown":
            skipped_files += 1
            continue

        # If external and copy requested, organize into data/raw/
        if is_external and copy_to_raw:
            # Preserve parent folder name (e.g. Actor_01)
            parent_folder_name = file_path.parent.name
            dest_dir = target_raw_base / meta["dataset"].lower() / parent_folder_name
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest_file = dest_dir / filename
            if not dest_file.exists():
                shutil.copy2(file_path, dest_file)
            saved_rel_path = dest_file.relative_to(config.PROJECT_ROOT)
        else:
            try:
                saved_rel_path = file_path.relative_to(config.PROJECT_ROOT)
            except ValueError:
                saved_rel_path = file_path

        record = {
            "filename": filename,
            "relative_path": str(saved_rel_path).replace("\\", "/"),
            "dataset": meta["dataset"],
            "emotion": meta["emotion"],
            "intensity": meta["intensity"],
            "actor_id": meta["actor_id"],
            "gender": meta["gender"],
        }
        metadata_records.append(record)

    df = pd.DataFrame(metadata_records)

    # Save metadata CSV
    df.to_csv(config.DATASET_METADATA_PATH, index=False)
    print(f"\nSuccessfully prepared metadata for {len(df)} samples.")
    if skipped_files > 0:
        print(f"Skipped {skipped_files} files that did not match known emotion conventions.")

    print(f"Metadata saved to: {config.DATASET_METADATA_PATH}")
    print("\nDataset Breakdown by Emotion:")
    print("-" * 35)
    counts = df["emotion"].value_counts()
    for emotion, count in counts.items():
        try:
            print(f"  * {emotion.capitalize():<12}: {count:>4} samples ({count/len(df)*100:>5.1f}%)")
        except Exception:
            print(f"  * {emotion.capitalize()}: {count} samples")

    print("-" * 35)
    print(f"Total Emotions Detected: {len(counts)}")
    print(f"Total Audio Samples:     {len(df)}")
    print("=" * 60)
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare Speech Emotion Recognition Dataset")
    parser.add_argument("--source", type=str, default=None, help="Custom path to raw audio folder")
    parser.add_argument("--no-copy", action="store_true", help="Do not copy external files into data/raw/")
    args = parser.parse_args()

    prepare_dataset(source_dir=args.source, copy_to_raw=not args.no_copy)
