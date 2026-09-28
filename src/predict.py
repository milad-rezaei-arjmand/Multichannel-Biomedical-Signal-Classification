"""
Inference pipeline for Multichannel Biomedical Signal Classification.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
from catboost import CatBoostClassifier


ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))


from src.config import DEFAULT_FS, N_CHANNELS
from src.feature_extraction.build_features import build_feature_vector
from src.models.catboost_classifier import predict_average_probability
from src.preprocessing.signal_processing import preprocess_multichannel_signal


def load_models(model_dir="outputs/checkpoints"):
    """Load all saved CatBoost ensemble members."""

    model_dir = Path(model_dir)
    model_paths = sorted(
        model_dir.glob("catboost_model_*.cbm")
    )

    if not model_paths:
        raise FileNotFoundError(
            f"No CatBoost model files found in: {model_dir}"
        )

    models = []

    for model_path in model_paths:
        model = CatBoostClassifier()
        model.load_model(model_path)
        models.append(model)

    reference_classes = np.asarray(
        models[0].classes_
    )

    for model_index, model in enumerate(
        models[1:],
        start=1,
    ):
        if not np.array_equal(
            np.asarray(model.classes_),
            reference_classes,
        ):
            raise ValueError(
                "Saved ensemble models use inconsistent class order "
                f"at model index {model_index}."
            )

    return models


def load_selector(model_dir="outputs/checkpoints"):
    """Load the fitted feature selector."""

    path = Path(model_dir) / "feature_selector.pkl"

    if not path.exists():
        raise FileNotFoundError(
            f"Feature selector not found: {path}"
        )

    with open(path, "rb") as handle:
        return pickle.load(handle)


def load_metadata(model_dir="outputs/checkpoints"):
    """Load training metadata when available."""

    path = Path(model_dir) / "training_metadata.json"

    if not path.exists():
        return {}

    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def predict(
    input_path,
    model_dir="outputs/checkpoints",
):
    """Predict one 4-channel signal stored as a NumPy .npy file."""

    signal = np.load(input_path)

    signal = np.asarray(
        signal,
        dtype=np.float32,
    )

    if signal.ndim != 2:
        raise ValueError(
            "Input .npy signal must have shape (time_points, channels)."
        )

    if signal.shape[1] != N_CHANNELS:
        raise ValueError(
            f"Expected {N_CHANNELS} channels; found {signal.shape[1]}."
        )

    metadata = load_metadata(
        model_dir,
    )

    sampling_frequency = int(
        metadata.get(
            "sampling_frequency_hz",
            DEFAULT_FS,
        )
    )

    signal = preprocess_multichannel_signal(
        signal,
        fs=sampling_frequency,
    )

    features, _ = build_feature_vector(
        signal,
        fs=sampling_frequency,
    )

    features = features.reshape(1, -1)

    selector = load_selector(
        model_dir,
    )

    features = selector.transform(
        features
    )

    models = load_models(
        model_dir,
    )

    probability = predict_average_probability(
        models,
        features,
    )

    prediction_index = int(
        np.argmax(
            probability,
            axis=1,
        )[0]
    )

    # Important: probability columns follow CatBoost's learned classes_ order.
    classes = np.asarray(
        models[0].classes_
    )

    prediction = str(
        classes[prediction_index]
    )

    confidence = float(
        probability[0, prediction_index]
    )

    return prediction, confidence


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Biomedical multichannel signal prediction."
    )

    parser.add_argument(
        "--input",
        required=True,
        help=(
            "Path to a .npy signal with shape "
            "(time_points, 4)."
        ),
    )

    parser.add_argument(
        "--model-dir",
        default="outputs/checkpoints",
        help="Directory containing trained models and selector.",
    )

    args = parser.parse_args()

    prediction, confidence = predict(
        args.input,
        model_dir=args.model_dir,
    )

    print("\nPrediction:", prediction)
    print(
        "Confidence:",
        round(confidence * 100.0, 2),
        "%",
    )
