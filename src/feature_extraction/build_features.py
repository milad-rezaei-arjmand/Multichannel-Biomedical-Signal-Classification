"""
Feature-building pipeline for multichannel biomedical signals.

Combines:
- Time-domain features
- Frequency-domain features
- Wavelet features
- Global/cross-channel features
"""

from __future__ import annotations

import numpy as np

from src.config import DEFAULT_FS

from .frequency_features import extract_multichannel_frequency_features
from .global_features import extract_global_features
from .time_features import extract_multichannel_time_features
from .wavelet_features import extract_multichannel_wavelet_features


def build_feature_vector(
    X,
    fs=DEFAULT_FS,
    wavelet="db4",
    level=4,
):
    """Build a deterministic feature vector for one multichannel signal."""

    X = np.asarray(X, dtype=np.float32)

    if X.ndim != 2:
        raise ValueError(
            "Input signal must have shape (time_points, channels)."
        )

    if X.shape[0] == 0 or X.shape[1] == 0:
        raise ValueError("Input signal cannot be empty.")

    features = {}

    features.update(
        extract_multichannel_time_features(X)
    )
    features.update(
        extract_multichannel_frequency_features(X, fs)
    )
    features.update(
        extract_multichannel_wavelet_features(
            X,
            wavelet=wavelet,
            level=level,
        )
    )
    features.update(
        extract_global_features(X)
    )

    feature_names = sorted(features.keys())

    feature_vector = np.asarray(
        [features[name] for name in feature_names],
        dtype=np.float32,
    )

    if not np.all(np.isfinite(feature_vector)):
        raise ValueError(
            "Extracted feature vector contains NaN or infinite values."
        )

    return feature_vector, feature_names


def build_dataset_features(
    signals,
    fs=DEFAULT_FS,
):
    """Build a feature matrix from a 3-D signal dataset."""

    signals = np.asarray(signals, dtype=np.float32)

    if signals.ndim != 3:
        raise ValueError(
            "Dataset must have shape (samples, time_points, channels)."
        )

    if len(signals) == 0:
        raise ValueError("Dataset is empty.")

    all_features = []
    reference_feature_names = None

    for index, signal_sample in enumerate(signals):
        vector, names = build_feature_vector(
            signal_sample,
            fs=fs,
        )

        if reference_feature_names is None:
            reference_feature_names = names
        elif names != reference_feature_names:
            raise RuntimeError(
                "Feature schema changed between samples at "
                f"sample index {index}."
            )

        all_features.append(vector)

    X_features = np.vstack(all_features).astype(np.float32)

    return X_features, reference_feature_names
