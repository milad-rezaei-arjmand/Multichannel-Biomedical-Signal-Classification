"""
Signal preprocessing utilities.

Includes:
- NaN/Inf handling
- Butterworth band-pass filtering
- Per-channel Z-score normalization
- Multichannel and dataset-level preprocessing
"""

from __future__ import annotations

import numpy as np
from scipy import signal

from src.config import (
    DEFAULT_FS,
    FILTER_ORDER,
    HIGH_FREQ,
    LOW_FREQ,
)


def clean_signal(x):
    """Replace NaN and infinite values with finite zeros."""

    x = np.asarray(x, dtype=np.float32)

    return np.nan_to_num(
        x,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype(np.float32)


def butter_bandpass(
    x,
    fs=DEFAULT_FS,
    lo=LOW_FREQ,
    hi=HIGH_FREQ,
    order=FILTER_ORDER,
):
    """Apply a Butterworth band-pass filter."""

    x = clean_signal(x)

    fs = float(fs)
    lo = float(lo)
    hi = float(hi)

    if fs <= 0:
        raise ValueError("Sampling frequency must be positive.")

    nyquist = fs / 2.0

    if not (0.0 < lo < hi < nyquist):
        raise ValueError(
            "Band-pass cutoffs must satisfy "
            f"0 < low < high < Nyquist. "
            f"Received low={lo}, high={hi}, fs={fs}."
        )

    sos = signal.butter(
        int(order),
        [lo, hi],
        btype="bandpass",
        fs=fs,
        output="sos",
    )

    try:
        filtered = signal.sosfiltfilt(sos, x)
    except ValueError:
        # Very short signals may not satisfy zero-phase padding requirements.
        filtered = signal.sosfilt(sos, x)

    return np.asarray(filtered, dtype=np.float32)


def normalize_signal(x):
    """Apply per-channel Z-score normalization."""

    x = clean_signal(x)

    mean = float(np.mean(x))
    std = float(np.std(x))

    if std <= 1e-6:
        return np.zeros_like(x, dtype=np.float32)

    return ((x - mean) / std).astype(np.float32)


def preprocess_channel(
    x,
    fs=DEFAULT_FS,
    low_freq=LOW_FREQ,
    high_freq=HIGH_FREQ,
    filter_order=FILTER_ORDER,
):
    """Clean, band-pass filter, and normalize one channel."""

    x = clean_signal(x)

    x = butter_bandpass(
        x,
        fs=fs,
        lo=low_freq,
        hi=high_freq,
        order=filter_order,
    )

    return normalize_signal(x)


def preprocess_multichannel_signal(
    signals,
    fs=DEFAULT_FS,
    low_freq=LOW_FREQ,
    high_freq=HIGH_FREQ,
    filter_order=FILTER_ORDER,
):
    """
    Preprocess one signal with shape (time_points, channels).
    """

    signals = np.asarray(signals, dtype=np.float32)

    if signals.ndim != 2:
        raise ValueError(
            "Single signal must have shape (time_points, channels)."
        )

    processed_channels = [
        preprocess_channel(
            signals[:, channel_index],
            fs=fs,
            low_freq=low_freq,
            high_freq=high_freq,
            filter_order=filter_order,
        )
        for channel_index in range(signals.shape[1])
    ]

    return np.stack(processed_channels, axis=1)


def preprocess_dataset(
    signals,
    fs=DEFAULT_FS,
    low_freq=LOW_FREQ,
    high_freq=HIGH_FREQ,
    filter_order=FILTER_ORDER,
):
    """
    Preprocess a dataset with shape (samples, time_points, channels).
    """

    signals = np.asarray(signals, dtype=np.float32)

    if signals.ndim != 3:
        raise ValueError(
            "Dataset must have shape (samples, time_points, channels)."
        )

    processed = [
        preprocess_multichannel_signal(
            sample,
            fs=fs,
            low_freq=low_freq,
            high_freq=high_freq,
            filter_order=filter_order,
        )
        for sample in signals
    ]

    return np.asarray(processed, dtype=np.float32)
