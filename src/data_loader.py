"""
Dataset loading utilities for
Multichannel Biomedical Signal Classification.

Supported formats:
- NPZ (recommended)
- CSV (basic support)
"""

from pathlib import Path

import numpy as np
import pandas as pd



def load_npz_dataset(file_path):
    """
    Load biomedical signal dataset
    from NPZ format.

    Expected structure:

    signals:
        (samples, time_points, channels)

    labels:
        (samples,)
    """

    data = np.load(
        file_path,
        allow_pickle=True
    )


    if "signals" not in data or "labels" not in data:

        raise ValueError(
            "NPZ file must contain 'signals' and 'labels'."
        )


    signals = data["signals"]

    labels = data["labels"]


    signals = np.asarray(
        signals,
        dtype=np.float32
    )


    labels = np.asarray(
        labels
    )


    if signals.ndim != 3:

        raise ValueError(
            "Signals must have shape "
            "(samples, time_points, channels)."
        )


    if len(signals) != len(labels):

        raise ValueError(
            "Number of signals and labels must match."
        )


    return signals, labels



def load_csv_dataset(file_path):
    """
    Load dataset from CSV format.

    Expected columns:

    Amplitude
    Velocity
    Acceleration
    Alpha
    label
    """

    data = pd.read_csv(
        file_path
    )


    if "label" not in data.columns:

        raise ValueError(
            "CSV dataset must contain 'label' column."
        )


    labels = data["label"].values


    signals = data.drop(
        columns=["label"]
    ).values


    signals = signals.astype(
        np.float32
    )


    return signals, labels



def validate_dataset(
    signals,
    labels
):
    """
    Validate loaded dataset.
    """

    if len(signals) != len(labels):

        raise ValueError(
            "Signals and labels size mismatch."
        )


    print(
        "Signals shape:",
        signals.shape
    )

    print(
        "Labels shape:",
        labels.shape
    )



def load_dataset(
    data_path
):
    """
    General dataset loader.

    Supports:

    .npz
    .csv
    """

    path = Path(
        data_path
    )


    if not path.exists():

        raise FileNotFoundError(
            f"Dataset file not found: {path}"
        )


    if path.suffix.lower() == ".npz":

        signals, labels = load_npz_dataset(
            path
        )


    elif path.suffix.lower() == ".csv":

        signals, labels = load_csv_dataset(
            path
        )


    else:

        raise ValueError(
            "Unsupported dataset format. "
            "Use .npz or .csv"
        )


    validate_dataset(
        signals,
        labels
    )


    return signals, labels