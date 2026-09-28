"""
Dataset loading utilities for Multichannel Biomedical Signal Classification.

The public pipeline uses NPZ as its supported training-data format.

Required arrays
---------------
signals : (samples, time_points, 4)
labels  : (samples,)

Optional array
--------------
groups  : (samples,)
    Participant/recording group identifiers. When present, the training
    pipeline uses group-disjoint train/validation/test splitting.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.config import N_CHANNELS


def load_npz_dataset(file_path):
    """Load and validate a multichannel biomedical signal dataset."""

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    if path.suffix.lower() != ".npz":
        raise ValueError(
            "Unsupported dataset format. The public pipeline supports .npz only."
        )

    data = np.load(path, allow_pickle=True)

    if "signals" not in data or "labels" not in data:
        raise ValueError(
            "NPZ file must contain 'signals' and 'labels' arrays."
        )

    signals = np.asarray(data["signals"], dtype=np.float32)
    labels = np.asarray(data["labels"])

    if signals.ndim != 3:
        raise ValueError(
            "Signals must have shape (samples, time_points, channels)."
        )

    if signals.shape[2] != N_CHANNELS:
        raise ValueError(
            f"Expected exactly {N_CHANNELS} channels; "
            f"found {signals.shape[2]}."
        )

    if labels.ndim != 1:
        labels = labels.reshape(-1)

    if len(signals) != len(labels):
        raise ValueError(
            "Number of signal samples and labels must match."
        )

    if len(signals) == 0:
        raise ValueError("Dataset is empty.")

    groups = None

    if "groups" in data:
        groups = np.asarray(data["groups"])

        if groups.ndim != 1:
            groups = groups.reshape(-1)

        if len(groups) != len(labels):
            raise ValueError(
                "Optional 'groups' array must have one value per sample."
            )

    return signals, labels, groups


def load_dataset(data_path):
    """General dataset loader for the supported NPZ format."""

    signals, labels, groups = load_npz_dataset(data_path)

    print("Signals shape:", signals.shape)
    print("Labels shape:", labels.shape)

    if groups is not None:
        print("Groups shape:", groups.shape)
        print("Unique groups:", len(np.unique(groups)))
    else:
        print("Groups: not provided (sample-level split will be used)")

    return signals, labels, groups
