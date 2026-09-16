"""
Dataset loading utilities for
Multichannel Biomedical Signal Classification.
"""

from pathlib import Path

import numpy as np
import pandas as pd



def load_csv_dataset(
    file_path
):
    """
    Load biomedical signal dataset
    from CSV file.

    Parameters
    ----------
    file_path : str or Path
        Dataset path.

    Returns
    -------
    signals : numpy.ndarray
        Signal samples.

    labels : numpy.ndarray
        Class labels.
    """

    data = pd.read_csv(
        file_path
    )


    if "label" not in data.columns:
        raise ValueError(
            "Dataset must contain a 'label' column."
        )


    labels = data["label"].values


    signals = data.drop(
        columns=["label"]
    ).values


    signals = signals.astype(
        np.float32
    )


    return signals, labels



def load_dataset(
    data_path
):
    """
    General dataset loader.
    """

    path = Path(
        data_path
    )


    if not path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {path}"
        )


    if path.suffix.lower() == ".csv":

        return load_csv_dataset(
            path
        )


    raise ValueError(
        "Unsupported dataset format. Only CSV files are supported."
    )
