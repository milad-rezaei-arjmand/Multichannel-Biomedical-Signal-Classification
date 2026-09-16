"""
Signal preprocessing utilities for
Multichannel Biomedical Signal Classification.

Includes:
- NaN/Inf handling
- Bandpass filtering
- Channel normalization
- Multichannel preprocessing

Designed for multichannel biomedical signal classification pipeline.
"""


import numpy as np
from scipy import signal



DEFAULT_FS = 1000



def clean_signal(x):
    """
    Replace invalid values in signal.

    Parameters
    ----------
    x : numpy.ndarray
        Input signal.

    Returns
    -------
    numpy.ndarray
        Cleaned signal.
    """

    x = np.asarray(
        x,
        dtype=np.float32
    )


    x = np.nan_to_num(
        x,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )


    return x.astype(
        np.float32
    )



def butter_bandpass(
    x,
    fs,
    lo,
    hi,
    order=4
):
    """
    Apply Butterworth bandpass filter.

    Parameters
    ----------
    x : numpy.ndarray
        Input signal.

    fs : float
        Sampling frequency.

    lo : float
        Lower cutoff frequency.

    hi : float
        Upper cutoff frequency.

    order : int
        Filter order.

    Returns
    -------
    numpy.ndarray
        Filtered signal.
    """

    x = clean_signal(x)


    nyq = 0.5 * fs


    low = max(
        lo / nyq,
        1e-6
    )

    high = min(
        hi / nyq,
        0.999999
    )


    b, a = signal.butter(
        order,
        [low, high],
        btype="bandpass"
    )


    filtered = signal.filtfilt(
        b,
        a,
        x
    )


    return filtered.astype(
        np.float32
    )



def normalize_signal(x):
    """
    Apply Z-score normalization.

    Parameters
    ----------
    x : numpy.ndarray
        Input signal.

    Returns
    -------
    numpy.ndarray
        Normalized signal.
    """

    x = clean_signal(x)


    mean = np.mean(x)

    std = np.std(x) + 1e-6


    x = (
        x - mean
    ) / std


    return x.astype(
        np.float32
    )



def preprocess_channel(
    x,
    fs=DEFAULT_FS,
    low_freq=5.0,
    high_freq=3500.0,
    filter_order=4
):
    """
    Complete preprocessing pipeline for one signal channel.

    Steps:
    1. Remove invalid values
    2. Bandpass filtering
    3. Z-score normalization
    """

    x = clean_signal(x)


    x = butter_bandpass(
        x,
        fs,
        low_freq,
        high_freq,
        filter_order
    )


    x = normalize_signal(
        x
    )


    return x



def preprocess_multichannel_signal(
    signals,
    fs=DEFAULT_FS,
    low_freq=5.0,
    high_freq=3500.0,
    filter_order=4
):
    """
    Apply preprocessing pipeline
    to multichannel biomedical signals.

    Parameters
    ----------
    signals : numpy.ndarray
        Multichannel signals.

    fs : float
        Sampling frequency.

    Returns
    -------
    numpy.ndarray
        Preprocessed multichannel signals.
    """


    signals = np.asarray(
        signals,
        dtype=np.float32
    )


    processed = []


    for channel in signals.T:

        processed.append(
            preprocess_channel(
                channel,
                fs,
                low_freq,
                high_freq,
                filter_order
            )
        )
def preprocess_dataset(
    signals,
    fs=DEFAULT_FS,
    low_freq=5.0,
    high_freq=3500.0,
    filter_order=4
):
    """
    Preprocess complete dataset.

    Expected input:

    (samples, time_points, channels)

    Example:

    (1000, 20000, 4)
    """

    signals = np.asarray(
        signals,
        dtype=np.float32
    )


    if signals.ndim != 3:

        raise ValueError(
            "Dataset must have shape "
            "(samples, time_points, channels)"
        )


    processed = []


    for sample in signals:

        processed.append(
            preprocess_multichannel_signal(
                sample,
                fs,
                low_freq,
                high_freq,
                filter_order
            )
        )
def preprocess_dataset(
    signals,
    fs=DEFAULT_FS,
    low_freq=5.0,
    high_freq=3500.0,
    filter_order=4
):
    """
    Preprocess complete dataset.

    Expected input:

    (samples, time_points, channels)
    """

    signals = np.asarray(
        signals,
        dtype=np.float32
    )

    if signals.ndim != 3:
        raise ValueError(
            "Dataset must have shape "
            "(samples, time_points, channels)"
        )

    processed = []

    for sample in signals:

        processed.append(
            preprocess_multichannel_signal(
                sample,
                fs,
                low_freq,
                high_freq,
                filter_order
            )
        )

    return np.asarray(
        processed,
        dtype=np.float32
    )

    return np.asarray(
        processed,
        dtype=np.float32
    )

    return np.asarray(
        processed,
        dtype=np.float32
    ).T
