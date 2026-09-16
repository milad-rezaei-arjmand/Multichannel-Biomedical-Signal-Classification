"""
Time-domain feature extraction utilities
for multichannel biomedical signal classification.

Features include:
- Mean
- Standard deviation
- Variance
- RMS
- Range
- Skewness
- Kurtosis
- Percentiles
- Zero crossing rate
"""


import numpy as np
from scipy import stats



def zero_crossing_rate(signal):
    """
    Calculate zero crossing rate.

    Parameters
    ----------
    signal : numpy.ndarray
        Input one-dimensional signal.

    Returns
    -------
    float
        Zero crossing rate.
    """

    signal = np.asarray(
        signal,
        dtype=np.float32
    )


    if len(signal) == 0:
        return 0.0


    crossings = np.where(
        np.diff(np.sign(signal)) != 0
    )[0]


    return len(crossings) / len(signal)



def extract_time_features(signal):
    """
    Extract statistical time-domain features.

    Parameters
    ----------
    signal : numpy.ndarray
        Single channel biomedical signal.

    Returns
    -------
    dict
        Extracted feature dictionary.
    """

    signal = np.asarray(
        signal,
        dtype=np.float32
    )


    features = {}


    # Basic statistics

    features["mean"] = np.mean(
        signal
    )

    features["std"] = np.std(
        signal
    )

    features["variance"] = np.var(
        signal
    )


    # Energy related features

    features["rms"] = np.sqrt(
        np.mean(signal ** 2)
    )


    features["signal_range"] = (
        np.max(signal)
        -
        np.min(signal)
    )


    # Distribution features

    features["skewness"] = np.nan_to_num(
        stats.skew(signal)
    )


    features["kurtosis"] = np.nan_to_num(
        stats.kurtosis(signal)
    )


    # Percentile features

    features["percentile_25"] = np.percentile(
        signal,
        25
    )

    features["percentile_50"] = np.percentile(
        signal,
        50
    )

    features["percentile_75"] = np.percentile(
        signal,
        75
    )


    # Activity feature

    features["zero_crossing_rate"] = (
        zero_crossing_rate(signal)
    )


    return features



def extract_multichannel_time_features(X):
    """
    Extract time-domain features from
    multichannel signals.

    Parameters
    ----------
    X : numpy.ndarray
        Signal shape:
        (time_samples, channels)

    Returns
    -------
    dict
        Combined feature dictionary.
    """


    X = np.asarray(
        X,
        dtype=np.float32
    )


    features = {}


    n_channels = X.shape[1]


    for ch in range(n_channels):

        channel_features = (
            extract_time_features(
                X[:, ch]
            )
        )


        for name, value in channel_features.items():

            features[
                f"ch{ch+1}_{name}"
            ] = value


    return features
