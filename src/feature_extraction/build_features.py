"""
Feature building pipeline for multichannel biomedical signals.

Combines:
- Time-domain features
- Frequency-domain features
- Wavelet features
- Global cross-channel features
"""

import numpy as np


from .time_features import (
    extract_multichannel_time_features
)

from .frequency_features import (
    extract_multichannel_frequency_features
)

from .wavelet_features import (
    extract_multichannel_wavelet_features
)

from .global_features import (
    extract_global_features
)



DEFAULT_FS = 8000



def build_feature_vector(
    X,
    fs=DEFAULT_FS,
    wavelet="db4",
    level=4
):
    """
    Build complete feature vector
    from one multichannel signal.

    Parameters
    ----------
    X : numpy.ndarray
        Signal shape:
        (time_samples, channels)

    Returns
    -------
    feature_vector : numpy.ndarray
        Combined features.

    feature_names : list
        Feature names.
    """

    features = {}


    # Time-domain features

    features.update(
        extract_multichannel_time_features(
            X
        )
    )


    # Frequency-domain features

    features.update(
        extract_multichannel_frequency_features(
            X,
            fs
        )
    )


    # Wavelet features

    features.update(
        extract_multichannel_wavelet_features(
            X,
            wavelet,
            level
        )
    )


    # Cross-channel features

    features.update(
        extract_global_features(
            X
        )
    )


    feature_names = sorted(
        features.keys()
    )


    feature_vector = np.array(
        [
            features[name]
            for name in feature_names
        ],
        dtype=np.float32
    )


    return (
        feature_vector,
        feature_names
    )



def build_dataset_features(
    signals,
    fs=DEFAULT_FS
):
    """
    Build feature matrix from dataset.

    Parameters
    ----------
    signals : list or ndarray
        Multiple multichannel signals.

    Returns
    -------
    X_features : numpy.ndarray
        Feature matrix.

    feature_names : list
        Feature names.
    """


    all_features = []

    feature_names = None


    for signal_sample in signals:


        vector, names = build_feature_vector(
            signal_sample,
            fs
        )


        all_features.append(
            vector
        )


        if feature_names is None:

            feature_names = names



    X_features = np.vstack(
        all_features
    )


    return (
        X_features,
        feature_names
    )
