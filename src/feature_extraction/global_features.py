"""
Global and cross-channel feature extraction utilities
for multichannel biomedical signal classification.

Includes:
- Channel statistics
- Inter-channel correlation
- Cross-channel interaction features
"""


import numpy as np



def channel_statistics(X):
    """
    Calculate global statistics across channels.

    Input shape:
    (time_samples, channels)
    """


    X = np.asarray(
        X,
        dtype=np.float32
    )


    features = {}


    features["global_mean"] = float(
        np.mean(X)
    )


    features["global_std"] = float(
        np.std(X)
    )


    features["global_variance"] = float(
        np.var(X)
    )


    features["global_max"] = float(
        np.max(X)
    )


    features["global_min"] = float(
        np.min(X)
    )


    return features



def channel_correlation(X):
    """
    Calculate correlation between channels.
    """


    X = np.asarray(
        X,
        dtype=np.float32
    )


    features = {}


    channels = X.shape[1]


    corr_matrix = np.corrcoef(
        X.T
    )


    corr_matrix = np.nan_to_num(
        corr_matrix,
        nan=0.0
    )


    for i in range(channels):

        for j in range(i + 1, channels):

            features[
                f"corr_ch{i+1}_ch{j+1}"
            ] = float(
                corr_matrix[i, j]
            )


    return features



def channel_difference_features(X):
    """
    Extract differences between channels.
    """


    X = np.asarray(
        X,
        dtype=np.float32
    )


    features = {}


    channels = X.shape[1]


    for i in range(channels):

        for j in range(i + 1, channels):

            diff = (
                X[:, i]
                -
                X[:, j]
            )


            features[
                f"diff_ch{i+1}_ch{j+1}_mean"
            ] = float(
                np.mean(diff)
            )


            features[
                f"diff_ch{i+1}_ch{j+1}_std"
            ] = float(
                np.std(diff)
            )


    return features



def extract_global_features(X):
    """
    Extract complete global feature set.

    Input shape:
    (time_samples, channels)

    Returns:
    Combined feature dictionary.
    """


    features = {}


    features.update(
        channel_statistics(X)
    )


    features.update(
        channel_correlation(X)
    )


    features.update(
        channel_difference_features(X)
    )


    return features
