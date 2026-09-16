"""
Wavelet-based feature extraction utilities
for multichannel biomedical signal classification.

Features include:
- Wavelet decomposition
- Coefficient energy
- Mean coefficient value
- Standard deviation
- Statistical wavelet features
"""


import numpy as np
import pywt



def wavelet_decomposition(
    signal_data,
    wavelet="db4",
    level=4
):
    """
    Perform discrete wavelet decomposition.
    """


    signal_data = np.asarray(
        signal_data,
        dtype=np.float32
    )


    signal_data = np.nan_to_num(
        signal_data,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )


    max_level = pywt.dwt_max_level(
        len(signal_data),
        pywt.Wavelet(wavelet).dec_len
    )


    level = min(
        level,
        max_level
    )


    coefficients = pywt.wavedec(
        signal_data,
        wavelet,
        level=level
    )


    return coefficients



def coefficient_energy(coefficients):
    """
    Calculate wavelet coefficient energy.
    """

    energies = []


    for coeff in coefficients:

        coeff = np.nan_to_num(
            coeff
        )


        energy = np.sum(
            coeff ** 2
        )


        energies.append(
            float(energy)
        )


    return energies



def extract_wavelet_features(
    signal_data,
    wavelet="db4",
    level=4
):
    """
    Extract statistical wavelet features
    from one signal channel.
    """


    coefficients = wavelet_decomposition(
        signal_data,
        wavelet,
        level
    )


    features = {}


    energies = coefficient_energy(
        coefficients
    )


    for i, coeff in enumerate(coefficients):

        coeff = np.nan_to_num(
            coeff
        )


        features[
            f"wavelet_level_{i}_mean"
        ] = float(
            np.mean(coeff)
        )


        features[
            f"wavelet_level_{i}_std"
        ] = float(
            np.std(coeff)
        )


        features[
            f"wavelet_level_{i}_energy"
        ] = energies[i]


        features[
            f"wavelet_level_{i}_max"
        ] = float(
            np.max(coeff)
        )


        features[
            f"wavelet_level_{i}_min"
        ] = float(
            np.min(coeff)
        )


    return features



def extract_multichannel_wavelet_features(
    X,
    wavelet="db4",
    level=4
):
    """
    Extract wavelet features from
    multichannel signals.

    Input shape:
    (time_samples, channels)
    """


    X = np.asarray(
        X,
        dtype=np.float32
    )


    features = {}


    n_channels = X.shape[1]


    for ch in range(n_channels):

        channel_features = (
            extract_wavelet_features(
                X[:, ch],
                wavelet,
                level
            )
        )


        for name, value in channel_features.items():

            features[
                f"ch{ch+1}_{name}"
            ] = value


    return features
