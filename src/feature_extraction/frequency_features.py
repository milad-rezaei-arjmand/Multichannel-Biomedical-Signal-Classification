"""
Frequency-domain feature extraction utilities
for multichannel biomedical signal classification.

Features include:
- FFT based spectral features
- Dominant frequency
- Spectral energy
- Spectral entropy
- Frequency band power
"""


import numpy as np
from scipy import signal



def compute_fft(signal_data, fs):
    """
    Compute frequency spectrum using FFT.
    """

    signal_data = np.asarray(
        signal_data,
        dtype=np.float32
    )


    n = len(signal_data)


    if n == 0:
        return (
            np.array([]),
            np.array([])
        )


    fft_values = np.fft.rfft(
        signal_data
    )


    magnitude = (
        np.abs(fft_values)
        /
        n
    )


    frequencies = np.fft.rfftfreq(
        n,
        d=1/fs
    )


    return frequencies, magnitude



def dominant_frequency(signal_data, fs):
    """
    Find dominant frequency component.
    """

    freq, mag = compute_fft(
        signal_data,
        fs
    )


    if len(mag) <= 1:
        return 0.0


    index = np.argmax(
        mag[1:]
    ) + 1


    return float(
        freq[index]
    )



def spectral_energy(signal_data, fs):
    """
    Calculate total spectral energy.
    """

    _, mag = compute_fft(
        signal_data,
        fs
    )


    return float(
        np.sum(mag ** 2)
    )



def spectral_entropy(signal_data, fs):
    """
    Calculate spectral entropy.
    """

    _, mag = compute_fft(
        signal_data,
        fs
    )


    if len(mag) == 0:
        return 0.0


    power = mag ** 2


    power = power / (
        np.sum(power)
        +
        1e-12
    )


    entropy = -np.sum(
        power *
        np.log2(
            power + 1e-12
        )
    )


    return float(entropy)



def band_power(
    signal_data,
    fs,
    low,
    high
):
    """
    Calculate power inside frequency band.
    """


    freq, power = signal.welch(
        signal_data,
        fs=fs
    )


    mask = (
        (freq >= low)
        &
        (freq <= high)
    )


    if not np.any(mask):
        return 0.0


    return float(
      np.trapezoid(
        power[mask],
        freq[mask]
         )
    )



def extract_frequency_features(
    signal_data,
    fs
):
    """
    Extract frequency-domain features.
    """

    features = {}


    features["dominant_frequency"] = (
        dominant_frequency(
            signal_data,
            fs
        )
    )


    features["spectral_energy"] = (
        spectral_energy(
            signal_data,
            fs
        )
    )


    features["spectral_entropy"] = (
        spectral_entropy(
            signal_data,
            fs
        )
    )


    features["low_band_power"] = (
        band_power(
            signal_data,
            fs,
            0,
            fs / 4
        )
    )


    features["high_band_power"] = (
        band_power(
            signal_data,
            fs,
            fs / 4,
            fs / 2
        )
    )


    return features



def extract_multichannel_frequency_features(
    X,
    fs
):
    """
    Extract frequency features from
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
            extract_frequency_features(
                X[:, ch],
                fs
            )
        )


        for name, value in channel_features.items():

            features[
                f"ch{ch+1}_{name}"
            ] = value


    return features
