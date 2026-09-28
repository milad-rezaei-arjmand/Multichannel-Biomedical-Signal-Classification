"""
Central configuration for Multichannel Biomedical Signal Classification.
"""

DEFAULT_FS = 8000

N_CHANNELS = 4

CHANNEL_NAMES = [
    "Amplitude",
    "Velocity",
    "Acceleration",
    "Alpha",
]

CLASS_NAMES = [
    "N",
    "MVP",
    "MS",
    "MR",
    "AS",
]

LOW_FREQ = 5.0
HIGH_FREQ = 3500.0
FILTER_ORDER = 4

RANDOM_STATE = 42
