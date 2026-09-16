# Dataset

## Overview

The original biomedical signal dataset is not included in this
repository due to data usage restrictions and privacy considerations.

This repository provides the complete machine learning pipeline,
including:

-   Signal preprocessing
-   Multi-domain feature extraction
-   Feature selection
-   CatBoost ensemble classification
-   Model evaluation

Users can apply the pipeline by providing their own compatible
multichannel biomedical signal dataset.

------------------------------------------------------------------------

## Dataset Structure

The framework is designed for multichannel biomedical signals.

Expected input format:

``` text
(samples, time_points, channels)
```

Example:

``` text
(1000, 20000, 4)
```

where:

-   `samples` = number of signal samples
-   `time_points` = number of temporal points per signal
-   `channels` = number of synchronized signal channels

------------------------------------------------------------------------

## Signal Channels

The model expects four input channels:

  Channel     Description
  ----------- --------------
  Channel 1   Amplitude
  Channel 2   Velocity
  Channel 3   Acceleration
  Channel 4   Alpha

------------------------------------------------------------------------

## Labels

The classification task contains five target classes:

  Label   Description
  ------- -----------------------
  N       Normal
  MVP     Mitral Valve Prolapse
  MS      Mitral Stenosis
  MR      Mitral Regurgitation
  AS      Aortic Stenosis

The label array format:

``` text
(samples,)
```

Example:

``` text
[
N,
MVP,
MS,
MR,
AS
]
```

------------------------------------------------------------------------

## Dataset Preparation

Place your dataset inside:

``` text
data/
```

Example:

``` text
data/
├── dataset.csv
└── README.md
```

Update the dataset path in:

``` text
src/train.py
```

Example:

``` python
DATASET_PATH = "data/dataset.csv"
```

------------------------------------------------------------------------

## CSV Input Format

For CSV-based datasets, the expected columns are:

``` text
Amplitude,Velocity,Acceleration,Alpha,label
```

Example:

``` text
Amplitude,Velocity,Acceleration,Alpha,label
0.12,0.31,0.21,0.45,N
0.15,0.28,0.19,0.41,MVP
0.10,0.35,0.25,0.50,MS
```

------------------------------------------------------------------------

## Recommended Data Format

For large biomedical signal datasets, a structured NumPy format is
recommended:

``` text
dataset.npz
```

containing:

``` python
signals
labels
```

Expected shapes:

``` text
signals.shape = (samples, time_points, channels)

labels.shape = (samples,)
```

Example:

``` text
signals.shape

(1000, 20000, 4)


labels.shape

(1000,)
```

------------------------------------------------------------------------

## Training Pipeline

After preparing the dataset:

``` bash
python3 src/train.py
```

The pipeline performs:

``` text
Multichannel Biomedical Signal

        ↓

Signal Preprocessing

        ↓

Feature Extraction

        ↓

Feature Selection

        ↓

CatBoost Ensemble Classification

        ↓

Performance Evaluation
```

------------------------------------------------------------------------

## Dataset Availability

The complete dataset used for experiments is not distributed with this
repository.

Users should provide their own dataset following the required structure
before running experiments.
