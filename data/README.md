# Dataset

This directory contains the dataset used for the Multichannel Biomedical Signal Classification project.

## Dataset Overview

The dataset consists of multichannel biomedical signal recordings designed for supervised classification tasks.

The dataset includes:

- Multichannel signal recordings
- Corresponding class labels
- Signal samples prepared for feature extraction and machine learning classification

Due to dataset availability, privacy considerations, and file size limitations, raw data files are not included in this repository.

---

## Data Format

The expected input format is:

```text
channel_1
channel_2
channel_3
channel_4
label
```

Each sample should contain:

```text
channel_1, channel_2, channel_3, channel_4, label
```

Example:

```text
channel_1,channel_2,channel_3,channel_4,label
0.12,0.45,0.31,0.22,Class_A
0.21,0.33,0.51,0.18,Class_B
```

---

## Dataset Loading

The dataset is loaded using:

```text
src/data_loader.py
```

The loader provides:

- Dataset reading
- Signal extraction
- Label separation

---

## Usage

To use your own dataset:

1. Place the dataset file inside this directory.

Example:

```text
data/dataset.csv
```

2. Update the dataset path in:

```text
src/train.py
```

3. Run the training pipeline.

---

## Data Privacy

This repository does not distribute raw biomedical recordings.

Users should provide their own legally available dataset before running experiments.
