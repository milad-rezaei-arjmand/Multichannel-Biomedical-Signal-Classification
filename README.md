# Multichannel-Biomedical-Signal-Classification

## Multichannel Biomedical Signal Classification Using Feature Engineering and CatBoost Ensemble

This repository presents a machine learning framework for multichannel biomedical signal classification using signal preprocessing, multi-domain feature extraction, feature selection, and CatBoost-based ensemble learning.

The framework processes four-channel biomedical signals and investigates the effectiveness of handcrafted time-domain, frequency-domain, wavelet, and cross-channel features combined with machine learning classification.

---

# Overview

Biomedical signals contain complex temporal, spectral, and inter-channel patterns. Effective preprocessing and feature representation are essential for reliable classification.

This project develops an end-to-end classification pipeline including:

- Signal preprocessing
- Multi-domain feature extraction
- Feature selection
- CatBoost ensemble classification
- Binary specialist classification
- Performance evaluation

The main objective is to investigate how signal processing methods and machine learning algorithms can be combined for automated biomedical signal classification.

---

# Dataset

The dataset consists of multichannel biomedical signal recordings.

Dataset characteristics:

- Multichannel biomedical signals
- Four signal channels
- Sampling frequency: 8000 Hz
- Multi-class classification task

The signal channels include:

- Amplitude
- Velocity
- Acceleration
- Alpha

Target classes:

| Class |
|---|
| N |
| MVP |
| MS |
| MR |
| AS |

The original biomedical signal files are not included in this repository due to data usage and distribution restrictions.

---

# Methodology

The proposed workflow:

```text
Multichannel Biomedical Signal

            ↓

Signal Preprocessing

            ↓

Multi-domain Feature Extraction

            ↓

Feature Selection

            ↓

CatBoost Ensemble Classification

            ↓

Specialist Binary Classification

            ↓

Performance Evaluation
```

---

# Signal Processing Pipeline

## Preprocessing

The preprocessing stage includes:

- Invalid value removal
- Signal cleaning
- Band-pass filtering
- Signal normalization

Filtering configuration:

```text
Low frequency cutoff: 5 Hz

High frequency cutoff: 3500 Hz

Filter order: 4
```

The preprocessing module is implemented in:

```text
src/preprocessing/signal_processing.py
```

---

# Feature Engineering

A multi-domain handcrafted feature extraction strategy is applied.

## Time-domain Features

Extracted features include:

- Mean
- Standard deviation
- Variance
- RMS
- Signal range
- Skewness
- Kurtosis
- Percentile statistics
- Zero crossing rate

Implementation:

```text
src/feature_extraction/time_features.py
```

---

## Frequency-domain Features

Frequency analysis includes:

- FFT-based spectral features
- Dominant frequency
- Spectral energy
- Spectral entropy
- Frequency band power

Implementation:

```text
src/feature_extraction/frequency_features.py
```

---

## Wavelet Features

Wavelet-based analysis includes:

- Discrete wavelet decomposition
- Wavelet coefficient statistics
- Multi-resolution analysis

Implementation:

```text
src/feature_extraction/wavelet_features.py
```

---

## Global Cross-channel Features

Cross-channel features include:

- Global signal statistics
- Channel correlation
- Channel difference features

Implementation:

```text
src/feature_extraction/global_features.py
```

---

# Feature Selection

Feature selection is performed using:

```text
SelectKBest (ANOVA F-test)
```

The selected feature subset is used as input for the classification models.

Implementation:

```text
src/train.py
```

---

# Classification Models

## CatBoost Multiclass Ensemble

The main classification model is based on CatBoost gradient boosting.

The ensemble approach uses:

- Multiple CatBoost classifiers
- Different random seeds
- Validation-based early stopping
- Probability averaging

Implementation:

```text
src/models/catboost_classifier.py
```

---

## Binary Specialist Classifier

A specialist classifier is provided for difficult class separation problems.

The specialist model:

- Selects two target classes
- Creates a binary classification problem
- Uses a dedicated CatBoost classifier

Implementation:

```text
src/models/specialist_model.py
```

---

# Evaluation

The evaluation pipeline provides:

- Accuracy
- Precision
- Recall
- Macro F1-score
- Weighted F1-score
- Confusion Matrix
- Classification Report

Implementation:

```text
src/evaluation/metrics.py
```

Generated outputs:

```text
results/

├── metrics.json

├── confusion_matrix.csv

└── classification_report.txt
```

---

# Project Structure

```text
Multichannel-Biomedical-Signal-Classification/

├── data/
│   └── Dataset information

├── notebooks/
│   └── Experimental notebooks

├── results/
│   └── Evaluation outputs

├── src/

│   ├── data_loader.py
│   ├── train.py

│   ├── preprocessing/
│   │   └── signal_processing.py

│   ├── feature_extraction/
│   │   ├── time_features.py
│   │   ├── frequency_features.py
│   │   ├── wavelet_features.py
│   │   ├── global_features.py
│   │   └── build_features.py

│   ├── models/
│   │   ├── catboost_classifier.py
│   │   └── specialist_model.py

│   └── evaluation/
│       └── metrics.py

├── requirements.txt

├── LICENSE

└── README.md
```

---

# Installation

Install required dependencies:

```bash
pip install -r requirements.txt
```

Main dependencies:

```text
numpy
pandas
scikit-learn
scipy
PyWavelets
catboost
```

---

# Usage

After preparing the dataset:

1. Place the dataset file inside:

```text
data/
```

2. Update the dataset path in:

```text
src/train.py
```

3. Run the training pipeline:

```bash
python src/train.py
```

The pipeline performs:

```text
Dataset Loading

↓

Signal Preprocessing

↓

Feature Extraction

↓

Model Training

↓

Evaluation

↓

Result Saving
```

---

# Future Work

Future improvements include:

- Deep learning based waveform models
- CNN and transformer architectures
- Hybrid machine learning and deep learning approaches
- Advanced feature learning methods
- Evaluation on larger biomedical datasets

---

# Author

**Milad Rezaei Arjmand**

M.Sc. Student in Biomedical Engineering (Bioelectric)

Research Interests:

- Medical Artificial Intelligence
- Biomedical Signal Processing
- Medical Imaging
- Deep Learning
