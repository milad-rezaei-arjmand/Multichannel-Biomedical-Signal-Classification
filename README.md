# Multichannel Biomedical Signal Classification

## Multichannel Biomedical Signal Classification Using Feature Engineering and CatBoost Ensemble

This repository presents an end-to-end machine learning framework for
multichannel biomedical signal classification using signal
preprocessing, multi-domain handcrafted feature extraction, feature
selection, and CatBoost ensemble learning.

The framework processes four-channel biomedical signals and combines
temporal, spectral, wavelet, and cross-channel information to build a
complete classification pipeline.

------------------------------------------------------------------------

# Overview

Biomedical signals contain complex temporal, frequency, and
inter-channel characteristics. Effective preprocessing and feature
representation are essential for reliable automated classification.

This project provides a complete workflow including:

-   Multichannel signal preprocessing
-   Time-domain feature extraction
-   Frequency-domain feature extraction
-   Wavelet-based feature extraction
-   Cross-channel feature analysis
-   Feature selection using ANOVA F-test
-   CatBoost ensemble classification
-   Model saving and inference
-   Performance evaluation

------------------------------------------------------------------------

# Dataset

The original biomedical dataset is not included in this repository due
to data usage restrictions and privacy considerations.

Users should provide their own legally available dataset following the
required format.

## Dataset Format

The framework supports:

-   `.npz`
-   `.csv`

### NPZ Format

The dataset should contain:

signals\
labels

Expected shapes:

signals: (number_of_samples, time_points, channels)

labels: (number_of_samples,)

Example:

signals.shape = (1000, 20000, 4)

labels.shape = (1000,)

Classification classes:

  Class
  -------
  N
  MVP
  MS
  MR
  AS

------------------------------------------------------------------------

# Methodology

The complete pipeline:

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

Prediction

↓

Performance Evaluation

------------------------------------------------------------------------

# Signal Processing Pipeline

## Preprocessing

The preprocessing stage includes:

-   Invalid value removal
-   Signal cleaning
-   Band-pass filtering
-   Z-score normalization

Configuration:

Sampling frequency: 8000 Hz

Low cutoff frequency: 5 Hz

High cutoff frequency: 3500 Hz

Filter order: 4

Implementation:

src/preprocessing/signal_processing.py

------------------------------------------------------------------------

# Feature Engineering

The framework extracts handcrafted features from multiple domains.

## Time-domain Features

Includes:

-   Mean
-   Standard deviation
-   Variance
-   RMS
-   Signal range
-   Skewness
-   Kurtosis
-   Percentile statistics
-   Zero crossing rate

Implementation:

src/feature_extraction/time_features.py

------------------------------------------------------------------------

## Frequency-domain Features

Includes:

-   FFT-based spectral features
-   Dominant frequency
-   Spectral energy
-   Spectral entropy
-   Frequency band power

Implementation:

src/feature_extraction/frequency_features.py

------------------------------------------------------------------------

## Wavelet Features

Includes:

-   Discrete wavelet decomposition
-   Wavelet coefficient statistics
-   Multi-resolution analysis

Implementation:

src/feature_extraction/wavelet_features.py

------------------------------------------------------------------------

## Global Cross-channel Features

Includes:

-   Channel correlation
-   Global signal statistics
-   Channel difference features

Implementation:

src/feature_extraction/global_features.py

------------------------------------------------------------------------

# Feature Selection

Feature selection is performed using:

SelectKBest\
ANOVA F-test

------------------------------------------------------------------------

# Classification Model

## CatBoost Ensemble

The main classification model is based on CatBoost gradient boosting.

The ensemble approach uses:

-   Multiple CatBoost classifiers
-   Different random seeds
-   Validation-based early stopping
-   Probability averaging

Implementation:

src/models/catboost_classifier.py

------------------------------------------------------------------------

# Training

Install dependencies:

pip install -r requirements.txt

Run training:

python3 src/train.py --dataset path/to/dataset.npz

------------------------------------------------------------------------

# Prediction / Inference

After training:

python3 src/predict.py --input signal.npy

------------------------------------------------------------------------

# Evaluation

The evaluation module provides:

-   Accuracy
-   Precision
-   Recall
-   Macro F1-score
-   Weighted F1-score
-   Confusion Matrix
-   Classification Report

------------------------------------------------------------------------

# Dependencies

numpy\
pandas\
scikit-learn\
scipy\
PyWavelets\
catboost

------------------------------------------------------------------------

# Author

Milad Rezaei Arjmand

M.Sc. Student in Biomedical Engineering (Bioelectric)
