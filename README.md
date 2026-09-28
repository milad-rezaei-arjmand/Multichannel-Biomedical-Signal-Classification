# Multichannel Biomedical Signal Classification

## Feature Engineering and CatBoost Ensemble for Four-Channel Biomedical Signals

This repository provides an end-to-end machine-learning framework for classifying four-channel biomedical signals using deterministic preprocessing, multi-domain handcrafted feature extraction, ANOVA-based feature selection, and a CatBoost ensemble.

The project combines:

- signal cleaning and band-pass filtering,
- per-channel normalization,
- time-domain features,
- frequency-domain features,
- discrete-wavelet features,
- global and cross-channel features,
- ANOVA F-test feature selection,
- CatBoost ensemble classification,
- saved-model inference,
- and reproducible evaluation outputs.

---

## Classification Task

The configured five-class problem uses:

| Label | Class |
|---|---|
| N | Normal |
| MVP | Mitral Valve Prolapse |
| MS | Mitral Stenosis |
| MR | Mitral Regurgitation |
| AS | Aortic Stenosis |

The public pipeline expects four synchronized channels.

---

## Dataset

The original experimental dataset is not distributed in this repository.

The cleaned public training pipeline supports **NPZ** input with:

```text
signals : (samples, time_points, 4)
labels  : (samples,)
groups  : (samples,)  # optional
```

If `groups` is present, it is used to keep groups disjoint across train, validation, and test splits.

If `groups` is absent, the pipeline uses sample-level stratified splitting.

See [`data/README.md`](data/README.md) for the complete format.

### Important CSV note

An earlier version of this repository advertised a simple CSV format. That loader returned a 2-D table, while the preprocessing pipeline requires 3-D time-series data shaped as `(samples, time_points, channels)`. Because the old CSV convention did not preserve sample/time boundaries, it has been removed from the supported end-to-end workflow.

---

## Signal Preprocessing

Default configuration:

| Parameter | Value |
|---|---:|
| Sampling frequency | 8000 Hz |
| Low cutoff | 5 Hz |
| High cutoff | 3500 Hz |
| Butterworth order | 4 |
| Channels | 4 |

Each channel is:

1. converted to finite numeric values,
2. band-pass filtered,
3. Z-score normalized.

Implementation:

```text
src/preprocessing/signal_processing.py
```

---

## Feature Engineering

The public feature extractor combines four domains.

### Time domain

Per channel:

- mean,
- standard deviation,
- variance,
- RMS,
- range,
- skewness,
- kurtosis,
- 25th/50th/75th percentiles,
- zero-crossing rate.

For four channels: **44 features**.

### Frequency domain

Per channel:

- dominant frequency,
- spectral energy,
- spectral entropy,
- low-band power,
- high-band power.

For four channels: **20 features**.

### Wavelet domain

Using `db4` with default decomposition level 4, when that level is supported by the signal length:

- mean,
- standard deviation,
- energy,
- maximum,
- minimum

for each wavelet coefficient array.

For four channels at level 4: **100 features**.

### Global / cross-channel

- global mean/std/variance/min/max,
- pairwise channel correlations,
- pairwise channel-difference mean/std.

For four channels: **23 features**.

### Current default feature count

For a four-channel signal long enough to support wavelet level 4:

```text
44 + 20 + 100 + 23 = 187 features
```

The training code uses `SelectKBest(f_classif)` fit on the training split only. The default maximum is 1000 features, so with the current 187-feature extractor all 187 are retained unless a smaller `--k-features` value is supplied.

---

## Data Splitting

Default proportions are approximately:

```text
Train      64%
Validation 16%
Test       20%
```

When no group identifiers are available, splits are stratified at the sample level.

When the NPZ file contains a `groups` array, `GroupShuffleSplit` is used and train/validation/test group IDs are checked for overlap.

This distinction matters when several samples may originate from the same participant or recording.

---

## CatBoost Ensemble

The main model consists of three CatBoost multiclass classifiers with different random seeds/depths.

The ensemble uses:

- validation-based early stopping,
- balanced class weighting,
- probability averaging,
- CPU execution by default,
- optional GPU execution.

Implementation:

```text
src/models/catboost_classifier.py
```

---

## Installation

```bash
git clone https://github.com/milad-rezaei-arjmand/Multichannel-Biomedical-Signal-Classification.git
cd Multichannel-Biomedical-Signal-Classification

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

---

## Training

CPU:

```bash
python src/train.py \
  --dataset data/dataset.npz
```

GPU:

```bash
python src/train.py \
  --dataset data/dataset.npz \
  --task-type GPU
```

Optional feature-count cap:

```bash
python src/train.py \
  --dataset data/dataset.npz \
  --k-features 128
```

Generated evaluation files are written to:

```text
outputs/evaluation/
```

Model files and metadata are written to:

```text
outputs/checkpoints/
```

Both directories are ignored by Git.

---

## Inference

Inference expects one NumPy signal file with:

```text
(time_points, 4)
```

Example:

```bash
python src/predict.py \
  --input path/to/signal.npy \
  --model-dir outputs/checkpoints
```

Prediction labels are derived from the trained CatBoost model's `classes_` ordering, avoiding assumptions about probability-column order.

---

## Evaluation

The evaluation module reports:

- Accuracy
- Macro precision
- Macro recall
- Macro F1-score
- Weighted F1-score
- Confusion matrix
- Classification report

The configured class order is passed explicitly to confusion-matrix and classification-report functions.

---

## Archived Experimental Result

The repository preserves an earlier reported experiment summary in:

[`results/final_metrics.md`](results/final_metrics.md)

That historical document reports:

- 1000 samples,
- 640/160/200 train/validation/test split,
- 98% test accuracy,
- 98% macro F1,
- 98% weighted F1.

However, the earlier result description also states a `1514 -> 1000` feature-selection pipeline, while the current public feature extractor produces **187 features** under its default four-channel configuration.

Because the original dataset and the exact experiment-generation implementation needed to reconcile that discrepancy are not distributed, the 98% result is retained as an **archived reported result** rather than presented as a direct reproduction of the cleaned public pipeline.

See [`results/README.md`](results/README.md) for details.

---

## Repository Structure

```text
Multichannel-Biomedical-Signal-Classification/
├── data/
│   └── README.md
├── results/
│   ├── README.md
│   └── final_metrics.md
├── src/
│   ├── config.py
│   ├── data_loader.py
│   ├── train.py
│   ├── predict.py
│   ├── preprocessing/
│   ├── feature_extraction/
│   ├── models/
│   └── evaluation/
├── .gitignore
├── LICENSE
├── requirements.txt
└── README.md
```

---

## Reproducibility and Scope

The public repository is intended as a reproducible engineering framework.

Important limitations:

- the original experimental dataset is not publicly included,
- the archived result cannot be fully reproduced from public files alone,
- sample-level splitting can overestimate generalization if repeated samples from the same subject are present,
- group-disjoint validation requires group identifiers in the NPZ file,
- and the framework is not a clinically validated diagnostic system.

---

## License

Repository code and documentation are provided under the MIT License.

Any external biomedical dataset used with the framework remains subject to its own license, access conditions, and ethical requirements.

---

## Author

**Milad Rezaei Arjmand**

M.Sc. Student in Biomedical Engineering (Bioelectric)
