# Final Experimental Results

## Dataset

The experiments were performed on the original multichannel biomedical signal dataset.

Dataset size:

- Total samples: 1000
- Channels: 4
- Sampling frequency: 8000 Hz

Classes:

- AS
- MR
- MS
- MVP
- N


## Feature Representation

The feature extraction pipeline generated:

```
1514 features/sample
```

Feature selection:

```
SelectKBest (ANOVA F-test)

1514 -> 1000 features
```


## Data Split

The dataset was divided using stratified splitting:

| Split | Samples |
|---|---:|
| Train | 640 |
| Validation | 160 |
| Test | 200 |


## Model

CatBoost ensemble classifier:

- GPU training
- Multiple random seeds
- Validation-based early stopping
- Probability averaging


## Test Performance

| Metric | Value |
|---|---:|
| Accuracy | 98.00% |
| Macro F1-score | 98.00% |
| Weighted F1-score | 98.00% |


## Classification Report

| Class | Precision | Recall | F1-score |
|---|---:|---:|---:|
| AS | 1.00 | 1.00 | 1.00 |
| MR | 1.00 | 0.97 | 0.99 |
| MS | 0.97 | 0.93 | 0.95 |
| MVP | 0.93 | 1.00 | 0.96 |
| N | 1.00 | 1.00 | 1.00 |


## Confusion Matrix

```
[[40 0 0 0 0]
 [0 39 1 0 0]
 [0 0 37 3 0]
 [0 0 0 40 0]
 [0 0 0 0 40]]
```


## Reproducibility

The complete dataset is not included due to usage restrictions.

The repository provides the complete training and inference pipeline.
