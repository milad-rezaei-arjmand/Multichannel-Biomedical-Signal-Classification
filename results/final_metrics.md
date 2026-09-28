# Archived Experimental Results

## Status

This file preserves the previously reported final experiment summary for the project.

The original dataset is not distributed in this repository, so these values cannot be independently regenerated from the public repository alone.

In addition, the **current public feature extractor does not match the archived feature-count description**. For a four-channel signal with the current default feature configuration and wavelet level 4 available, the tracked implementation produces:

- 44 time-domain features
- 20 frequency-domain features
- 100 wavelet features
- 23 global/cross-channel features
- **187 total features**

The earlier result document reported `1514 -> 1000` features. Because the exact experiment-generation code/data needed to reconcile that difference are not available in the public repository, the values below are retained as **archived reported results**, not claimed as a direct reproduction of the cleaned pipeline.

---

## Archived Dataset Summary

Previously reported experiment setup:

- Total samples: 1000
- Channels: 4
- Sampling frequency: 8000 Hz
- Five classes: AS, MR, MS, MVP, N

Previously reported split:

| Split | Samples |
|---|---:|
| Train | 640 |
| Validation | 160 |
| Test | 200 |

The historical document described a stratified sample-level split.

If multiple samples originated from the same participant, participant-disjoint validation cannot be confirmed from the archived public files.

---

## Archived Model Description

The reported model was a three-member CatBoost ensemble with:

- multiple random seeds,
- validation-based early stopping,
- probability averaging.

The earlier document stated GPU training, but the previously tracked training entry point did not pass a GPU execution mode into the ensemble trainer. Hardware execution mode is therefore not independently verifiable from the public repository history.

The cleaned training script now exposes:

```bash
--task-type CPU
--task-type GPU
```

explicitly.

---

## Archived Test Performance

| Metric | Reported value |
|---|---:|
| Accuracy | 98.00% |
| Macro F1-score | 98.00% |
| Weighted F1-score | 98.00% |

### Reported per-class metrics

The archived report used the class order:

```text
AS, MR, MS, MVP, N
```

| Class | Precision | Recall | F1-score |
|---|---:|---:|---:|
| AS | 1.00 | 1.00 | 1.00 |
| MR | 1.00 | 0.97 | 0.99 |
| MS | 0.97 | 0.93 | 0.95 |
| MVP | 0.93 | 1.00 | 0.96 |
| N | 1.00 | 1.00 | 1.00 |

### Reported confusion matrix

Class order: `AS, MR, MS, MVP, N`

```text
[[40  0  0  0  0]
 [ 0 39  1  0  0]
 [ 0  0 37  3  0]
 [ 0  0  0 40  0]
 [ 0  0  0  0 40]]
```

This matrix is internally consistent with 196 correct predictions out of 200 samples, corresponding to 98% accuracy.

---

## Current Public Pipeline

The cleaned implementation fixes several reproducibility issues:

- removes the invalid row-wise CSV training format,
- makes CPU/GPU execution explicit,
- maps inference probabilities using the trained CatBoost `classes_` order,
- passes an explicit class order to evaluation functions,
- validates four-channel NPZ input,
- supports optional group-disjoint splitting when an NPZ `groups` array is available,
- records training metadata alongside model checkpoints,
- and keeps generated outputs separate from this archived result record.

New results produced by the cleaned pipeline should be reported separately from the archived 98% result.
