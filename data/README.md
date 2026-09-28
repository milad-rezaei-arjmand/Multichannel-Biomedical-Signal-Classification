# Dataset

## Availability

The original dataset used for the archived experiment is not distributed in this repository.

The public training pipeline expects users to provide a legally available, compatible four-channel biomedical signal dataset.

---

## Supported Public Input Format

The cleaned public pipeline supports **NPZ** training datasets.

Required arrays:

```text
signals
labels
```

Optional array:

```text
groups
```

Expected shapes:

```text
signals.shape = (samples, time_points, 4)
labels.shape  = (samples,)
groups.shape  = (samples,)   # optional
```

`groups` may contain participant IDs, recording IDs, or another grouping variable. When it is provided, the training pipeline keeps groups disjoint across train, validation, and test splits.

When `groups` is absent, the pipeline uses sample-level stratified splitting.

---

## Four Signal Channels

The project uses four synchronized channels with the following names:

| Channel | Project label |
|---|---|
| 1 | Amplitude |
| 2 | Velocity |
| 3 | Acceleration |
| 4 | Alpha |

The current public pipeline assumes that all samples use the same channel ordering.

---

## Classification Labels

The configured five-class task is:

| Label | Description |
|---|---|
| N | Normal |
| MVP | Mitral Valve Prolapse |
| MS | Mitral Stenosis |
| MR | Mitral Regurgitation |
| AS | Aortic Stenosis |

All five configured labels must be present for the standard training pipeline.

---

## Example NPZ Creation

```python
import numpy as np

np.savez(
    "data/dataset.npz",
    signals=signals,  # (samples, time_points, 4)
    labels=labels,    # (samples,)
    groups=groups,    # optional
)
```

The training dataset itself should not be committed to the repository.

---

## Training

From the repository root:

```bash
python src/train.py --dataset data/dataset.npz
```

For CatBoost GPU execution:

```bash
python src/train.py \
  --dataset data/dataset.npz \
  --task-type GPU
```

Generated models and evaluation files are written under `outputs/` by default.

---

## Why CSV Support Was Removed

An earlier repository version advertised a simple row-wise CSV format. That loader produced a 2-D array, while the preprocessing and feature-extraction pipeline requires a 3-D dataset shaped as:

```text
(samples, time_points, channels)
```

Because the old CSV convention did not encode sample boundaries or time-series structure, it was not a valid end-to-end input format. The public workflow therefore now documents and supports NPZ only.
