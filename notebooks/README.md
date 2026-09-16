# Notebooks

This directory contains exploratory analysis and experimental notebooks for the Multichannel Biomedical Signal Classification project.

---

## Purpose

The notebooks are used for:

- Dataset exploration
- Signal visualization
- Feature analysis
- Model experiments
- Result inspection

---

## Recommended Organization

A recommended notebook structure:

```text
notebooks/

├── 01_data_exploration.ipynb

├── 02_signal_analysis.ipynb

├── 03_feature_extraction_analysis.ipynb

├── 04_model_training_experiments.ipynb

└── 05_results_visualization.ipynb
```

---

## Relationship With Source Code

The notebooks are designed for analysis and experimentation.

The main reproducible training pipeline is implemented in:

```text
src/train.py
```

Core processing modules are available in:

```text
src/

├── preprocessing/

├── feature_extraction/

├── models/

└── evaluation/
```

---

## Usage

Notebooks can be used to:

1. Explore biomedical signal characteristics.
2. Analyze extracted features.
3. Compare model behaviors.
4. Visualize classification results.

Experimental notebooks should complement the main source-code pipeline.
