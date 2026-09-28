"""
Training pipeline for Multichannel Biomedical Signal Classification.

Pipeline
--------
1. Load a 4-channel NPZ dataset
2. Preprocess signals
3. Extract deterministic handcrafted features
4. Split train/validation/test data
5. Fit ANOVA feature selection on training data only
6. Train a CatBoost ensemble
7. Save checkpoints and metadata
8. Evaluate on the held-out test split
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import GroupShuffleSplit, train_test_split


ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))


from src.config import (
    CLASS_NAMES,
    DEFAULT_FS,
    N_CHANNELS,
    RANDOM_STATE,
)
from src.data_loader import load_dataset
from src.evaluation.metrics import (
    evaluate_model,
    save_evaluation_results,
)
from src.feature_extraction.build_features import build_dataset_features
from src.models.catboost_classifier import (
    predict_ensemble,
    save_models,
    train_ensemble,
)
from src.preprocessing.signal_processing import preprocess_dataset


def validate_labels(labels, class_names):
    """Validate labels against the configured five-class task."""

    labels = np.asarray(labels)
    observed = set(labels.astype(str).tolist())
    expected = set(class_names)

    unknown = observed - expected

    if unknown:
        raise ValueError(
            f"Unknown class labels found: {sorted(unknown)}. "
            f"Expected labels are: {list(class_names)}"
        )

    missing = expected - observed

    if missing:
        raise ValueError(
            "Dataset does not contain all configured classes. "
            f"Missing: {sorted(missing)}"
        )

    return labels.astype(str)


def split_indices(
    labels,
    groups=None,
    random_state=RANDOM_STATE,
):
    """
    Create 64/16/20 train/validation/test splits.

    If groups are provided, group identifiers are kept disjoint across splits.
    Otherwise, sample-level stratified splitting is used.
    """

    labels = np.asarray(labels)
    indices = np.arange(len(labels))

    if groups is None:
        train_val_idx, test_idx = train_test_split(
            indices,
            test_size=0.20,
            random_state=random_state,
            stratify=labels,
        )

        train_idx, val_idx = train_test_split(
            train_val_idx,
            test_size=0.20,
            random_state=random_state,
            stratify=labels[train_val_idx],
        )

        return train_idx, val_idx, test_idx, "sample-stratified"

    groups = np.asarray(groups)

    outer = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=random_state,
    )

    train_val_relative, test_relative = next(
        outer.split(indices, labels, groups)
    )

    train_val_idx = indices[train_val_relative]
    test_idx = indices[test_relative]

    inner = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=random_state,
    )

    inner_train_relative, inner_val_relative = next(
        inner.split(
            train_val_idx,
            labels[train_val_idx],
            groups[train_val_idx],
        )
    )

    train_idx = train_val_idx[inner_train_relative]
    val_idx = train_val_idx[inner_val_relative]

    train_groups = set(groups[train_idx].tolist())
    val_groups = set(groups[val_idx].tolist())
    test_groups = set(groups[test_idx].tolist())

    if (
        train_groups.intersection(val_groups)
        or train_groups.intersection(test_groups)
        or val_groups.intersection(test_groups)
    ):
        raise RuntimeError("Group leakage detected across data splits.")

    return train_idx, val_idx, test_idx, "group-disjoint"


def select_features(
    X_train,
    X_val,
    X_test,
    y_train,
    k=1000,
):
    """Fit ANOVA SelectKBest on training data only."""

    selected_k = min(
        int(k),
        X_train.shape[1],
    )

    selector = SelectKBest(
        score_func=f_classif,
        k=selected_k,
    )

    X_train_selected = selector.fit_transform(
        X_train,
        y_train,
    )

    X_val_selected = selector.transform(
        X_val
    )

    X_test_selected = selector.transform(
        X_test
    )

    return (
        X_train_selected,
        X_val_selected,
        X_test_selected,
        selector,
    )


def run_training(
    dataset_path,
    output_dir="outputs/evaluation",
    checkpoint_dir="outputs/checkpoints",
    task_type="CPU",
    k_features=1000,
):
    """Run the complete training and evaluation pipeline."""

    output_dir = Path(output_dir)
    checkpoint_dir = Path(checkpoint_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Loading dataset...")
    signals, labels, groups = load_dataset(
        dataset_path
    )

    if signals.shape[2] != N_CHANNELS:
        raise ValueError(
            f"Expected {N_CHANNELS} channels, "
            f"found {signals.shape[2]}."
        )

    labels = validate_labels(
        labels,
        CLASS_NAMES,
    )

    print("Preprocessing signals...")
    signals = preprocess_dataset(
        signals,
        fs=DEFAULT_FS,
    )

    print("Extracting features...")
    X_features, feature_names = build_dataset_features(
        signals,
        fs=DEFAULT_FS,
    )

    print("Feature matrix:", X_features.shape)

    (
        train_idx,
        val_idx,
        test_idx,
        split_mode,
    ) = split_indices(
        labels,
        groups=groups,
        random_state=RANDOM_STATE,
    )

    X_train = X_features[train_idx]
    X_val = X_features[val_idx]
    X_test = X_features[test_idx]

    y_train = labels[train_idx]
    y_val = labels[val_idx]
    y_test = labels[test_idx]

    print(
        "Split sizes:",
        {
            "train": len(train_idx),
            "validation": len(val_idx),
            "test": len(test_idx),
        },
    )
    print("Split mode:", split_mode)

    print("Selecting features...")
    (
        X_train,
        X_val,
        X_test,
        selector,
    ) = select_features(
        X_train,
        X_val,
        X_test,
        y_train,
        k=k_features,
    )

    selected_mask = selector.get_support()
    selected_feature_names = [
        name
        for name, keep in zip(
            feature_names,
            selected_mask,
        )
        if keep
    ]

    print(
        "Selected feature count:",
        X_train.shape[1],
    )

    with open(
        checkpoint_dir / "feature_selector.pkl",
        "wb",
    ) as handle:
        pickle.dump(
            selector,
            handle,
        )

    print(
        f"Training CatBoost ensemble on {task_type.upper()}..."
    )

    models = train_ensemble(
        X_train,
        y_train,
        X_val,
        y_val,
        task_type=task_type,
    )

    save_models(
        models,
        checkpoint_dir,
    )

    predictions = predict_ensemble(
        models,
        X_test,
    )

    results = evaluate_model(
        y_test,
        predictions,
        CLASS_NAMES,
    )

    save_evaluation_results(
        results,
        output_dir,
    )

    metadata = {
        "sampling_frequency_hz": DEFAULT_FS,
        "n_channels": N_CHANNELS,
        "class_order": list(CLASS_NAMES),
        "model_classes": np.asarray(
            models[0].classes_
        ).astype(str).tolist(),
        "feature_count_before_selection": int(
            len(feature_names)
        ),
        "feature_count_after_selection": int(
            len(selected_feature_names)
        ),
        "selected_feature_names": selected_feature_names,
        "split_mode": split_mode,
        "split_sizes": {
            "train": int(len(train_idx)),
            "validation": int(len(val_idx)),
            "test": int(len(test_idx)),
        },
        "task_type": task_type.upper(),
        "random_state": RANDOM_STATE,
    }

    with open(
        checkpoint_dir / "training_metadata.json",
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            metadata,
            handle,
            indent=2,
        )

    print("\nTraining completed.")
    print(results)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Train the Multichannel Biomedical Signal "
            "Classification pipeline."
        )
    )

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to a dataset NPZ file.",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="outputs/evaluation",
        help="Directory for generated evaluation files.",
    )

    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="outputs/checkpoints",
        help="Directory for trained models and feature selector.",
    )

    parser.add_argument(
        "--task-type",
        choices=["CPU", "GPU"],
        default="CPU",
        help="CatBoost execution device.",
    )

    parser.add_argument(
        "--k-features",
        type=int,
        default=1000,
        help=(
            "Maximum number of ANOVA-selected features. "
            "If larger than the extracted feature count, all features are retained."
        ),
    )

    args = parser.parse_args()

    run_training(
        dataset_path=args.dataset,
        output_dir=args.output,
        checkpoint_dir=args.checkpoint_dir,
        task_type=args.task_type,
        k_features=args.k_features,
    )
