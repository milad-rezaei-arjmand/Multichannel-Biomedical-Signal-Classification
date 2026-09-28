"""
CatBoost ensemble utilities for multichannel biomedical signal classification.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from catboost import CatBoostClassifier


def create_multiclass_model(
    seed=42,
    task_type="CPU",
    iterations=8000,
    learning_rate=0.03,
    depth=8,
    l2_leaf_reg=6,
    od_wait=800,
):
    """Create a multiclass CatBoost classifier."""

    task_type = str(task_type).upper()

    if task_type not in {"CPU", "GPU"}:
        raise ValueError("task_type must be 'CPU' or 'GPU'.")

    params = {
        "loss_function": "MultiClass",
        "eval_metric": "MultiClass",
        "iterations": iterations,
        "learning_rate": learning_rate,
        "depth": depth,
        "l2_leaf_reg": l2_leaf_reg,
        "random_strength": 1.0,
        "auto_class_weights": "Balanced",
        "od_type": "Iter",
        "od_wait": od_wait,
        "use_best_model": True,
        "random_seed": seed,
        "verbose": 200,
        "thread_count": -1,
        "task_type": task_type,
        "allow_writing_files": False,
    }

    if task_type == "GPU":
        params["devices"] = "0"

    return CatBoostClassifier(**params)


def create_binary_specialist(
    seed=2026,
    task_type="CPU",
    iterations=6000,
    learning_rate=0.03,
    depth=6,
    l2_leaf_reg=6,
    od_wait=600,
):
    """Create an optional binary specialist CatBoost classifier."""

    task_type = str(task_type).upper()

    if task_type not in {"CPU", "GPU"}:
        raise ValueError("task_type must be 'CPU' or 'GPU'.")

    params = {
        "loss_function": "Logloss",
        "eval_metric": "Logloss",
        "iterations": iterations,
        "learning_rate": learning_rate,
        "depth": depth,
        "l2_leaf_reg": l2_leaf_reg,
        "random_strength": 1.0,
        "od_type": "Iter",
        "od_wait": od_wait,
        "use_best_model": True,
        "random_seed": seed,
        "verbose": 200,
        "thread_count": -1,
        "task_type": task_type,
        "allow_writing_files": False,
    }

    if task_type == "GPU":
        params["devices"] = "0"

    return CatBoostClassifier(**params)


def train_ensemble(
    X_train,
    y_train,
    X_val,
    y_val,
    task_type="CPU",
):
    """Train the three-model CatBoost ensemble."""

    models = []

    configurations = [
        (42, 8),
        (49, 7),
        (55, 8),
    ]

    for seed, depth in configurations:
        model = create_multiclass_model(
            seed=seed,
            depth=depth,
            task_type=task_type,
        )

        model.fit(
            X_train,
            y_train,
            eval_set=(X_val, y_val),
        )

        models.append(model)

    return models


def _validate_ensemble_classes(models):
    if not models:
        raise ValueError("No trained models found.")

    reference = np.asarray(models[0].classes_)

    for model_index, model in enumerate(models[1:], start=1):
        current = np.asarray(model.classes_)

        if not np.array_equal(current, reference):
            raise ValueError(
                "Ensemble models use inconsistent class ordering: "
                f"model 0={reference.tolist()}, "
                f"model {model_index}={current.tolist()}."
            )

    return reference


def predict_average_probability(models, X):
    """Average class probabilities across models."""

    classes = _validate_ensemble_classes(models)

    probability = np.zeros(
        (X.shape[0], len(classes)),
        dtype=np.float64,
    )

    for model in models:
        probability += np.asarray(
            model.predict_proba(X),
            dtype=np.float64,
        )

    return probability / len(models)


def predict_ensemble(
    models,
    X,
    return_probability=False,
):
    """Predict original class labels using averaged probabilities."""

    classes = _validate_ensemble_classes(models)

    probability = predict_average_probability(
        models,
        X,
    )

    prediction_index = np.argmax(
        probability,
        axis=1,
    )

    prediction = classes[prediction_index]

    if return_probability:
        return prediction, probability

    return prediction


def save_models(
    models,
    output_dir="outputs/checkpoints",
):
    """Save trained CatBoost models."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    for index, model in enumerate(models):
        model.save_model(
            output / f"catboost_model_{index}.cbm"
        )


def get_feature_importance(
    model,
    feature_names,
):
    """Map feature names to model feature importances."""

    importance = model.get_feature_importance()

    return dict(zip(feature_names, importance))
