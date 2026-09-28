"""
Evaluation utilities for multiclass biomedical signal classification.
"""

from __future__ import annotations

from pathlib import Path
import json

import numpy as np

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def calculate_metrics(
    y_true,
    y_pred,
):
    """Calculate aggregate multiclass metrics."""

    return {
        "accuracy": float(
            accuracy_score(y_true, y_pred)
        ),
        "precision_macro": float(
            precision_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            )
        ),
        "recall_macro": float(
            recall_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            )
        ),
        "f1_macro": float(
            f1_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            )
        ),
        "f1_weighted": float(
            f1_score(
                y_true,
                y_pred,
                average="weighted",
                zero_division=0,
            )
        ),
    }


def get_confusion_matrix(
    y_true,
    y_pred,
    class_names,
):
    """Generate a confusion matrix in the explicit configured class order."""

    return confusion_matrix(
        y_true,
        y_pred,
        labels=list(class_names),
    )


def get_classification_report(
    y_true,
    y_pred,
    class_names,
):
    """Generate a classification report in the explicit class order."""

    return classification_report(
        y_true,
        y_pred,
        labels=list(class_names),
        target_names=list(class_names),
        zero_division=0,
    )


def evaluate_model(
    y_true,
    y_pred,
    class_names,
):
    """Complete evaluation pipeline."""

    results = calculate_metrics(
        y_true,
        y_pred,
    )

    results["confusion_matrix"] = get_confusion_matrix(
        y_true,
        y_pred,
        class_names,
    )

    results["classification_report"] = get_classification_report(
        y_true,
        y_pred,
        class_names,
    )

    results["class_order"] = list(class_names)

    return results


def save_evaluation_results(
    results,
    output_dir="outputs/evaluation",
):
    """Save evaluation outputs."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    serializable_metrics = {
        key: value
        for key, value in results.items()
        if key not in {
            "confusion_matrix",
            "classification_report",
        }
    }

    with open(
        output_path / "metrics.json",
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            serializable_metrics,
            handle,
            indent=4,
        )

    np.savetxt(
        output_path / "confusion_matrix.csv",
        results["confusion_matrix"],
        fmt="%d",
        delimiter=",",
    )

    with open(
        output_path / "classification_report.txt",
        "w",
        encoding="utf-8",
    ) as handle:
        handle.write(
            results["classification_report"]
        )
