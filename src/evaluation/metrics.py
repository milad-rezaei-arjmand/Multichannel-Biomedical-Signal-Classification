"""
Evaluation utilities for biomedical
signal classification models.

Includes:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix
- Classification report
- Result saving
"""


from pathlib import Path
import json

import numpy as np

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score
)



def calculate_accuracy(
    y_true,
    y_pred
):
    """
    Calculate accuracy.
    """

    return float(
        accuracy_score(
            y_true,
            y_pred
        )
    )



def calculate_metrics(
    y_true,
    y_pred
):
    """
    Calculate main classification metrics.
    """

    return {

        "accuracy": float(
            accuracy_score(
                y_true,
                y_pred
            )
        ),

        "precision_macro": float(
            precision_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0
            )
        ),

        "recall_macro": float(
            recall_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0
            )
        ),

        "f1_macro": float(
            f1_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0
            )
        ),

        "f1_weighted": float(
            f1_score(
                y_true,
                y_pred,
                average="weighted",
                zero_division=0
            )
        )
    }



def get_confusion_matrix(
    y_true,
    y_pred
):
    """
    Generate confusion matrix.
    """

    return confusion_matrix(
        y_true,
        y_pred
    )



def get_classification_report(
    y_true,
    y_pred,
    class_names
):
    """
    Generate classification report.
    """

    return classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        zero_division=0
    )



def evaluate_model(
    y_true,
    y_pred,
    class_names
):
    """
    Complete evaluation pipeline.
    """


    results = {}


    results.update(
        calculate_metrics(
            y_true,
            y_pred
        )
    )


    results["confusion_matrix"] = (
        get_confusion_matrix(
            y_true,
            y_pred
        )
    )


    results["classification_report"] = (
        get_classification_report(
            y_true,
            y_pred,
            class_names
        )
    )


    return results



def save_evaluation_results(
    results,
    output_dir="results"
):
    """
    Save evaluation outputs.
    """


    output_path = Path(
        output_dir
    )


    output_path.mkdir(
        parents=True,
        exist_ok=True
    )


    with open(
        output_path / "metrics.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            {
                k: v
                for k, v in results.items()
                if k != "confusion_matrix"
                and k != "classification_report"
            },
            f,
            indent=4
        )



    np.savetxt(
        output_path / "confusion_matrix.csv",
        results["confusion_matrix"],
        fmt="%d",
        delimiter=","
    )



    with open(
        output_path / "classification_report.txt",
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            results["classification_report"]
        )
