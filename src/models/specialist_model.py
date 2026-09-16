"""
Specialist classifier for difficult class separation
in multichannel biomedical signal classification.

Used for binary classification between
two selected classes.
"""


from pathlib import Path

import numpy as np

from sklearn.metrics import (
    accuracy_score,
    f1_score
)

from .catboost_classifier import (
    create_binary_specialist
)



def prepare_binary_dataset(
    X,
    y,
    class_a,
    class_b
):
    """
    Prepare binary dataset from two classes.

    Mapping:
    class_a -> 0
    class_b -> 1
    """


    mask = np.isin(
        y,
        [
            class_a,
            class_b
        ]
    )


    X_binary = X[mask]

    y_binary = y[mask]


    y_binary = np.where(
        y_binary == class_a,
        0,
        1
    )


    return (
        X_binary,
        y_binary.astype(int)
    )



def train_specialist(
    X_train,
    y_train,
    X_val,
    y_val,
    class_a,
    class_b,
    task_type="CPU"
):
    """
    Train binary specialist model.

    Uses predefined train/validation split
    to avoid data leakage.
    """


    X_train_binary, y_train_binary = (
        prepare_binary_dataset(
            X_train,
            y_train,
            class_a,
            class_b
        )
    )


    X_val_binary, y_val_binary = (
        prepare_binary_dataset(
            X_val,
            y_val,
            class_a,
            class_b
        )
    )


    if len(
        np.unique(y_train_binary)
    ) < 2:

        raise ValueError(
            "Specialist requires two classes."
        )


    model = create_binary_specialist(
        task_type=task_type
    )


    model.fit(
        X_train_binary,
        y_train_binary,
        eval_set=(
            X_val_binary,
            y_val_binary
        )
    )


    prediction = model.predict(
        X_val_binary
    )


    metrics = {

        "accuracy": accuracy_score(
            y_val_binary,
            prediction
        ),

        "f1_score": f1_score(
            y_val_binary,
            prediction
        )
    }


    return model, metrics



def specialist_predict(
    model,
    X,
    threshold=0.65
):
    """
    Predict specialist class.

    Returns binary prediction.
    """


    probability = (
        model.predict_proba(X)[:, 1]
    )


    return (
        probability >= threshold
    ).astype(int)



def save_specialist_model(
    model,
    path="checkpoints/specialist_model.cbm"
):
    """
    Save specialist model.
    """


    output = Path(
        path
    )


    output.parent.mkdir(
        parents=True,
        exist_ok=True
    )


    model.save_model(
        output
    )
