"""
CatBoost models for multichannel biomedical
signal classification.

Includes:
- Multiclass CatBoost classifier
- Binary specialist classifier
- Ensemble probability averaging
- Model saving/loading utilities
"""


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
    od_wait=800
):
    """
    Create CatBoost multiclass classifier.
    """


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

        "task_type": task_type
    }


    if task_type == "GPU":
        params["devices"] = "0"


    return CatBoostClassifier(
        **params
    )

def create_binary_specialist(
    seed=2026,
    task_type="CPU",
    iterations=6000,
    learning_rate=0.03,
    depth=6,
    l2_leaf_reg=6,
    od_wait=600
):
    """
    Create CatBoost binary specialist classifier.
    """


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

        "task_type": task_type
    }


    if task_type == "GPU":
        params["devices"] = "0"


    return CatBoostClassifier(
        **params
    )


def train_ensemble(
    X_train,
    y_train,
    X_val,
    y_val,
    task_type="CPU"
):
    """
    Train CatBoost ensemble.
    """


    models = []


    configurations = [

        (42,8),
        (49,7),
        (55,8)

    ]


    for seed, depth in configurations:


        model = create_multiclass_model(
            seed=seed,
            depth=depth,
            task_type=task_type
        )


        model.fit(
            X_train,
            y_train,
            eval_set=(
                X_val,
                y_val
            )
        )


        models.append(
            model
        )


    return models



def predict_average_probability(
    models,
    X
):
    """
    Average ensemble probabilities.
    """


    if len(models) == 0:
        raise ValueError(
            "No trained models found."
        )


    probability = np.zeros(
        (
            X.shape[0],
            models[0].classes_.shape[0]
        )
    )


    for model in models:

        probability += model.predict_proba(
            X
        )


    return probability / len(models)



def predict_ensemble(
    models,
    X,
    return_probability=False
):
    """
    Generate ensemble prediction.

    Returns original class labels.
    """


    probability = (
        predict_average_probability(
            models,
            X
        )
    )


    prediction_index = np.argmax(
        probability,
        axis=1
    )


    # Convert class index back to original labels
    classes = models[0].classes_

    prediction = classes[
        prediction_index
    ]


    if return_probability:

        return prediction, probability


    return prediction


def save_models(
    models,
    output_dir="checkpoints"
):
    """
    Save trained CatBoost models.
    """


    output = Path(
        output_dir
    )

    output.mkdir(
        parents=True,
        exist_ok=True
    )


    for i, model in enumerate(models):

        model.save_model(
            output / f"catboost_model_{i}.cbm"
        )



def get_feature_importance(
    model,
    feature_names
):
    """
    Extract feature importance.
    """


    importance = model.get_feature_importance()


    return dict(
        zip(
            feature_names,
            importance
        )
    )
