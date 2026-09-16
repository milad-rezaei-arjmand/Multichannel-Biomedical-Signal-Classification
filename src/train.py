"""
Main training pipeline for
Multichannel Biomedical Signal Classification.

Pipeline:
1. Load dataset
2. Preprocess signals
3. Extract features
4. Feature selection
5. Train CatBoost ensemble
6. Save models
7. Evaluate model
"""


from pathlib import Path
import sys
import argparse
import pickle
import numpy as np


from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif


# Add project root to Python path

ROOT_DIR = Path(__file__).resolve().parent.parent

sys.path.append(
    str(ROOT_DIR)
)



from src.config import (
    DEFAULT_FS,
    CLASS_NAMES,
    RANDOM_STATE
)


from src.data_loader import (
    load_dataset
)


from src.preprocessing.signal_processing import (
    preprocess_dataset
)


from src.feature_extraction.build_features import (
    build_dataset_features
)


from src.models.catboost_classifier import (
    train_ensemble,
    predict_ensemble,
    save_models
)


from src.evaluation.metrics import (
    evaluate_model,
    save_evaluation_results
)





def select_features(
    X_train,
    X_test,
    y_train,
    k=1000
):
    """
    Feature selection using ANOVA F-test.
    """


    selector = SelectKBest(
        score_func=f_classif,
        k=min(
            k,
            X_train.shape[1]
        )
    )


    X_train_selected = selector.fit_transform(
        X_train,
        y_train
    )


    X_test_selected = selector.transform(
        X_test
    )


    return (
        X_train_selected,
        X_test_selected,
        selector
    )





def run_training(
    dataset_path,
    class_names,
    output_dir="results"
):
    """
    Complete training pipeline.
    """


    output_dir = Path(
        output_dir
    )


    checkpoint_dir = Path(
        "checkpoints"
    )


    output_dir.mkdir(
        exist_ok=True
    )


    checkpoint_dir.mkdir(
        exist_ok=True
    )



    # -----------------------
    # Load dataset
    # -----------------------

    print(
        "Loading dataset..."
    )


    signals, labels = load_dataset(
        dataset_path
    )


    print(
        "Raw signal shape:",
        signals.shape
    )



    # -----------------------
    # Preprocessing
    # -----------------------

    print(
        "Preprocessing signals..."
    )


    signals = preprocess_dataset(
        signals,
        fs=DEFAULT_FS
    )



    # -----------------------
    # Feature extraction
    # -----------------------

    print(
        "Extracting features..."
    )


    X_features, feature_names = (
        build_dataset_features(
            signals,
            fs=DEFAULT_FS
        )
    )


    print(
        "Feature matrix:",
        X_features.shape
    )


    y = np.asarray(
        labels
    )



    # -----------------------
    # Train/Test split
    # -----------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X_features,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y
    )



    # -----------------------
    # Validation split
    # -----------------------

    X_train, X_val, y_train, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y_train
    )



    # -----------------------
    # Feature selection
    # -----------------------

    print(
        "Selecting features..."
    )


    X_train, X_test, selector = (
        select_features(
            X_train,
            X_test,
            y_train
        )
    )


    X_val = selector.transform(
        X_val
    )


    print(
        "Selected feature count:",
        X_train.shape[1]
    )



    with open(
        checkpoint_dir / "feature_selector.pkl",
        "wb"
    ) as f:

        pickle.dump(
            selector,
            f
        )



    # -----------------------
    # Training
    # -----------------------

    print(
        "Training CatBoost ensemble..."
    )


    models = train_ensemble(
        X_train,
        y_train,
        X_val,
        y_val
    )


    save_models(
        models,
        checkpoint_dir
    )



    # -----------------------
    # Prediction
    # -----------------------

    print(
        "Predicting..."
    )


    predictions = predict_ensemble(
        models,
        X_test
    )



    # -----------------------
    # Evaluation
    # -----------------------

    print(
        "Evaluating..."
    )


    results = evaluate_model(
        y_test,
        predictions,
        class_names
    )


    save_evaluation_results(
        results,
        output_dir
    )


    return results





if __name__ == "__main__":


    parser = argparse.ArgumentParser(
        description=
        "Multichannel Biomedical Signal Classification"
    )


    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help=
        "Path to dataset file (.npz or .csv)"
    )


    parser.add_argument(
        "--output",
        type=str,
        default="results",
        help=
        "Output directory"
    )


    args = parser.parse_args()



    results = run_training(
        args.dataset,
        CLASS_NAMES,
        args.output
    )



    print(
        "\nTraining completed."
    )


    print(
        results
    )