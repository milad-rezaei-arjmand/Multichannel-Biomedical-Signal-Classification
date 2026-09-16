"""
Prediction pipeline for
Multichannel Biomedical Signal Classification.
"""


from pathlib import Path
import sys
import argparse
import pickle
import numpy as np


ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))


from src.config import (
    DEFAULT_FS,
    CLASS_NAMES
)


from src.preprocessing.signal_processing import (
    preprocess_multichannel_signal
)


from src.feature_extraction.build_features import (
    build_feature_vector
)


from src.models.catboost_classifier import (
    predict_average_probability
)

from catboost import CatBoostClassifier





def load_models(
    model_dir="checkpoints"
):

    models = []


    for i in range(3):

        model = CatBoostClassifier()

        model.load_model(
            Path(model_dir) /
            f"catboost_model_{i}.cbm"
        )

        models.append(
            model
        )


    return models





def load_selector(
    path="checkpoints/feature_selector.pkl"
):

    with open(
        path,
        "rb"
    ) as f:

        return pickle.load(f)





def predict(
    input_path
):

    signal = np.load(
        input_path
    )


    print(
        "Input shape:",
        signal.shape
    )


    signal = preprocess_multichannel_signal(
        signal,
        fs=DEFAULT_FS
    )


    features, _ = build_feature_vector(
        signal,
        fs=DEFAULT_FS
    )


    features = features.reshape(
        1,
        -1
    )


    selector = load_selector()


    features = selector.transform(
        features
    )


    models = load_models()


    probability = predict_average_probability(
        models,
        features
    )


    prediction_index = np.argmax(
        probability,
        axis=1
    )[0]


    prediction = CLASS_NAMES[
        prediction_index
    ]


    confidence = probability[0][
        prediction_index
    ]


    return prediction, confidence





if __name__ == "__main__":


    parser = argparse.ArgumentParser(
        description=
        "Biomedical Signal Prediction"
    )


    parser.add_argument(
        "--input",
        required=True,
        help=
        "Path to signal numpy file"
    )


    args = parser.parse_args()


    prediction, confidence = predict(
        args.input
    )


    print(
        "\nPrediction:",
        prediction
    )


    print(
        "Confidence:",
        round(
            float(confidence)*100,
            2
        ),
        "%"
    )
