import os
import joblib
import numpy as np
import pandas as pd

from backend.extract_features import extract_eeg_features


# =========================================================
# PROJECT PATH
# =========================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)


# =========================================================
# LOAD ML MODELS
# =========================================================

random_forest = joblib.load(
    os.path.join(
        MODEL_DIR,
        "random_forest.pkl"
    )
)

svm = joblib.load(
    os.path.join(
        MODEL_DIR,
        "svm.pkl"
    )
)

xgboost = joblib.load(
    os.path.join(
        MODEL_DIR,
        "xgboost.pkl"
    )
)


# =========================================================
# FEATURE ORDER
# =========================================================

FEATURE_NAMES = [
    "mean",
    "std",
    "variance",
    "minimum",
    "maximum",
    "skewness",
    "kurtosis",
    "energy",
    "rms",
    "zero_crossings"
]


# =========================================================
# EXTRACT FEATURES FOR ONE EEG SAMPLE
# =========================================================

def prepare_features(signal):

    feature_dict = extract_eeg_features(
        signal
    )

    features = pd.DataFrame(
        [[
            feature_dict["mean"],
            feature_dict["std"],
            feature_dict["variance"],
            feature_dict["minimum"],
            feature_dict["maximum"],
            feature_dict["skewness"],
            feature_dict["kurtosis"],
            feature_dict["energy"],
            feature_dict["rms"],
            feature_dict["zero_crossings"]
        ]],
        columns=FEATURE_NAMES
    )

    return features


# =========================================================
# GET PREDICTIONS FROM ALL ML MODELS
# =========================================================

def get_ml_predictions(features):

    models = {
        "Random Forest": random_forest,
        "SVM": svm,
        "XGBoost": xgboost
    }

    results = {}

    for name, model in models.items():

        probability = model.predict_proba(
            features
        )[0][1]

        if probability >= 0.5:

            prediction = (
                "Seizure-related pattern"
            )

        else:

            prediction = (
                "Normal pattern"
            )

        results[name] = {

            "probability": round(
                float(probability) * 100,
                2
            ),

            "prediction": prediction
        }

    return results


# =========================================================
# CALCULATE MODEL UNCERTAINTY
# =========================================================

def calculate_uncertainty(results):

    probabilities = [

        result["probability"]

        for result in results.values()

    ]

    mean_probability = np.mean(
        probabilities
    )

    probability_std = np.std(
        probabilities
    )

    minimum_probability = min(
        probabilities
    )

    maximum_probability = max(
        probabilities
    )

    probability_range = (
        maximum_probability
        - minimum_probability
    )

    # Engineering agreement score
    agreement_score = max(
        0,
        100 - (probability_std * 2)
    )

    # Model agreement level

    if probability_range <= 15:

        agreement_level = "High"

    elif probability_range <= 30:

        agreement_level = "Moderate"

    else:

        agreement_level = "Low"


    # Uncertainty level

    distance_from_threshold = abs(
        mean_probability - 50
    )

    if (
        agreement_level == "Low"
        or distance_from_threshold < 15
    ):

        uncertainty_level = "High"

    elif (
        agreement_level == "Moderate"
        or distance_from_threshold < 25
    ):

        uncertainty_level = "Moderate"

    else:

        uncertainty_level = "Low"


    return {

        "mean_probability": round(
            float(mean_probability),
            2
        ),

        "probability_std": round(
            float(probability_std),
            2
        ),

        "probability_range": round(
            float(probability_range),
            2
        ),

        "agreement_score": round(
            float(agreement_score),
            2
        ),

        "agreement_level":
            agreement_level,

        "uncertainty_level":
            uncertainty_level
    }


# =========================================================
# ADAPTIVE DECISION
# =========================================================

def adaptive_decision(signal):

    # Step 1:
    # Extract the 10 trained features

    features = prepare_features(
        signal
    )


    # Step 2:
    # Get predictions from RF, SVM and XGBoost

    results = get_ml_predictions(
        features
    )


    # Step 3:
    # Calculate uncertainty

    uncertainty = calculate_uncertainty(
        results
    )


    # Step 4:
    # Calculate final ML probability

    mean_probability = (
        uncertainty[
            "mean_probability"
        ]
    )


    # Step 5:
    # Final ML decision

    if mean_probability >= 50:

        majority_prediction = (
            "Seizure-related pattern"
        )

    else:

        majority_prediction = (
            "Normal pattern"
        )


    # Step 6:
    # Adaptive decision

    if (
        uncertainty[
            "uncertainty_level"
        ] == "High"
    ):

        decision = (
            "Needs deeper analysis"
        )

        action = (
            "Escalate to CNN-LSTM analysis"
        )

    else:

        decision = (
            majority_prediction
        )

        action = (
            "Accept adaptive ML screening result"
        )


    # Step 7:
    # Return complete result

    return {

        "model_predictions":
            results,

        "uncertainty":
            uncertainty,

        "decision":
            decision,

        "action":
            action
    }