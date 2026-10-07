import os

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay
)

from tensorflow.keras.models import load_model


# =========================================================
# PROJECT PATHS
# =========================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATASET_PATH = os.path.join(
    BASE_DIR,
    "dataset",
    "features.csv"
)

PREPROCESSED_PATH = os.path.join(
    BASE_DIR,
    "dataset",
    "preprocessed_eeg.csv"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "frontend",
    "assets"
)


# Create output folder if it doesn't exist
os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


print("\n======================================")
print("       NEUROSENSE AI EVALUATION")
print("======================================")


# =========================================================
# 1. TRADITIONAL ML DATA
# =========================================================

print("\nLoading feature dataset...")

data = pd.read_csv(
    DATASET_PATH
)

X = data.drop(
    "label",
    axis=1
)

y = data["label"]


# Same split used during ML training
X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=42,

    stratify=y

)


print(
    "ML test samples:",
    len(X_test)
)


# =========================================================
# 2. TRADITIONAL ML MODELS
# =========================================================

ml_models = {

    "Random Forest":
        "random_forest.pkl",

    "SVM":
        "svm.pkl",

    "XGBoost":
        "xgboost.pkl"

}


for model_name, model_file in ml_models.items():

    print("\n--------------------------------------")
    print(
        "Creating confusion matrix:",
        model_name
    )
    print("--------------------------------------")


    model_path = os.path.join(
        MODEL_DIR,
        model_file
    )


    # Load model
    model = joblib.load(
        model_path
    )


    # Predict
    predictions = model.predict(
        X_test
    )


    # Confusion matrix
    cm = confusion_matrix(

        y_test,

        predictions

    )


    print("\nConfusion Matrix:")

    print(cm)


    # Display
    display = ConfusionMatrixDisplay(

        confusion_matrix=cm,

        display_labels=[
            "Normal",
            "Seizure"
        ]

    )


    display.plot()

    plt.title(
        model_name + " Confusion Matrix"
    )

    plt.tight_layout()


    # File name
    output_file = os.path.join(

        OUTPUT_DIR,

        model_name.lower()
        .replace(" ", "_")
        + "_confusion_matrix.png"

    )


    plt.savefig(
        output_file,
        dpi=150
    )

    plt.close()


    print(
        "\nSaved:",
        output_file
    )


# =========================================================
# 3. CNN-LSTM DATA
# =========================================================

print("\n======================================")
print("       CNN-LSTM CONFUSION MATRIX")
print("======================================")


X_test_dl = np.load(

    os.path.join(
        BASE_DIR,
        "dataset",
        "X_test.npy"
    )

)

y_test_dl = np.load(

    os.path.join(
        BASE_DIR,
        "dataset",
        "y_test.npy"
    )

)


print(
    "\nCNN-LSTM test samples:",
    len(X_test_dl)
)


# =========================================================
# 4. LOAD CNN-LSTM
# =========================================================

cnn_path = os.path.join(

    MODEL_DIR,

    "cnn_lstm.keras"

)


print(
    "\nLoading CNN-LSTM model..."
)


cnn_model = load_model(
    cnn_path
)


print(
    "CNN-LSTM loaded successfully!"
)


# =========================================================
# 5. CNN-LSTM PREDICTIONS
# =========================================================

probabilities = cnn_model.predict(

    X_test_dl,

    verbose=0

)


probabilities = probabilities.flatten()


predictions_dl = (

    probabilities >= 0.5

).astype(int)


# =========================================================
# 6. CNN-LSTM CONFUSION MATRIX
# =========================================================

cm = confusion_matrix(

    y_test_dl,

    predictions_dl

)


print("\nCNN-LSTM Confusion Matrix:")

print(cm)


display = ConfusionMatrixDisplay(

    confusion_matrix=cm,

    display_labels=[
        "Normal",
        "Seizure"
    ]

)


display.plot()


plt.title(
    "CNN-LSTM Confusion Matrix"
)

plt.tight_layout()


output_file = os.path.join(

    OUTPUT_DIR,

    "cnn_lstm_confusion_matrix.png"

)


plt.savefig(

    output_file,

    dpi=150

)


plt.close()


print(
    "\nSaved:",
    output_file
)


# =========================================================
# 7. FINISHED
# =========================================================

print("\n======================================")
print("       CONFUSION MATRICES COMPLETE")
print("======================================")

print(
    "\nImages saved in:"
)

print(
    OUTPUT_DIR
)