import os

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score

from tensorflow.keras.models import load_model


# =========================================================
# PROJECT PATHS
# =========================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

FEATURES_PATH = os.path.join(
    BASE_DIR,
    "dataset",
    "features.csv"
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

# Create output folder
os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


print("\n======================================")
print("       NEUROSENSE AI ROC / AUC")
print("======================================")


# =========================================================
# 1. LOAD FEATURE DATA
# =========================================================

print("\nLoading feature dataset...")

data = pd.read_csv(
    FEATURES_PATH
)

X = data.drop(
    "label",
    axis=1
)

y = data["label"]


# Use the SAME split as model training
X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=42,

    stratify=y

)


print(
    "Testing samples:",
    len(X_test)
)


# =========================================================
# 2. STORE ROC RESULTS
# =========================================================

roc_results = {}


# =========================================================
# 3. RANDOM FOREST
# =========================================================

print("\n--------------------------------------")
print("Evaluating Random Forest")
print("--------------------------------------")

rf_path = os.path.join(
    MODEL_DIR,
    "random_forest.pkl"
)

rf_model = joblib.load(
    rf_path
)

rf_probability = rf_model.predict_proba(
    X_test
)[:, 1]

rf_auc = roc_auc_score(
    y_test,
    rf_probability
)

rf_fpr, rf_tpr, _ = roc_curve(
    y_test,
    rf_probability
)

roc_results["Random Forest"] = {
    "fpr": rf_fpr,
    "tpr": rf_tpr,
    "auc": rf_auc
}

print(
    "Random Forest AUC:",
    round(rf_auc, 4)
)


# =========================================================
# 4. SVM
# =========================================================

print("\n--------------------------------------")
print("Evaluating SVM")
print("--------------------------------------")

svm_path = os.path.join(
    MODEL_DIR,
    "svm.pkl"
)

svm_model = joblib.load(
    svm_path
)

svm_probability = svm_model.predict_proba(
    X_test
)[:, 1]

svm_auc = roc_auc_score(
    y_test,
    svm_probability
)

svm_fpr, svm_tpr, _ = roc_curve(
    y_test,
    svm_probability
)

roc_results["SVM"] = {
    "fpr": svm_fpr,
    "tpr": svm_tpr,
    "auc": svm_auc
}

print(
    "SVM AUC:",
    round(svm_auc, 4)
)


# =========================================================
# 5. XGBOOST
# =========================================================

print("\n--------------------------------------")
print("Evaluating XGBoost")
print("--------------------------------------")

xgb_path = os.path.join(
    MODEL_DIR,
    "xgboost.pkl"
)

xgb_model = joblib.load(
    xgb_path
)

xgb_probability = xgb_model.predict_proba(
    X_test
)[:, 1]

xgb_auc = roc_auc_score(
    y_test,
    xgb_probability
)

xgb_fpr, xgb_tpr, _ = roc_curve(
    y_test,
    xgb_probability
)

roc_results["XGBoost"] = {
    "fpr": xgb_fpr,
    "tpr": xgb_tpr,
    "auc": xgb_auc
}

print(
    "XGBoost AUC:",
    round(xgb_auc, 4)
)


# =========================================================
# 6. CNN-LSTM
# =========================================================

print("\n======================================")
print("Evaluating CNN-LSTM")
print("======================================")


# Load deep learning test data
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
    "CNN-LSTM test samples:",
    len(X_test_dl)
)


# Load CNN-LSTM
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


# Predict probabilities
cnn_probability = cnn_model.predict(
    X_test_dl,
    verbose=0
).flatten()


# Calculate AUC
cnn_auc = roc_auc_score(
    y_test_dl,
    cnn_probability
)


# Calculate ROC
cnn_fpr, cnn_tpr, _ = roc_curve(
    y_test_dl,
    cnn_probability
)


roc_results["CNN-LSTM"] = {
    "fpr": cnn_fpr,
    "tpr": cnn_tpr,
    "auc": cnn_auc
}


print(
    "CNN-LSTM AUC:",
    round(cnn_auc, 4)
)


# =========================================================
# 7. CREATE ROC CURVE
# =========================================================

print("\n======================================")
print("Creating ROC Curve")
print("======================================")


plt.figure(
    figsize=(10, 7)
)


for model_name, result in roc_results.items():

    plt.plot(

        result["fpr"],

        result["tpr"],

        linewidth=2,

        label=(
            model_name
            + " (AUC = "
            + str(round(result["auc"], 4))
            + ")"
        )

    )


# Random classifier line
plt.plot(

    [0, 1],

    [0, 1],

    linestyle="--",

    label="Random Classifier"

)


plt.xlabel(
    "False Positive Rate"
)

plt.ylabel(
    "True Positive Rate"
)

plt.title(
    "ROC Curve - NeuroSense AI"
)

plt.legend()

plt.grid(
    True
)

plt.tight_layout()


# =========================================================
# 8. SAVE ROC IMAGE
# =========================================================

roc_image_path = os.path.join(

    OUTPUT_DIR,

    "roc_curve.png"

)


plt.savefig(

    roc_image_path,

    dpi=150

)


plt.close()


print(
    "\nROC curve saved:"
)

print(
    roc_image_path
)


# =========================================================
# 9. PRINT FINAL AUC RESULTS
# =========================================================

print("\n======================================")
print("             AUC RESULTS")
print("======================================")


for model_name, result in roc_results.items():

    print(

        model_name
        + ": "
        + str(round(result["auc"], 4))

    )


print("\n======================================")
print("       ROC / AUC COMPLETE")
print("======================================")