import os
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis


# =========================================================
# PROJECT PATHS
# =========================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATASET_DIR = os.path.join(
    BASE_DIR,
    "dataset"
)

PREPROCESSED_FILE = os.path.join(
    DATASET_DIR,
    "preprocessed_eeg.csv"
)

FEATURE_FILE = os.path.join(
    DATASET_DIR,
    "features.csv"
)


# =========================================================
# FUNCTION 1: Extract features from ONE EEG signal
# =========================================================

def extract_eeg_features(signal):

    signal = np.asarray(
        signal,
        dtype=float
    )

    feature = {
        "mean": np.mean(signal),

        "std": np.std(signal),

        "variance": np.var(signal),

        "minimum": np.min(signal),

        "maximum": np.max(signal),

        "skewness": skew(signal),

        "kurtosis": kurtosis(signal),

        "energy": np.sum(signal ** 2),

        "rms": np.sqrt(
            np.mean(signal ** 2)
        ),

        "zero_crossings": np.sum(
            np.diff(np.sign(signal)) != 0
        )
    }

    return feature


# =========================================================
# FUNCTION 2: Create feature dataset
# =========================================================

def create_feature_dataset():

    print("Loading preprocessed EEG dataset...")

    data = pd.read_csv(
        PREPROCESSED_FILE
    )

    print(
        "Dataset shape:",
        data.shape
    )

    X = data.drop(
        "label",
        axis=1
    )

    y = data["label"]

    features = []

    print(
        "\nExtracting EEG features..."
    )

    for _, row in X.iterrows():

        signal = row.values.astype(
            float
        )

        feature = extract_eeg_features(
            signal
        )

        features.append(feature)

    features_df = pd.DataFrame(
        features
    )

    features_df["label"] = y.values

    # Save feature dataset
    features_df.to_csv(
        FEATURE_FILE,
        index=False
    )

    print(
        "\n================================"
    )

    print(
        "Feature extraction completed!"
    )

    print(
        "================================"
    )

    print(
        "\nFeature dataset shape:"
    )

    print(
        features_df.shape
    )

    print(
        "\nFirst 5 rows:"
    )

    print(
        features_df.head()
    )

    print(
        "\nClass distribution:"
    )

    print(
        features_df["label"].value_counts()
    )

    print(
        "\nFeatures:"
    )

    print(
        features_df.columns.tolist()
    )

    print(
        "\nSaved successfully:"
    )

    print(
        FEATURE_FILE
    )


# =========================================================
# RUN PROGRAM
# =========================================================

if __name__ == "__main__":

    create_feature_dataset()