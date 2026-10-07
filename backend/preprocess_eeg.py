import pandas as pd
import numpy as np

# Load cleaned dataset
data = pd.read_csv("../dataset/cleaned_eeg.csv")

print("Original dataset shape:")
print(data.shape)


# Separate EEG signals and labels
X = data.drop("label", axis=1).values
y = data["label"].values


# Convert EEG values to float
X = X.astype("float32")


# Normalize each EEG sample
mean = X.mean(axis=1, keepdims=True)
std = X.std(axis=1, keepdims=True)

X_normalized = (X - mean) / (std + 1e-8)


# Create dataframe
X_normalized = pd.DataFrame(
    X_normalized,
    columns=[f"X{i}" for i in range(1, 179)]
)


# Add label
X_normalized["label"] = y


# Save preprocessed dataset
X_normalized.to_csv(
    "../dataset/preprocessed_eeg.csv",
    index=False
)


print("\nPreprocessing completed!")

print("Preprocessed shape:")
print(X_normalized.shape)

print("\nFirst 5 rows:")
print(X_normalized.head())

print("\nClass distribution:")
print(X_normalized["label"].value_counts())

print("\nSaved as:")
print("../dataset/preprocessed_eeg.csv")