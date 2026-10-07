import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split


# ==========================================
# 1. Load preprocessed EEG data
# ==========================================

data = pd.read_csv(
    "../dataset/preprocessed_eeg.csv"
)

print("Dataset loaded!")
print("Shape:", data.shape)


# ==========================================
# 2. Separate EEG signals and labels
# ==========================================

X = data.drop(
    "label",
    axis=1
).values

y = data["label"].values


# ==========================================
# 3. Convert to float32
# ==========================================

X = X.astype("float32")

y = y.astype("int32")


# ==========================================
# 4. Add channel dimension
# ==========================================

# Before:
# (11500, 178)

# After:
# (11500, 178, 1)

X = np.expand_dims(
    X,
    axis=2
)


print("\nEEG data shape:")
print(X.shape)

print("Labels shape:")
print(y.shape)


# ==========================================
# 5. Train/Test split
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=42,

    stratify=y
)


print("\nTraining data:")
print(X_train.shape)

print("\nTesting data:")
print(X_test.shape)


# ==========================================
# 6. Save data
# ==========================================

np.save(
    "../dataset/X_train.npy",
    X_train
)

np.save(
    "../dataset/X_test.npy",
    X_test
)

np.save(
    "../dataset/y_train.npy",
    y_train
)

np.save(
    "../dataset/y_test.npy",
    y_test
)


print("\n================================")
print("Deep Learning data prepared!")
print("================================")

print("Files saved:")
print("X_train.npy")
print("X_test.npy")
print("y_train.npy")
print("y_test.npy")