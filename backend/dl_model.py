import os
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input,
    Conv1D,
    MaxPooling1D,
    LSTM,
    Dense,
    Dropout
)

from tensorflow.keras.callbacks import EarlyStopping

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# ==========================================
# PROJECT PATH
# ==========================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATASET_DIR = os.path.join(BASE_DIR, "dataset")
MODEL_DIR = os.path.join(BASE_DIR, "models")
ASSET_DIR = os.path.join(BASE_DIR, "frontend", "assets")


# ==========================================
# LOAD DATA
# ==========================================

X_train = np.load(
    os.path.join(DATASET_DIR, "X_train.npy")
)

X_test = np.load(
    os.path.join(DATASET_DIR, "X_test.npy")
)

y_train = np.load(
    os.path.join(DATASET_DIR, "y_train.npy")
)

y_test = np.load(
    os.path.join(DATASET_DIR, "y_test.npy")
)


print("======================================")
print("       NEUROSENSE AI CNN-LSTM")
print("======================================")

print("\nTraining data:")
print(X_train.shape)

print("\nTesting data:")
print(X_test.shape)


# ==========================================
# BUILD CNN-LSTM MODEL
# ==========================================

model = Sequential([

    Input(
        shape=(
            X_train.shape[1],
            X_train.shape[2]
        )
    ),

    Conv1D(
        filters=32,
        kernel_size=3,
        activation="relu"
    ),

    MaxPooling1D(
        pool_size=2
    ),

    Conv1D(
        filters=64,
        kernel_size=3,
        activation="relu"
    ),

    MaxPooling1D(
        pool_size=2
    ),

    LSTM(64),

    Dropout(0.3),

    Dense(
        32,
        activation="relu"
    ),

    Dense(
        1,
        activation="sigmoid"
    )
])


# ==========================================
# COMPILE MODEL
# ==========================================

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)


print("\nModel created successfully!")


# ==========================================
# EARLY STOPPING
# ==========================================

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)


# ==========================================
# TRAIN MODEL
# ==========================================

print("\n======================================")
print("          MODEL TRAINING")
print("======================================")

history = model.fit(

    X_train,
    y_train,

    epochs=20,

    batch_size=32,

    validation_split=0.2,

    callbacks=[
        early_stopping
    ]
)


# ==========================================
# EVALUATE MODEL
# ==========================================

loss, accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=0
)


probabilities = model.predict(
    X_test,
    verbose=0
)


predictions = (
    probabilities >= 0.5
).astype(int).flatten()


precision = precision_score(
    y_test,
    predictions,
    zero_division=0
)

recall = recall_score(
    y_test,
    predictions,
    zero_division=0
)

f1 = f1_score(
    y_test,
    predictions,
    zero_division=0
)


# ==========================================
# PRINT RESULTS
# ==========================================

print("\n======================================")
print("          CNN-LSTM RESULTS")
print("======================================")

print(
    "Accuracy:",
    round(accuracy * 100, 2),
    "%"
)

print(
    "Precision:",
    round(precision * 100, 2),
    "%"
)

print(
    "Recall:",
    round(recall * 100, 2),
    "%"
)

print(
    "F1 Score:",
    round(f1 * 100, 2),
    "%"
)


# ==========================================
# SAVE MODEL
# ==========================================

model_path = os.path.join(
    MODEL_DIR,
    "cnn_lstm.keras"
)

model.save(model_path)

print("\nCNN-LSTM model saved:")
print(model_path)


# ==========================================
# CREATE TRAINING ACCURACY GRAPH
# ==========================================

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.title(
    "CNN-LSTM Training and Validation Accuracy"
)

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.legend()

plt.grid(True)

accuracy_graph = os.path.join(
    ASSET_DIR,
    "training_validation_accuracy.png"
)

plt.savefig(
    accuracy_graph,
    dpi=150,
    bbox_inches="tight"
)

plt.close()


# ==========================================
# CREATE TRAINING LOSS GRAPH
# ==========================================

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.title(
    "CNN-LSTM Training and Validation Loss"
)

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.legend()

plt.grid(True)

loss_graph = os.path.join(
    ASSET_DIR,
    "training_validation_loss.png"
)

plt.savefig(
    loss_graph,
    dpi=150,
    bbox_inches="tight"
)

plt.close()


# ==========================================
# FINAL MESSAGE
# ==========================================

print("\n======================================")
print("       TRAINING GRAPHS SAVED")
print("======================================")

print("\nAccuracy graph:")
print(accuracy_graph)

print("\nLoss graph:")
print(loss_graph)

print("\n======================================")
print("       CNN-LSTM COMPLETE")
print("======================================")