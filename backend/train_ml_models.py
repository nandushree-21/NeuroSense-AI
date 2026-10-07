import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# ==========================================
# 1. Load feature dataset
# ==========================================

data = pd.read_csv("../dataset/features.csv")

print("Dataset loaded successfully!")
print("Shape:", data.shape)


# ==========================================
# 2. Separate features and target
# ==========================================

X = data.drop("label", axis=1)

y = data["label"]


# ==========================================
# 3. Split dataset
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ==========================================
# 4. Create models
# ==========================================

models = {

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced"
    ),

    "SVM": SVC(
        kernel="rbf",
        probability=True,
        class_weight="balanced"
    ),

    "XGBoost": XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        random_state=42
    )
}


# ==========================================
# 5. Train models
# ==========================================

results = {}


for name, model in models.items():

    print("\n================================")
    print("Training:", name)
    print("================================")

    model.fit(X_train, y_train)


    # Prediction
    predictions = model.predict(X_test)


    # Metrics
    accuracy = accuracy_score(
        y_test,
        predictions
    )

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


    # Store results
    results[name] = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


    # Display results
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


    # ======================================
    # Save model
    # ======================================

    filename = name.lower().replace(
        " ",
        "_"
    ) + ".pkl"


    joblib.dump(
        model,
        "../models/" + filename
    )


    print("Model saved:", filename)


# ==========================================
# 6. Final comparison
# ==========================================

print("\n\n========================================")
print("FINAL MODEL COMPARISON")
print("========================================")


for name, result in results.items():

    print("\n", name)

    print(
        "Accuracy:",
        round(result["accuracy"] * 100, 2),
        "%"
    )

    print(
        "Precision:",
        round(result["precision"] * 100, 2),
        "%"
    )

    print(
        "Recall:",
        round(result["recall"] * 100, 2),
        "%"
    )

    print(
        "F1 Score:",
        round(result["f1"] * 100, 2),
        "%"
    )