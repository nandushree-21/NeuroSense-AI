import pandas as pd

data = pd.read_csv("../dataset/cleaned_eeg.csv")

print("========== CLEANED DATASET ==========")

print("\nShape:")
print(data.shape)

print("\nFirst 5 rows:")
print(data.head())

print("\nMissing values:")
print(data.isnull().sum().sum())

print("\nClass distribution:")
print(data["label"].value_counts())

print("\nColumns:")
print(data.columns.tolist())