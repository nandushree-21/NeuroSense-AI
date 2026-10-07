import pandas as pd

file_path = "../dataset/eeg_data.csv"

data = pd.read_csv(file_path)

print("========== DATASET INFORMATION ==========")

print("\nDataset Shape:")
print(data.shape)

print("\nFirst 5 Rows:")
print(data.head())

print("\nColumn Names:")
print(data.columns.tolist())

print("\nData Types:")
print(data.dtypes)

print("\nMissing Values:")
print(data.isnull().sum())

print("\nDuplicate Rows:")
print(data.duplicated().sum())