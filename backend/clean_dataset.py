import pandas as pd

# Load dataset
data = pd.read_csv("../dataset/eeg_data.csv")

print("Original shape:")
print(data.shape)


# Remove unnecessary index column
data = data.drop("Unnamed", axis=1)


# Convert 5 classes into 2 classes
# 1 = Seizure
# 2,3,4,5 = Normal

data["label"] = data["y"].apply(
    lambda x: 1 if x == 1 else 0
)


# Remove original y column
data = data.drop("y", axis=1)


# Save cleaned dataset
data.to_csv(
    "../dataset/cleaned_eeg.csv",
    index=False
)


print("\nCleaned shape:")
print(data.shape)


print("\nNew columns:")
print(data.columns.tolist())


print("\nClass distribution:")
print(data["label"].value_counts())


print("\n0 = Normal")
print("1 = Seizure")


print("\nCleaned dataset saved successfully!")