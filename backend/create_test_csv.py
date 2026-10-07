import pandas as pd

# Load preprocessed EEG dataset
data = pd.read_csv("../dataset/preprocessed_eeg.csv")

# Take the first EEG sample
test_sample = data.iloc[[0]]

# Remove label
test_sample = test_sample.drop("label", axis=1)

# Save test sample
test_sample.to_csv(
    "../dataset/test_eeg.csv",
    index=False
)

print("Test EEG CSV created successfully!")
print("Shape:", test_sample.shape)
print("Saved as: ../dataset/test_eeg.csv")