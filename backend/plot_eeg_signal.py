import pandas as pd
import matplotlib.pyplot as plt
import os

# EEG dataset
file_path = "dataset/eeg_data.csv"

# Load dataset
df = pd.read_csv(file_path)

# EEG columns X1 to X178
eeg_columns = [f"X{i}" for i in range(1, 179)]

# Take first EEG sample
eeg_signal = df.loc[0, eeg_columns].astype(float)

# Create assets folder if it does not exist
os.makedirs("frontend/assets", exist_ok=True)

# Plot EEG signal
plt.figure(figsize=(12, 5))

plt.plot(
    range(1, 179),
    eeg_signal,
    linewidth=1.2
)

plt.title("EEG Signal Waveform")
plt.xlabel("Sample Number")
plt.ylabel("EEG Amplitude")

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

# Save image
output_file = "frontend/assets/eeg_signal.png"

plt.savefig(
    output_file,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("EEG graph created successfully!")
print("Saved at:", output_file)