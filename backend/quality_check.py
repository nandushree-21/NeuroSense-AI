import pandas as pd
import numpy as np


def check_eeg_quality(file_path):

    print("\n======================================")
    print("        EEG SIGNAL QUALITY CHECK")
    print("======================================")

    # ------------------------------------------------
    # 1. Load EEG CSV file
    # ------------------------------------------------

    try:
        data = pd.read_csv(file_path)

    except FileNotFoundError:
        print("\n❌ ERROR: EEG file not found.")
        print("File path:", file_path)
        return False

    except Exception as e:
        print("\n❌ ERROR: Could not read EEG file.")
        print("Reason:", e)
        return False

    print("\nFile loaded successfully!")

    # ------------------------------------------------
    # 2. Expected EEG columns
    # ------------------------------------------------

    expected_columns = [
        f"X{i}" for i in range(1, 179)
    ]

    # ------------------------------------------------
    # 3. Check EEG columns
    # ------------------------------------------------

    missing_columns = [
        column
        for column in expected_columns
        if column not in data.columns
    ]

    if missing_columns:

        print("\n❌ QUALITY CHECK FAILED")

        print("\nMissing EEG columns:")

        print(missing_columns)

        print("\nExpected 178 EEG signal columns:")
        print("X1 to X178")

        return False

    print("\n✓ EEG columns detected")
    print("Number of EEG features:", len(expected_columns))

    # ------------------------------------------------
    # 4. Select EEG data
    # ------------------------------------------------

    eeg_data = data[expected_columns]

    # ------------------------------------------------
    # 5. Number of samples
    # ------------------------------------------------

    number_of_samples = len(eeg_data)

    print("\nNumber of EEG samples:", number_of_samples)

    if number_of_samples == 0:

        print("\n❌ QUALITY CHECK FAILED")

        print("Reason: No EEG samples found.")

        return False

    # ------------------------------------------------
    # 6. Check missing values
    # ------------------------------------------------

    missing_values = eeg_data.isnull().sum().sum()

    print("Missing values:", missing_values)

    if missing_values > 0:

        print("\n❌ QUALITY CHECK FAILED")

        print("Reason: Missing EEG values detected.")

        return False

    print("✓ No missing values")

    # ------------------------------------------------
    # 7. Convert values to numeric
    # ------------------------------------------------

    try:

        eeg_data = eeg_data.apply(
            pd.to_numeric,
            errors="raise"
        )

    except Exception as e:

        print("\n❌ QUALITY CHECK FAILED")

        print("Reason: EEG data contains non-numeric values.")

        print(e)

        return False

    print("✓ EEG values are numeric")

    # ------------------------------------------------
    # 8. Check infinite values
    # ------------------------------------------------

    infinite_values = np.isinf(
        eeg_data.values
    ).sum()

    print("Infinite values:", infinite_values)

    if infinite_values > 0:

        print("\n❌ QUALITY CHECK FAILED")

        print("Reason: Infinite values detected.")

        return False

    print("✓ No infinite values")

    # ------------------------------------------------
    # 9. Check constant samples
    # ------------------------------------------------

    constant_samples = 0

    for index, row in eeg_data.iterrows():

        if row.std() == 0:

            constant_samples += 1

    print("Constant samples:", constant_samples)

    if constant_samples > 0:

        print("\n⚠ WARNING")

        print(
            "Some EEG samples contain no signal variation."
        )

    else:

        print("✓ Signal variation detected")

    # ------------------------------------------------
    # 10. Signal statistics
    # ------------------------------------------------

    minimum_value = eeg_data.min().min()

    maximum_value = eeg_data.max().max()

    average_value = eeg_data.values.mean()

    standard_deviation = eeg_data.values.std()

    print("\n--------------------------------------")
    print("        SIGNAL STATISTICS")
    print("--------------------------------------")

    print(
        "Minimum signal value:",
        round(minimum_value, 4)
    )

    print(
        "Maximum signal value:",
        round(maximum_value, 4)
    )

    print(
        "Average signal value:",
        round(average_value, 4)
    )

    print(
        "Signal standard deviation:",
        round(standard_deviation, 4)
    )

    # ------------------------------------------------
    # 11. Final quality result
    # ------------------------------------------------

    print("\n======================================")

    if constant_samples == number_of_samples:

        print("❌ EEG QUALITY: POOR")

        print(
            "All samples contain constant values."
        )

        print("======================================")

        return False

    else:

        print("✅ EEG QUALITY: GOOD")

        print(
            "EEG file passed the basic quality checks."
        )

        print(
            "The file can be sent for AI analysis."
        )

        print("======================================")

        return True


# ====================================================
# TEST THE EEG FILE
# ====================================================

if __name__ == "__main__":

    # Since this script is being run from the
    # NeuroSenseAI project root using:
    #
    # python backend/quality_check.py
    #
    # we use this path:

    file_path = "dataset/test_eeg.csv"

    result = check_eeg_quality(file_path)

    print("\nFinal result:", result)