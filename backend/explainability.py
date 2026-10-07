import numpy as np


def explain_eeg_signal(signal):
    """
    Simple explainability analysis for NeuroSense AI.

    The signal is divided into several regions.
    Each region is analyzed using its signal energy.
    """

    signal = np.asarray(signal, dtype=float)

    # Make sure signal is one-dimensional
    signal = signal.flatten()

    if len(signal) == 0:
        return {
            "status": "error",
            "message": "No EEG signal found."
        }

    # --------------------------------------
    # Divide EEG signal into 6 regions
    # --------------------------------------

    number_of_regions = 6

    regions = np.array_split(
        signal,
        number_of_regions
    )

    region_results = []

    # --------------------------------------
    # Calculate energy for each region
    # --------------------------------------

    for index, region in enumerate(regions):

        energy = np.mean(
            np.square(region)
        )

        standard_deviation = np.std(region)

        region_results.append({
            "region": index + 1,
            "energy": float(energy),
            "standard_deviation": float(
                standard_deviation
            )
        })

    # --------------------------------------
    # Find most active region
    # --------------------------------------

    highest_region = max(
        region_results,
        key=lambda x: x["energy"]
    )

    # --------------------------------------
    # Calculate overall signal statistics
    # --------------------------------------

    mean_value = np.mean(signal)

    std_value = np.std(signal)

    minimum_value = np.min(signal)

    maximum_value = np.max(signal)

    signal_energy = np.mean(
        np.square(signal)
    )

    # --------------------------------------
    # Create simple explanation
    # --------------------------------------

    explanation = (
        f"The EEG signal was divided into "
        f"{number_of_regions} regions for analysis. "
        f"Region {highest_region['region']} showed "
        f"the highest relative signal energy. "
        f"The overall signal standard deviation was "
        f"{std_value:.4f}, indicating the amount of "
        f"variation present in the analyzed EEG sample."
    )

    return {
        "status": "success",

        "signal_length": int(
            len(signal)
        ),

        "mean": round(
            float(mean_value),
            4
        ),

        "standard_deviation": round(
            float(std_value),
            4
        ),

        "minimum": round(
            float(minimum_value),
            4
        ),

        "maximum": round(
            float(maximum_value),
            4
        ),

        "signal_energy": round(
            float(signal_energy),
            4
        ),

        "most_active_region": int(
            highest_region["region"]
        ),

        "region_analysis": region_results,

        "explanation": explanation
    }


# --------------------------------------
# Test the module
# --------------------------------------

if __name__ == "__main__":

    print("\n======================================")
    print("      NEUROSENSE AI - XAI TEST")
    print("======================================")

    # Example EEG signal
    test_signal = np.random.randn(178)

    result = explain_eeg_signal(
        test_signal
    )

    print("\nXAI Result:")

    print(
        "Signal length:",
        result["signal_length"]
    )

    print(
        "Mean:",
        result["mean"]
    )

    print(
        "Standard deviation:",
        result["standard_deviation"]
    )

    print(
        "Minimum:",
        result["minimum"]
    )

    print(
        "Maximum:",
        result["maximum"]
    )

    print(
        "Signal energy:",
        result["signal_energy"]
    )

    print(
        "Most active region:",
        result["most_active_region"]
    )

    print(
        "\nExplanation:"
    )

    print(
        result["explanation"]
    )

    print("\n======================================")
    print("             XAI COMPLETE")
    print("======================================")