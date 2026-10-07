# ============================================================
# NeuroSense AI
# AI-Driven EEG Signal Analysis for Automated Epileptic
# Seizure Detection
#
# Backend: Flask + TensorFlow + ML + Adaptive Analysis
# ============================================================

import os
import sys
import tempfile
import traceback

import numpy as np
import pandas as pd

from flask import Flask, request, jsonify
from flask_cors import CORS

# ============================================================
# PATH SETUP
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

BACKEND_DIR = os.path.join(
    BASE_DIR,
    "backend"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

EXPECTED_COLUMNS = [
    f"X{i}"
    for i in range(1, 179)
]


# ============================================================
# FLASK APP
# ============================================================

app = Flask(__name__)

CORS(app)


# ============================================================
# TENSORFLOW
# ============================================================

CNN_LSTM_MODEL = None


try:

    import tensorflow as tf

    # Reduce TensorFlow memory usage on CPU
    try:
        tf.config.threading.set_inter_op_parallelism_threads(2)
        tf.config.threading.set_intra_op_parallelism_threads(2)
    except Exception:
        pass

    model_path = os.path.join(
        MODEL_DIR,
        "cnn_lstm.keras"
    )

    if os.path.exists(model_path):

        CNN_LSTM_MODEL = tf.keras.models.load_model(
            model_path,
            compile=False
        )

        print(
            "CNN-LSTM model loaded successfully."
        )

    else:

        print(
            "WARNING: CNN-LSTM model not found:"
        )

        print(model_path)

except Exception as e:

    print(
        "WARNING: TensorFlow model could not be loaded."
    )

    print(
        "Reason:",
        str(e)
    )

    CNN_LSTM_MODEL = None


# ============================================================
# ADAPTIVE ML ENGINE
# ============================================================

try:

    from backend.adaptive_engine import adaptive_decision

    ADAPTIVE_ENGINE_LOADED = True

    print(
        "Adaptive ML Engine: Loaded"
    )

except Exception as e:

    print(
        "WARNING: Adaptive ML Engine could not be loaded."
    )

    print(
        "Reason:",
        str(e)
    )

    ADAPTIVE_ENGINE_LOADED = False


# ============================================================
# EXPLAINABILITY
# ============================================================

try:

    from backend.explainability import explain_eeg_signal

    XAI_LOADED = True

    print(
        "Explainable AI module: Loaded"
    )

except Exception as e:

    print(
        "WARNING: Explainability module could not be loaded."
    )

    print(
        "Reason:",
        str(e)
    )

    XAI_LOADED = False


# ============================================================
# GENAI
# ============================================================

try:

    from backend.genai import generate_explanation

    GENAI_LOADED = True

except Exception:

    GENAI_LOADED = False


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def normalize_signal(signal):
    """
    Row-wise normalization.

    This is used because the exact EEG sampling rate and
    physical amplitude calibration are not assumed here.
    """

    signal = np.asarray(
        signal,
        dtype=np.float32
    )

    minimum = np.min(signal)
    maximum = np.max(signal)

    if maximum - minimum == 0:

        return np.zeros_like(
            signal,
            dtype=np.float32
        )

    normalized = (
        signal - minimum
    ) / (
        maximum - minimum
    )

    return normalized.astype(
        np.float32
    )


def clean_dataframe(df):
    """
    Remove unnecessary columns and convert EEG columns
    to numeric values.
    """

    df = df.copy()

    # Remove common unwanted columns
    columns_to_remove = [
        "Unnamed",
        "Unnamed: 0",
        "y",
        "label"
    ]

    for column in columns_to_remove:

        if column in df.columns:

            # Keep label only if necessary
            if column == "label":
                continue

            df = df.drop(
                columns=[column]
            )


    # Make sure all expected EEG columns exist
    missing_columns = [
        column
        for column in EXPECTED_COLUMNS
        if column not in df.columns
    ]


    if missing_columns:

        raise ValueError(
            "Missing EEG columns: "
            + ", ".join(
                missing_columns[:10]
            )
        )


    # Select only X1-X178
    eeg_df = df[
        EXPECTED_COLUMNS
    ].copy()


    # Convert values to numbers
    for column in EXPECTED_COLUMNS:

        eeg_df[column] = pd.to_numeric(
            eeg_df[column],
            errors="coerce"
        )


    return eeg_df


def quality_check(eeg_df):
    """
    Perform basic EEG data quality checking.
    """

    issues = []


    # Missing values
    missing_values = int(
        eeg_df.isnull().sum().sum()
    )

    if missing_values > 0:

        issues.append(
            f"Missing values: {missing_values}"
        )


    # Infinite values
    numeric_values = eeg_df.to_numpy(
        dtype=np.float64
    )

    infinite_values = int(
        np.isinf(numeric_values).sum()
    )

    if infinite_values > 0:

        issues.append(
            f"Infinite values: {infinite_values}"
        )


    # Number of samples
    if len(eeg_df) == 0:

        issues.append(
            "No EEG samples found."
        )


    # Constant rows
    constant_rows = 0

    for row in numeric_values:

        if len(row) > 0:

            if np.nanstd(row) == 0:

                constant_rows += 1


    if constant_rows > 0:

        issues.append(
            f"Constant EEG samples: {constant_rows}"
        )


    if len(issues) == 0:

        status = "Good"

    else:

        status = "Needs Review"


    return {
        "status": status,
        "issues": issues,
        "missing_values": missing_values,
        "infinite_values": infinite_values,
        "constant_samples": constant_rows
    }


# ============================================================
# CNN-LSTM PREDICTION
# ============================================================

def predict_cnn_lstm(signal_array):
    """
    Predict one EEG sample using CNN-LSTM.
    """

    if CNN_LSTM_MODEL is None:

        return None


    signal = normalize_signal(
        signal_array
    )


    # Expected CNN-LSTM shape:
    # (batch, 178, 1)

    model_input = signal.reshape(
        1,
        len(signal),
        1
    )


    prediction = CNN_LSTM_MODEL.predict(
        model_input,
        verbose=0
    )


    probability = float(
        np.asarray(prediction).reshape(-1)[0]
    )


    # Probability should be between 0 and 1
    probability = max(
        0.0,
        min(
            1.0,
            probability
        )
    )


    seizure_probability = (
        probability * 100
    )


    if probability >= 0.5:

        prediction_label = (
            "Seizure pattern"
        )

    else:

        prediction_label = (
            "Normal pattern"
        )


    confidence = (
        max(
            probability,
            1 - probability
        ) * 100
    )


    return {
        "prediction": prediction_label,
        "probability": probability,
        "seizure_probability": seizure_probability,
        "confidence": confidence
    }


# ============================================================
# HOME
# ============================================================

@app.route("/", methods=["GET"])
def home():

    return jsonify({

        "project":
            "NeuroSense AI",

        "description":
            "AI-Driven EEG Signal Analysis for Automated "
            "Epileptic Seizure Detection",

        "status":
            "Backend running",

        "endpoints": [

            "/health",
            "/predict",
            "/model-comparison",
            "/generate-explanation"

        ]

    })


# ============================================================
# HEALTH
# ============================================================

@app.route("/health", methods=["GET"])
def health():

    return jsonify({

        "project":
            "NeuroSense AI",

        "description":
            "AI-Driven EEG Signal Analysis for Automated "
            "Epileptic Seizure Detection",

        "status":
            "Backend running",

        "cnn_lstm_loaded":
            CNN_LSTM_MODEL is not None,

        "adaptive_ml":
            "Loaded"
            if ADAPTIVE_ENGINE_LOADED
            else "Unavailable",

        "xai":
            "Loaded"
            if XAI_LOADED
            else "Unavailable"

    })


# ============================================================
# MODEL COMPARISON
# ============================================================

@app.route(
    "/model-comparison",
    methods=["GET"]
)
def model_comparison():

    results = [

        {
            "model": "Random Forest",
            "accuracy": 83.43,
            "precision": 58.11,
            "recall": 61.52,
            "f1": 59.77,
            "roc_auc": 0.8673
        },

        {
            "model": "SVM",
            "accuracy": 57.48,
            "precision": 29.44,
            "recall": 80.65,
            "f1": 43.14,
            "roc_auc": 0.6511
        },

        {
            "model": "XGBoost",
            "accuracy": 85.96,
            "precision": 74.91,
            "recall": 44.78,
            "f1": 56.05,
            "roc_auc": 0.8772
        },

        {
            "model": "CNN-LSTM",
            "accuracy": 98.09,
            "precision": 95.02,
            "recall": 95.43,
            "f1": 95.23,
            "roc_auc": 0.9945
        }

    ]


    return jsonify({

        "success": True,

        "models": results

    })


# ============================================================
# EEG PREDICTION
# ============================================================

@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    temp_path = None


    try:

        # ----------------------------------------------------
        # Check uploaded file
        # ----------------------------------------------------

        if "file" not in request.files:

            return jsonify({

                "success": False,

                "error":
                    "No EEG file uploaded."

            }), 400


        uploaded_file = request.files["file"]


        if uploaded_file.filename == "":

            return jsonify({

                "success": False,

                "error":
                    "No file selected."

            }), 400


        # ----------------------------------------------------
        # Temporary file
        # ----------------------------------------------------

        suffix = os.path.splitext(
            uploaded_file.filename
        )[1]


        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix
        ) as temp_file:

            uploaded_file.save(
                temp_file.name
            )

            temp_path = temp_file.name


        # ----------------------------------------------------
        # Read CSV
        # ----------------------------------------------------

        df = pd.read_csv(
            temp_path
        )


        if df.empty:

            return jsonify({

                "success": False,

                "error":
                    "Uploaded CSV is empty."

            }), 400


        # ----------------------------------------------------
        # Clean EEG data
        # ----------------------------------------------------

        eeg_df = clean_dataframe(
            df
        )


        # ----------------------------------------------------
        # Quality check
        # ----------------------------------------------------

        quality = quality_check(
            eeg_df
        )


        # ----------------------------------------------------
        # Reject critical problems
        # ----------------------------------------------------

        if quality["missing_values"] > 0:

            return jsonify({

                "success": False,

                "error":
                    "EEG data contains missing values.",

                "quality_status":
                    quality["status"],

                "quality_details":
                    quality

            }), 400


        if quality["infinite_values"] > 0:

            return jsonify({

                "success": False,

                "error":
                    "EEG data contains infinite values.",

                "quality_status":
                    quality["status"],

                "quality_details":
                    quality

            }), 400


        # ----------------------------------------------------
        # CNN-LSTM analysis
        # ----------------------------------------------------

        all_predictions = []


        for index, row in eeg_df.iterrows():

            signal = row[
                EXPECTED_COLUMNS
            ].to_numpy(
                dtype=np.float32
            )


            cnn_result = predict_cnn_lstm(
                signal
            )


            if cnn_result is not None:

                all_predictions.append(
                    cnn_result
                )


        # ----------------------------------------------------
        # Final CNN-LSTM result
        # ----------------------------------------------------

        if len(all_predictions) > 0:

            average_probability = float(
                np.mean([
                    item["probability"]
                    for item in all_predictions
                ])
            )


            seizure_probability = (
                average_probability * 100
            )


            if average_probability >= 0.5:

                final_prediction = (
                    "Seizure pattern"
                )

            else:

                final_prediction = (
                    "Normal pattern"
                )


            confidence = (
                max(
                    average_probability,
                    1 - average_probability
                ) * 100
            )


        else:

            average_probability = 0.0

            seizure_probability = 0.0

            final_prediction = (
                "Model unavailable"
            )

            confidence = 0.0


        # ----------------------------------------------------
        # Adaptive ML
        # ----------------------------------------------------

        adaptive_results = []


        high_uncertainty_samples = 0

        low_uncertainty_samples = 0


        if ADAPTIVE_ENGINE_LOADED:

            for index, row in eeg_df.iterrows():

                signal = row[
                    EXPECTED_COLUMNS
                ].to_numpy(
                    dtype=np.float32
                )


                try:

                    adaptive_result = (
                        adaptive_decision(
                            signal
                        )
                    )


                    adaptive_results.append(
                        adaptive_result
                    )


                    uncertainty = (
                        adaptive_result.get(
                            "uncertainty",
                            {}
                        )
                    )


                    uncertainty_level = (
                        uncertainty.get(
                            "uncertainty_level",
                            ""
                        )
                    )


                    if (
                        uncertainty_level
                        == "High"
                    ):

                        high_uncertainty_samples += 1

                    else:

                        low_uncertainty_samples += 1


                except Exception as adaptive_error:

                    print(
                        "Adaptive analysis error:",
                        adaptive_error
                    )


        # ----------------------------------------------------
        # Adaptive status
        # ----------------------------------------------------

        if high_uncertainty_samples > 0:

            adaptive_status = (
                "Deeper analysis recommended "
                "for uncertain samples"
            )

            deep_analysis_required = True

        elif len(adaptive_results) > 0:

            adaptive_status = (
                "Adaptive ML result accepted"
            )

            deep_analysis_required = False

        else:

            adaptive_status = (
                "Adaptive ML unavailable"
            )

            deep_analysis_required = False


        # ----------------------------------------------------
        # XAI
        # ----------------------------------------------------

        xai_result = {}


        if XAI_LOADED:

            try:

                first_signal = eeg_df.iloc[
                    0
                ][
                    EXPECTED_COLUMNS
                ].to_numpy(
                    dtype=np.float32
                )


                xai_result = (
                    explain_eeg_signal(
                        first_signal
                    )
                )

            except Exception as xai_error:

                print(
                    "XAI error:",
                    xai_error
                )


        # ----------------------------------------------------
        # Final response
        # ----------------------------------------------------

        response = {

            "success": True,

            "project":
                "NeuroSense AI",

            "prediction":
                final_prediction,

            "seizure_probability":
                round(
                    seizure_probability,
                    2
                ),

            "confidence":
                round(
                    confidence,
                    2
                ),

            "model":
                "CNN-LSTM",

            "samples_analyzed":
                len(eeg_df),

            "features_per_sample":
                len(EXPECTED_COLUMNS),

            "quality_status":
                quality["status"],

            "quality_details":
                quality,

            "adaptive_analysis":
                adaptive_results,

            "adaptive_status":
                adaptive_status,

            "deep_analysis_required":
                deep_analysis_required,

            "high_uncertainty_samples":
                high_uncertainty_samples,

            "low_uncertainty_samples":
                low_uncertainty_samples,

            "xai":
                xai_result,

            "message":
                "This is an AI-based EEG screening result "
                "and is not a medical diagnosis."

        }


        return jsonify(
            response
        )


    except Exception as e:

        print(
            "\nERROR DURING EEG ANALYSIS:"
        )

        print(
            traceback.format_exc()
        )


        return jsonify({

            "success": False,

            "error":
                str(e),

            "message":
                "Unable to complete EEG analysis."

        }), 500


    finally:

        # ----------------------------------------------------
        # Delete temporary file
        # ----------------------------------------------------

        if (
            temp_path is not None
            and os.path.exists(temp_path)
        ):

            try:

                os.remove(
                    temp_path
                )

            except Exception:

                pass


# ============================================================
# GENERATE EXPLANATION
# ============================================================

@app.route(
    "/generate-explanation",
    methods=["POST"]
)
def generate_ai_explanation():

    try:

        data = request.get_json(
            silent=True
        )


        if not data:

            return jsonify({

                "success": False,

                "error":
                    "No data provided."

            }), 400


        if not GENAI_LOADED:

            return jsonify({

                "success": False,

                "error":
                    "Generative AI module unavailable."

            }), 503


        try:

            explanation = (
                generate_explanation(
                    data
                )
            )


            return jsonify({

                "success": True,

                "explanation":
                    explanation

            })


        except Exception as e:

            return jsonify({

                "success": False,

                "error":
                    str(e),

                "message":
                    "Generative AI explanation could not be generated."

            }), 500


    except Exception as e:

        return jsonify({

            "success": False,

            "error":
                str(e)

        }), 500


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":

    print("")
    print("======================================")
    print("       NEUROSENSE AI BACKEND")
    print("======================================")

    print(
        "CNN-LSTM:",
        "Loaded"
        if CNN_LSTM_MODEL is not None
        else "Unavailable"
    )

    print(
        "Adaptive ML Engine:",
        "Loaded"
        if ADAPTIVE_ENGINE_LOADED
        else "Unavailable"
    )

    print(
        "XAI:",
        "Loaded"
        if XAI_LOADED
        else "Unavailable"
    )

    print("")
    print("Server:")
    print("http://127.0.0.1:5000")
    print("======================================")
    print("")


    # IMPORTANT:
    # Debug is OFF and the reloader is OFF.
    # This prevents TensorFlow from loading the CNN-LSTM
    # model twice and causing Windows VirtualAlloc errors.

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False,
        use_reloader=False
    )