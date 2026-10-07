import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    print("ERROR: OPENAI_API_KEY was not found in .env")
    client = None
else:
    print("OpenAI API key loaded successfully.")
    client = OpenAI(api_key=api_key)


def generate_explanation(
    prediction,
    confidence,
    seizure_probability
):
    if client is None:
        raise Exception("OPENAI_API_KEY is missing.")

    prompt = f"""
You are an AI assistant for an EEG seizure screening project called NeuroSense AI.

The EEG machine-learning model produced this result:

Prediction: {prediction}
Confidence: {confidence}%
Seizure probability: {seizure_probability}%

Explain the result in simple language for a project dashboard.

Include:
1. What the model predicted.
2. What the confidence percentage means.
3. What the seizure probability means.
4. Why the result should be reviewed by a healthcare professional.

Important:
- Do not diagnose the patient.
- Do not recommend medication.
- Do not claim medical certainty.
- Clearly state that this is an AI-based screening result, not a medical diagnosis.
"""

    response = client.responses.create(
        model="gpt-5",
        input=prompt
    )

    return response.output_text