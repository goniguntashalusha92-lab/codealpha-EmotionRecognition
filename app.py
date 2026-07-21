import os
import tempfile
import joblib
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model

from src.extract_features import extract_features

# Page Title
st.set_page_config(page_title="Speech Emotion Recognition")

st.title("🎤 Speech Emotion Recognition")
st.write("Upload a WAV audio file to detect emotion.")

# Load model and scaler
model = load_model("models/emotion_model.keras")
scaler = joblib.load("models/scaler.pkl")

emotion_labels = [
    "angry",
    "calm",
    "disgust",
    "fearful",
    "happy",
    "neutral",
    "sad",
    "surprised"
]

uploaded_file = st.file_uploader(
    "Choose a WAV file",
    type=["wav"]
)

if uploaded_file is not None:

    st.audio(uploaded_file)

    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.read())
        temp_path = tmp.name

    # Extract Features
    features = extract_features(temp_path)
    features = features.reshape(1, -1)

    # Scale Features
    features = scaler.transform(features)

    # Predict
    prediction = model.predict(features)

    emotion = emotion_labels[np.argmax(prediction)]

    st.success(f"🎯 Predicted Emotion: **{emotion.upper()}**")

    os.remove(temp_path)