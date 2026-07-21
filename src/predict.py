import joblib
import numpy as np
from tensorflow.keras.models import load_model
from extract_features import extract_features

# Load model and scaler
model = load_model("models/emotion_model.keras")
scaler = joblib.load("models/scaler.pkl")

# Emotion labels (LabelEncoder alphabetical order)
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

# Audio file path
audio_file = input("Enter audio file path: ")

# Extract features
features = extract_features(audio_file)
features = features.reshape(1, -1)

# Scale features
features = scaler.transform(features)

# Predict
prediction = model.predict(features)

predicted_emotion = emotion_labels[np.argmax(prediction)]

print("\n🎤 Predicted Emotion:", predicted_emotion)