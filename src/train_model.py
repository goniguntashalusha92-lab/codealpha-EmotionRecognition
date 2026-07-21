import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

from extract_features import extract_features
from sklearn.preprocessing import StandardScaler
import joblib

# Emotion Labels
emotion_dict = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

X = []
y = []

# Load Dataset
for root, dirs, files in os.walk("dataset"):
    for file in files:
        if file.endswith(".wav"):
            emotion = file.split("-")[2]

            if emotion in emotion_dict:
                file_path = os.path.join(root, file)
                feature = extract_features(file_path)

                X.append(feature)
                y.append(emotion_dict[emotion])

# Convert to NumPy Arrays
X = np.array(X)
y = np.array(y)

# Label Encoding
encoder = LabelEncoder()
y = encoder.fit_transform(y)

# One-Hot Encoding
y = to_categorical(y)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training Samples:", X_train.shape)
print("Testing Samples :", X_test.shape)
scaler=StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build Neural Network
model = Sequential([
    Dense(128, activation="relu", input_shape=(180,)),
    Dropout(0.4),

    Dense(64, activation="relu"),
    Dropout(0.3),

    Dense(32, activation="relu"),
    Dense(8, activation="softmax")
])

# Compile Model
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# Train Model
history = model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_test, y_test)
)

# Evaluate Model
loss, accuracy = model.evaluate(X_test, y_test)

print(f"\nTest Accuracy: {accuracy * 100:.2f}%")

# Save Model
os.makedirs("models", exist_ok=True)

model.save("models/emotion_model.keras")
joblib.dump(scaler, "models/scaler.pkl")

print("✅ Model saved successfully!")
print("✅ Scaler saved successfully!")