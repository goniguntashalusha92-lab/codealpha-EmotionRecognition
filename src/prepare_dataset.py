import os
import numpy as np
from extract_features import extract_features

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

dataset_path = "dataset"

for root, dirs, files in os.walk(dataset_path):
    for file in files:
        if file.endswith(".wav"):

            emotion = file.split("-")[2]

            if emotion in emotion_dict:
                feature = extract_features(os.path.join(root, file))

                X.append(feature)
                y.append(emotion_dict[emotion])

X = np.array(X)
y = np.array(y)

print("Features Shape:", X.shape)
print("Labels Shape:", y.shape)
print("First Label:", y[0])