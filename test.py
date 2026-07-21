import os
from src.extract_features import extract_features

sample_file = None

for root, dirs, files in os.walk("dataset"):
    for file in files:
        if file.endswith(".wav"):
            sample_file = os.path.join(root, file)
            break
    if sample_file:
        break

features = extract_features(sample_file)

print("Sample File:", sample_file)
print("Feature Length:", len(features))
print(features)