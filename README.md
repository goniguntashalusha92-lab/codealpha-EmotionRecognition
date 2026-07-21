# 🎤 Speech Emotion Recognition using Deep Learning

## 📌 Project Overview

This project detects human emotions from speech audio using Deep Learning.  
The user uploads a WAV audio file, and the trained model predicts the emotion.

---

## 🎯 Features

- Upload WAV audio files
- Extract audio features using Librosa
- Predict speech emotion using a Deep Neural Network
- Simple Streamlit Web Interface
- Supports 8 emotions

---

## 😊 Supported Emotions

- Neutral
- Calm
- Happy
- Sad
- Angry
- Fearful
- Disgust
- Surprised

---

## 🛠 Technologies Used

- Python
- TensorFlow / Keras
- Librosa
- NumPy
- Scikit-learn
- Streamlit

---

## 📂 Project Structure

```
Emotion_Recognition/
│
├── dataset/
├── models/
│   ├── emotion_model.keras
│   └── scaler.pkl
│
├── src/
│   ├── extract_features.py
│   ├── train_model.py
│   └── predict.py
│
├── app.py
├── requirements.txt
├── README.md
└── venv/
```

---

## 📊 Dataset

Dataset Used:

**RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)**

Total Audio Files: **1440**

---

## 🔍 Feature Extraction

The following audio features are extracted:

- MFCC
- Chroma Features
- Mel Spectrogram

---

## 🧠 Model

Deep Neural Network (DNN)

Architecture:

- Dense (128)
- Dropout
- Dense (64)
- Dropout
- Dense (32)
- Dense (8 - Softmax)

---

## 📈 Model Accuracy

Test Accuracy:

**63.19%**

---

## ▶️ How to Run

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the Model

```bash
python src/train_model.py
```

### 3. Run Prediction

```bash
python src/predict.py
```

### 4. Run Streamlit App

```bash
streamlit run app.py
```

---

## 📷 Output

- Upload a WAV audio file.
- The application predicts the emotion.
- Displays the predicted emotion on the screen.

---

## 🚀 Future Improvements

- Improve model accuracy using CNN/LSTM.
- Real-time microphone input.
- Confidence score visualization.
- Support additional datasets.

---

## 👨‍💻 Author

**Name:** K. Hari Priya

Project: Speech Emotion Recognition using Deep Learning