import numpy as np
import librosa

def extract_features(file_path):
    audio, sample_rate = librosa.load(file_path, sr=None)

    # MFCC
    mfcc = np.mean(
        librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40).T,
        axis=0
    )

    # Chroma
    stft = np.abs(librosa.stft(audio))
    chroma = np.mean(
        librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,
        axis=0
    )

    # Mel Spectrogram
    mel = np.mean(
        librosa.feature.melspectrogram(y=audio, sr=sample_rate).T,
        axis=0
    )

    # Combine Features
    return np.hstack([mfcc, chroma, mel])