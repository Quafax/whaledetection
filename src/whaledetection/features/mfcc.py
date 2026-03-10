import librosa
import numpy as np
valid_features = {"mfcc","delta","delta2"}
def summarize(matrix):
    mean = np.mean(matrix, axis=1)
    std = np.std(matrix, axis=1)
    return np.concatenate([mean, std])

def extract_mfcc_features(signal, sr, n_mfcc,frame_length,hop_length,n_fft,feature_set):
    feature_set = set(feature_set)

    invalid = feature_set - valid_features
    if invalid:
        raise ValueError(f"Invalid features: {invalid}")

    if "mfcc" not in feature_set:
        raise ValueError("feature_set must include 'mfcc'")


    mfcc = librosa.feature.mfcc(
        y=signal,
        sr=sr,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=int(hop_length*sr),
        win_length=int(frame_length*sr)
    )
    parts = []

    if "mfcc" in feature_set:
        parts.append(summarize(mfcc))

    if "delta" in feature_set:
        delta = librosa.feature.delta(mfcc)
        parts.append(summarize(delta))

    if "delta2" in feature_set:
        delta2 = librosa.feature.delta(mfcc, order=2)
        parts.append(summarize(delta2))
    features = np.concatenate(parts).astype(np.float32)
    return features