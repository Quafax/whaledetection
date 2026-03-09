import librosa
import numpy as np
def extract_mfcc_features(signal, sr, n_mfcc,frame_length,hop_length,n_fft):
    

    mfcc = librosa.feature.mfcc(
        y=signal,
        sr=sr,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=int(hop_length*sr),
        win_length=int(frame_length*sr)
    )
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)

    features = np.concatenate([mfcc_mean, mfcc_std]).astype(np.float32)
    return features