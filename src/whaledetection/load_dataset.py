from pathlib import Path
import numpy as np
import librosa
import noisereduce as nr

from whaledetection.signal.swt import swt_denoise

from whaledetection.features.mfcc import extract_mfcc_features
from whaledetection.features.vggish import extract_vggish_features

def load_dataset(cfg):

    X = []
    y = []
    classes = []

    data_dir = Path(cfg.loadSignal.signal_base_dir_in)
    sr = cfg.loadSignal.sr

    feature_type = cfg.feature.feature_type.lower()

    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")
    
    class_dirs = sorted([p for p in data_dir.iterdir() if p.is_dir()])

    for label, class_dir in enumerate(class_dirs):
        classes.append(class_dir.name)

        for wav_file in class_dir.glob("*.wav"):
            try:
                signal, _ = librosa.load(wav_file, sr=sr,mono=True)

                if cfg.denoise.method.lower() == "swt":
                    signal = swt_denoise(signal,
                                        wavelet = cfg.swt.wavelet,
                                        level=cfg.swt.level,
                                        axis=-1,
                                        pad_mode=cfg.pad.pad_mode,
                                        t_mode=cfg.swt.t_mode,
                                        thresholding=cfg.swt.t_meth,
                                        k=cfg.swt.k
                                        )   
                elif cfg.denoise.method.lower() == "noisereduce":
                    signal = nr.reduce_noise(signal, sr)


                if feature_type == "mfcc":
                    features = extract_mfcc_features(
                        signal=signal,
                        sr=sr,
                        frame_length=cfg.mfcc.mfcc_frame_length,
                        n_mfcc=cfg.mfcc.n_mfcc,
                        n_fft=cfg.mfcc.n_fft,
                        hop_length=cfg.mfcc.hop_length,
                        feature_set=cfg.mfcc.feature_set
                    )

                elif feature_type == "vggish":
                    features = extract_vggish_features()

                else:
                    raise ValueError(f"Unknown feature_type '{feature_type}'. Use 'mfcc' or 'vggish'.")
                X.append(features)
                y.append(label)


            except Exception as e:
                print(f"Skipping {wav_file}: {e}")

    if  len(X) == 0:
        raise ValueError(f"No audio features could be loaded from: {data_dir}")
    return np.array(X), np.array(y), classes