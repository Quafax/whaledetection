from pathlib import Path
import sys

import librosa
import noisereduce as nr

from whaledetection.config.config_loader import load_config
from whaledetection.features.mfcc import extract_mfcc_features
from whaledetection.model.svm import load_model, predict
from whaledetection.signal.swt import swt_denoise


cfg = load_config("configs/config.yaml")


def preprocess_audio(wav_path, cfg):
    signal, _ = librosa.load(wav_path, sr=cfg.loadSignal.sr, mono=True)

    if cfg.denoise.method.lower() == "swt":
        signal = swt_denoise(
            signal,
            wavelet=cfg.swt.wavelet,
            level=cfg.swt.level,
            axis=cfg.swt.axis,
            pad_mode=cfg.pad.pad_mode,
            t_mode=cfg.swt.t_mode,
            thresholding=cfg.swt.t_meth,
            k=cfg.swt.k,
        )
    elif cfg.denoise.method.lower() == "noisereduce":
        signal = nr.reduce_noise(y=signal, sr=cfg.loadSignal.sr)

    features = extract_mfcc_features(
        signal=signal,
        sr=cfg.loadSignal.sr,
        frame_length=cfg.mfcc.mfcc_frame_length,
        n_mfcc=cfg.mfcc.n_mfcc,
        n_fft=cfg.mfcc.n_fft,
        hop_length=cfg.mfcc.hop_length,
        feature_set=cfg.mfcc.feature_set,
    )
    return features


def main():
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python scripts/svm_inference.py <audio.wav>")

    wav_path = Path(sys.argv[1])
    model_path = Path(cfg.svm.model_dir_out)
    if model_path.suffix == "":
        model_path = model_path / "svm_mffc_swt_model.joblib"

    model = load_model(model_path)
    features = preprocess_audio(wav_path, cfg)
    pred = predict(model, features)[0]

    class_dirs = sorted(
        [p for p in Path(cfg.loadSignal.signal_base_dir_in).iterdir() if p.is_dir()]
    )
    class_names = [p.name for p in class_dirs]

    print(f"Predicted class index: {pred}")
    print(f"Predicted class name:  {class_names[pred]}")


if __name__ == "__main__":
    main()