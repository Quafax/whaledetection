#here the plotting of different denoising methods like swt, noisereduce and swt with noiserecuse. Together with the different thresholdings like sureshrink, bayeshrink and visushrink. Could do percentile shrinking and garotte thresholding too

from pathlib import Path
import random

import librosa
import noisereduce as nr
import numpy as np

from whaledetection.config.config_loader import load_config
from whaledetection.signal.swt import swt_denoise
from whaledetection.visualizations.plotting import plot_denoising_comparison


cfg = load_config("configs/config.yaml")


def apply_method(signal, sr, method_name):
    method_name = method_name.lower()

    if method_name == "raw":
        return np.asarray(signal, dtype=np.float32)

    if method_name == "noisereduce":
        denoised = nr.reduce_noise(y=signal, sr=sr)
        return np.asarray(denoised, dtype=np.float32)

    if method_name == "swt_visu_soft":
        return swt_denoise(
            signal=signal,
            wavelet=cfg.swt.wavelet,
            level=cfg.swt.level,
            axis=cfg.swt.axis,
            pad_mode=cfg.pad.pad_mode,
            t_mode="visu",
            thresholding="soft",
            k=cfg.swt.k,
        )

    if method_name == "swt_sure_soft":
        return swt_denoise(
            signal=signal,
            wavelet=cfg.swt.wavelet,
            level=cfg.swt.level,
            axis=cfg.swt.axis,
            pad_mode=cfg.pad.pad_mode,
            t_mode="sure",
            thresholding="soft",
            k=cfg.swt.k,
        )

    if method_name == "swt_bayes_soft":
        return swt_denoise(
            signal=signal,
            wavelet=cfg.swt.wavelet,
            level=cfg.swt.level,
            axis=cfg.swt.axis,
            pad_mode=cfg.pad.pad_mode,
            t_mode="bayes",
            thresholding="soft",
            k=cfg.swt.k,
        )

    if method_name == "swt_percentile_soft":
        return swt_denoise(
            signal=signal,
            wavelet=cfg.swt.wavelet,
            level=cfg.swt.level,
            axis=cfg.swt.axis,
            pad_mode=cfg.pad.pad_mode,
            t_mode="percentile",
            thresholding="soft",
            k=cfg.swt.k,
            percentile=95.0,
        )

    if method_name == "swt_sure_garrote":
        return swt_denoise(
            signal=signal,
            wavelet=cfg.swt.wavelet,
            level=cfg.swt.level,
            axis=cfg.swt.axis,
            pad_mode=cfg.pad.pad_mode,
            t_mode="sure",
            thresholding="garrote",
            k=cfg.swt.k,
        )

    if method_name == "swt_visu_hard":
        return swt_denoise(
            signal=signal,
            wavelet=cfg.swt.wavelet,
            level=cfg.swt.level,
            axis=cfg.swt.axis,
            pad_mode=cfg.pad.pad_mode,
            t_mode="visu",
            thresholding="hard",
            k=cfg.swt.k,
        )

    raise ValueError(f"Unknown method: {method_name}")


def find_example_files(
    base_dir,
    folders=None,
    files_per_folder=1,
    max_files=None,
    random_selection=True,
    seed=42,
):

    base_dir = Path(base_dir)
    rng = random.Random(seed)

    if not base_dir.exists():
        raise FileNotFoundError(f"Base directory not found: {base_dir}")

    if folders is None:
        class_dirs = sorted([p for p in base_dir.iterdir() if p.is_dir()])
    else:
        class_dirs = []
        for folder in folders:
            folder_path = base_dir / folder
            if not folder_path.exists():
                print(f"[WARNING] Folder not found, skipping: {folder_path}")
                continue
            if not folder_path.is_dir():
                print(f"[WARNING] Not a directory, skipping: {folder_path}")
                continue
            class_dirs.append(folder_path)

    selected_files = []

    for class_dir in class_dirs:
        wav_files = sorted(class_dir.glob("*.wav"))

        if len(wav_files) == 0:
            print(f"[WARNING] No wav files in: {class_dir}")
            continue

        if random_selection:
            wav_files = wav_files.copy()
            rng.shuffle(wav_files)

        selected_files.extend(wav_files[:files_per_folder])

    if max_files is not None:
        selected_files = selected_files[:max_files]

    return selected_files


def main():
    sr = cfg.loadSignal.sr
    input_dir = Path(cfg.loadSignal.signal_base_dir_in)
    output_dir = Path("results/denoising")
    output_dir.mkdir(parents=True, exist_ok=True)

#settings
    selected_folders = None

    files_per_folder = 1
    max_files = 5
    random_selection = True
    seed = 42

    methods = [
        "raw",
        "noisereduce",
        "swt_visu_soft",
        "swt_sure_soft",
        "swt_bayes_soft",
        "swt_percentile_soft",
        "swt_sure_garrote",
    ]
#folder selection
    wav_files = find_example_files(
        base_dir=input_dir,
        folders=selected_folders,
        files_per_folder=files_per_folder,
        max_files=max_files,
        random_selection=random_selection,
        seed=seed,
    )

    if not wav_files:
        raise FileNotFoundError(f"No wav files found under {input_dir}")

    print(f"Found {len(wav_files)} example file(s).")
    for wav_path in wav_files:
        print(f" - {wav_path}")

#method
    for wav_path in wav_files:
        print(f"\nProcessing: {wav_path}")

        signal, _ = librosa.load(wav_path, sr=sr, mono=True)

        signals = []
        titles = []

        for method in methods:
            try:
                denoised = apply_method(signal, sr, method)
                signals.append(np.asarray(denoised, dtype=np.float32))
                titles.append(method)
                print(f"   done: {method}")
            except Exception as e:
                print(f"   failed: {method} -> {e}")

        if len(signals) == 0:
            print("   No methods succeeded, skipping file.")
            continue

        class_name = wav_path.parent.name
        save_path = output_dir / f"{class_name}_{wav_path.stem}_denoising_comparison.png"

        plot_denoising_comparison(
            signals=signals,
            sr=sr,
            titles=titles,
            save_path=save_path,
            n_fft=cfg.mfcc.n_fft,
            hop_length=int(cfg.mfcc.hop_length * sr) if cfg.mfcc.hop_length < 1 else cfg.mfcc.hop_length,
        )

        print(f"Saved plot to: {save_path}")


if __name__ == "__main__":
    main()