import pywt
import numpy as np
from whaledetection.signal.padding import swt_pad_size, padding
from whaledetection.signal.thresholding import get_threshold

VALID_T_METHS= {"soft","hard","garrote"}

def swt_deconstruct(signal, wavelet="db4", level=3, axis=-1):
    coeffs=pywt.swt(data=signal,wavelet=wavelet,level=level, axis=axis)
    return coeffs

def swt_reconstruct(coeffs, wavelet):
    signal=pywt.iswt(coeffs=coeffs, wavelet=wavelet)
    return signal

def swt_denoise(
        signal,
        wavelet = "db4",
        level=3,
        axis=-1,
        pad_mode="reflect",
        t_mode="visu",
        thresholding="soft",
        k=1.4826,
        percentile=95.0,
        threshold_scale = 1.0
        ):
    thresholding=thresholding.lower()
    if thresholding not in VALID_T_METHS:
        raise ValueError(f"Unknown thresholding method '{thresholding}'use one of: {sorted(VALID_T_METHS)}")
    #pad so right length
    original_length = len(signal)
    target_length = swt_pad_size(signal, level)
    padded_signal=padding(signal,target_length,mode=pad_mode)

    coeffs= swt_deconstruct(signal=padded_signal,
                            wavelet=wavelet,
                            level=level,
                            axis=axis)
    thresholds = get_threshold(coeffs=coeffs,
                                mode=t_mode,
                                signal_length=len(padded_signal),
                                k=k,
                                percentile=percentile)
    thresholds = np.asarray(thresholds, dtype=float) * float(threshold_scale)
    thresholds = np.nan_to_num(thresholds, nan=0.0, posinf=0.0, neginf=0.0)
    thresholds = np.maximum(thresholds, 0.0)
    denoised_coeffs =[]
    for j, (cA, cD) in enumerate(coeffs):
        T = float(thresholds[j])

        if not np.isfinite(T) or T <= 0:
            cD_denoised = np.asarray(cD, dtype=float).copy()
        else:
            cD_denoised = pywt.threshold(cD, value=T, mode=thresholding)

        #cD_denoised = pywt.threshold(cD, value=thresholds[j], mode=thresholding)
        denoised_coeffs.append((cA, cD_denoised))

    denoised_signal = swt_reconstruct(coeffs=denoised_coeffs, wavelet=wavelet)

    return np.asarray(denoised_signal[:original_length], dtype=np.float32)