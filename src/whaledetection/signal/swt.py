import pywt
import numpy as np
from whaledetection.signal.padding import swt_pad_size, padding
from whaledetection.signal.thresholding import (
    sure_threshold,
    visu_threshold,
    bayes_threshold,
    MAD_level_based,
    sigma_from_MAD
)


def swt_deconstruct(signal, wavelet="db4", level=3, axis=-1):
    coeffs=pywt.swt(data=signal,wavelet=wavelet,level=level, axis=axis)
    return coeffs

def swt_reconstruct(coeffs, wavelet):
    signal=pywt.iswt(coeffs=coeffs, wavelet=wavelet)
    return signal

def get_threshold(coeffs, mode, signal_length,k=1.4826):
    mads= MAD_level_based(coeffs)
    sigmas = sigma_from_MAD(mads,k=k)

    if mode.lower() == "visu":
        return visu_threshold(sigmas=sigmas,signal_length=signal_length) 

    elif mode.lower() == "sure":
        return sure_threshold(coeffs=coeffs, sigmas=sigmas)

    elif mode.lower() == "bayes":
        return bayes_threshold(coeffs=coeffs, sigmas=sigmas)

    else:
        raise ValueError(f"Unknown threshold_rule: {mode}")

def swt_denoise(
        signal,
        wavelet = "db4",
        level=3,
        axis=-1,
        pad_mode="reflect",
        t_mode="visu",
        thresholding="soft",
        k=1.4826
        ):
    
    #pad so right length
    original_length = len(signal)
    target_length = swt_pad_size(signal, level)
    padded_signal=padding(signal,target_length,mode=pad_mode)

    coeffs= swt_deconstruct(signal=padded_signal,
                            wavelet=wavelet,
                            level=level,
                            axis=axis)
    thresholds = get_threshold(
        coeffs=coeffs,
        mode=t_mode,
        signal_length=len(padded_signal),
        k=k
    )
    denoised_coeffs =[]
    for j, (cA, cD) in enumerate(coeffs):
        cD_denoised = pywt.threshold(cD, value=thresholds[j], mode=thresholding)
        denoised_coeffs.append((cA, cD_denoised))

    denoised_signal = swt_reconstruct(coeffs=denoised_coeffs, wavelet=wavelet)

    return np.asarray(denoised_signal[:original_length], dtype=np.float32)