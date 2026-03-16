"""
preprocessing.py

Load databases and streams, preprocess it and make it loadable for the svm/k means

Inputs:
###############################################

Outputs:
########################################
"""

# ======================================================
# IMPORTS
# ======================================================
from dataclasses import dataclass
import os
import librosa
import numpy as np
import soundfile as sf
#swt
import pywt
#plotting
import matplotlib.pyplot as plt
import librosa.display
# ======================================================
# Inputs
# ======================================================
class FeatureConfig:
    sr: int = 44032
    mono: bool = True
    species_list: list = ["Northern_Right_Whale"]
    base_dir: str = "C:/Users/luede/Seafile/WhaleData"
    frame_ms:int = 20
    frame_length:int = int(sr * frame_ms / 1000)
    hop_length:int = frame_length // 2
    #windowing
    tukey_alpha: float=0.5 # value is for difference from hann window
    #SWT
    level: int = 4
    wavelet: str="db4"
    axis:int=0 # doesnt matter on 1D audio shape because only 1 axis
    #thresholding
    sigma_mode: str="global_finest"
    energy_weight: bool = True
    beta =0.1
# ======================================================
# Functions
# ======================================================
def tukey_window(N, alpha=0.5):
    """
    Tukey window (ohne scipy).
    alpha=0 -> Rechteck, alpha=1 -> Hann.
    """
    if alpha <= 0.0:
        return np.ones(N, dtype=np.float64)
    if alpha >= 1.0:
        return np.hanning(N).astype(np.float64)

    w = np.ones(N, dtype=np.float64)
    edge = int(np.floor(alpha * (N - 1) / 2.0))  # Samples pro Taper-Seite

    if edge <= 0:
        return w

    n = np.arange(0, edge, dtype=np.float64)
    # linke Taper
    w[:edge] = 0.5 * (1.0 + np.cos(np.pi * (2.0*n/(alpha*(N-1)) - 1.0)))
    # rechte Taper (symmetrisch)
    w[-edge:] = w[:edge][::-1]
    return w


def tukey_middle_slice(N, alpha=0.5):
    """
    Gibt Slice für den 'flachen' Mittelteil des Tukey-Windows zurück.
    Den verwenden wir für robuste sigma/MAD-Schätzung.
    """
    if alpha <= 0.0:
        return slice(0, N)  # keine Taper -> alles gültig

    edge = int(np.floor(alpha * (N - 1) / 2.0))
    if edge <= 0:
        return slice(0, N)

    # falls alpha sehr groß ist, kann der flache Teil sehr klein werden
    if 2 * edge >= N:
        # Notfall: nimm die mittlere Hälfte
        q = N // 4
        return slice(q, N - q)

    return slice(edge, N - edge)
def plot_spectrogram(signal, sr, title="Spectrogram (dB)", n_fft=2048, hop_length=256, fmax=None):
    """
    Plottet ein Log-Power-Spektrogramm in dB.
    signal: 1D np.array
    sr: sampling rate
    """
    signal = np.asarray(signal, dtype=np.float64)

    S = np.abs(librosa.stft(signal, n_fft=n_fft, hop_length=hop_length, window="hann"))**2
    S_db = librosa.power_to_db(S + 1e-12, ref=np.max)

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(S_db, sr=sr, hop_length=hop_length, x_axis="time", y_axis="hz")
    if fmax is not None:
        plt.ylim(0, fmax)
    plt.colorbar(format="%+2.0f dB")
    plt.title(title)
    plt.tight_layout()
# if sr is right, not necessary
def reflect_padding(signal, level):
    signal = np.asarray(signal)
    pad_amount = (-len(signal)) % (2**level)
    return np.pad(signal, (0, pad_amount), mode="reflect")

def swt_decon(signal,wavelet,level,axis=-1):
    coeffs_before = pywt.swt(signal, wavelet=wavelet, level=level, axis=axis)
    return coeffs_before
def swt_recon(signal, wavelet):
    signal= pywt.iswt(signal,wavelet)
    return signal
def threshold(coeffs):
    new_coeffs=[]
    for (cA, cD) in coeffs:
        #############################unzufrieden damit
    ########################################################################################################
        t = np.percentile(np.abs(cD), percentile=80)
        #soft thresholding
        cD_new = np.sign(cD) * np.maximum(np.abs(cD) - t, 0.0)
    return new_coeffs
def mad_sigma(x, valid_slice=None):
    """
    Robust sigma via MAD. Optional nur auf einem Slice (z.B. mittlerer Teil) berechnen.
    """
    x = np.asarray(x, dtype=np.float64)
    if valid_slice is not None:
        x = x[valid_slice]
    return np.median(np.abs(x)) / 0.6745 + 1e-12
def threshold_sureshrink(coeffs, sigma_mode="global_finest", energy_weight=False, beta=0.0, valid_slice=None):
    """
    SureShrink pro SWT-Level.

    valid_slice: z.B. slice(edge, N-edge), um Randartefakte / getaperte Bereiche zu ignorieren.
    """
    # Energiepeak optional (auch hier: wenn valid_slice gesetzt, dann Energie nur im gültigen Bereich)
    if energy_weight:
        energies = []
        for (cA, cD) in coeffs:
            cD = np.asarray(cD, dtype=np.float64)
            if valid_slice is not None:
                cD = cD[valid_slice]
            energies.append(np.mean(cD**2))
        energies = np.array(energies, dtype=np.float64)
        peak_level = int(np.argmax(energies))
    else:
        peak_level = 0

    # Hinweis: bei pywt.swt ist coeffs[0] Level 1 = "finest"
    if sigma_mode == "global_finest":
        sigma_global = mad_sigma(coeffs[0][1], valid_slice=valid_slice)

    new_coeffs = []
    for level_idx, (cA, cD) in enumerate(coeffs):
        cD = np.asarray(cD, dtype=np.float64)

        if sigma_mode == "per_level":
            sigma = mad_sigma(cD, valid_slice=valid_slice)
        else:
            sigma = sigma_global

        # Threshold via SURE (Soft)
        t = sure_threshold_soft(cD if valid_slice is None else cD[valid_slice], sigma)

        if energy_weight:
            w = 1.0 + beta * abs(level_idx - peak_level)
            t *= w

        # Soft-threshold auf vollem cD anwenden (nicht nur Slice), aber t aus mittlerem Bereich stammt
        cD_new = np.sign(cD) * np.maximum(np.abs(cD) - t, 0.0)
        new_coeffs.append((cA, cD_new))

    return new_coeffs
def sure_threshold_soft(cD, sigma):
    """
    SureShrink: wählt t, das SURE(t) minimiert (soft-thresholding).
    cD: 1D array Detailkoeffizienten
    sigma: Rauschstd
    returns: optimaler Threshold t (>=0)
    """
    cD = np.asarray(cD, dtype=np.float64)
    N = cD.size
    if N == 0:
        return 0.0

    y = np.abs(cD)

    # Kandidaten t: sortierte |cD|
    y_sorted = np.sort(y)
    y2_sorted = y_sorted ** 2
    cumsum_y2 = np.cumsum(y2_sorted)

    # Für jeden Kandidaten t = y_sorted[k]:
    # count = k+1  (Anzahl <= t)
    # sum(min(y_i^2, t^2)) = sum_{i<=k} y_i^2 + (N-k-1)*t^2
    k = np.arange(N)
    t2 = y2_sorted
    sum_min = cumsum_y2 + (N - k - 1) * t2
    count_le = k + 1

    # SURE(t) = N*sigma^2 + sum(min(y^2,t^2)) - 2*sigma^2*count(|y|<=t)
    sure = N * (sigma ** 2) + sum_min - 2.0 * (sigma ** 2) * count_le

    best_k = int(np.argmin(sure))
    t_best = float(y_sorted[best_k])

    return t_best



# ======================================================
# MAIN
# ======================================================
for label, sp in enumerate(FeatureConfig.species_list):
    folder = os.path.join(FeatureConfig.base_dir, sp)
    #debugging
    if not os.path.isdir(folder):
        print(f"WARNING: folder not found: {folder}")
        continue
    #loop
    for filename in os.listdir(folder):
        #debugging
        if not filename.lower().endswith((".wav", ".flac", ".aiff", ".aif")):
            continue
        #loop
        path = os.path.join(folder, filename)
        try:
            data, sr = librosa.load(path, sr=FeatureConfig.sr, mono=FeatureConfig.mono)
        except Exception as e:
            print(f"Error loading {path}: {e}")
            continue
        #make frame windows without the uncompleted last
        for start in range(0, len(data) - FeatureConfig.frame_length + 1, FeatureConfig.hop_length):
            segment = data[start:start + FeatureConfig.frame_length].astype(np.float64)
            #windwoing
            # Tukey-Window + gültiger Mittelteil (für sigma)
            w_tukey = tukey_window(FeatureConfig.frame_length, alpha=FeatureConfig.tukey_alpha)
            valid = tukey_middle_slice(FeatureConfig.frame_length, alpha=FeatureConfig.tukey_alpha)
            segment_win = segment * w_tukey
            #padding?
            
            #swt denoising
            coeffs = swt_decon(segment,FeatureConfig.wavelet, FeatureConfig.level,axis=FeatureConfig.axis)
            #thresholding
            coeffs_thr = threshold_sureshrink(
                coeffs,
                sigma_mode=FeatureConfig.sigma_mode,
                energy_weight=FeatureConfig.energy_weight,
                beta=FeatureConfig.beta
            )
            data_den = swt_recon(coeffs_thr,FeatureConfig.wavelet)
            plot_spectrogram(
                data_den,
                sr,
                title=f"Denoised spectrogram | {filename}",
                n_fft=2048,
                hop_length=256,
                fmax=5000  # optional, anpassen/entfernen
            )
plt.show()