import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np
import os
import noisereduce as nr
import pywt
import sounddevice as sd

def swt_detail_band(sr, j):
    f_lo = sr / (2**(j+1))
    f_hi = sr / (2**j)
    return f_lo, f_hi

def plot_swt_level_bands_before_after(coeffs_before, coeffs_after, sr, wavelet,
                                      keep="detail", crop_len=None,
                                      n_fft=2048, hop_length=512,
                                      cmap="magma", prefix="SWT rekonstruiert"):
    level = len(coeffs_before)
    assert len(coeffs_after) == level

    # Rekonstruiere Band-Signale pro Level (Zeitbereich)
    bands_before = []
    bands_after = []
    for j in range(1, level + 1):
        b0 = reconstruct_single_swt_level(coeffs_before, wavelet, j, keep=keep)
        b1 = reconstruct_single_swt_level(coeffs_after,  wavelet, j, keep=keep)

        if crop_len is not None:
            b0 = b0[:crop_len]
            b1 = b1[:crop_len]

        bands_before.append(b0)
        bands_after.append(b1)

    # Gemeinsame Referenz für dB-Skalierung über ALLE Subplots
    S_list = []
    for y in bands_before + bands_after:
        S_list.append(np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length)))
    ref = max(S.max() for S in S_list)

    fig, axes = plt.subplots(level, 2, figsize=(12, 2.2 * level), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.07, right=0.86, top=0.95, bottom=0.06, hspace=0.40, wspace=0.10)

    im = None
    for j in range(level):
        # ohne Threshold
        S0 = np.abs(librosa.stft(bands_before[j], n_fft=n_fft, hop_length=hop_length))
        S0_db = librosa.amplitude_to_db(S0, ref=ref)
        im = librosa.display.specshow(S0_db, sr=sr, hop_length=hop_length,
                                      x_axis="time", y_axis="hz", cmap=cmap, ax=axes[j, 0])
        axes[j, 0].set_title(f"{prefix} L{j+1} ({keep}) - ohne TH")

        # mit Threshold
        S1 = np.abs(librosa.stft(bands_after[j], n_fft=n_fft, hop_length=hop_length))
        S1_db = librosa.amplitude_to_db(S1, ref=ref)
        librosa.display.specshow(S1_db, sr=sr, hop_length=hop_length,
                                 x_axis="time", y_axis="hz", cmap=cmap, ax=axes[j, 1])
        axes[j, 1].set_title(f"{prefix} L{j+1} ({keep}) - mit TH")

    # eine Colorbar rechts
    cax = fig.add_axes([0.88, 0.08, 0.02, 0.86])
    fig.colorbar(im, cax=cax, format="%+2.0f dB")

    return fig


def reflect_padding(signal, level):
    signal = np.asarray(signal)
    pad_amount = (-len(signal)) % (2**level)
    return np.pad(signal, (0, pad_amount), mode="reflect")


def plot_spectrogram(y, sr, title="Spectrogram", n_fft=2048, hop_length=512, cmap="magma"):
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    S_db = librosa.amplitude_to_db(S, ref=np.max)

    plt.figure(figsize=(9, 3))
    librosa.display.specshow(S_db, sr=sr, hop_length=hop_length,
                             x_axis="time", y_axis="hz", cmap=cmap)
    plt.colorbar(format="%+2.0f dB")
    plt.title(title)
    plt.tight_layout()

def reconstruct_single_swt_level(coeffs, wavelet, level_to_keep, keep="detail"):
    """
    Rekonstruiert Zeitbereichssignal nur aus einem SWT-Level.
    keep="detail" -> nur cD dieses Levels
    keep="approx" -> nur cA dieses Levels
    """
    new_coeffs = []
    for j, (cA, cD) in enumerate(coeffs, start=1):
        zA = np.zeros_like(cA)
        zD = np.zeros_like(cD)

        if j == level_to_keep:
            if keep == "detail":
                new_coeffs.append((zA, cD))
            elif keep == "approx":
                new_coeffs.append((cA, zD))
            else:
                raise ValueError("keep must be 'detail' or 'approx'")
        else:
            new_coeffs.append((zA, zD))

    return pywt.iswt(new_coeffs, wavelet)

def threshold(coeffs, threshold):
    """Threshold nur auf Details cD, cA bleibt unverändert."""
    new_coeffs = []
    for (cA, cD) in coeffs:
        cD_denoised = pywt.threshold(cD, threshold, mode="soft")
        new_coeffs.append((cA, cD_denoised))
    return new_coeffs

if __name__ == "__main__":
    base_dir ="C:/Users/luede/Seafile/WhaleData/Orginial_from_Watkins"
    #base_dir = "C:/Users/luede/Seafile/WhaleData"
    species = "Common_Dolphin"
    species ="Teststreams"
    species2 = "Northern_Right_Whale"
    file2 ="1194_5900300B.wav"#"0030_56018002"    
    file = "0984_5801400J.wav"
    file3 = "0466_5801402C.wav"
    fileStream="longhumpback.wav"
    filename= fileStream
    level = 7
    wavelet = "sym8"
    thr = 0.01
    percentile = 80
    coeffs_after = []
    folder = os.path.join(base_dir, species)

    path = os.path.join(folder, filename)

    data, sr = librosa.load(path, sr=None, mono=True)
    print(sr)
    orig_padded = reflect_padding(data, level=level)
    ######################
    #orig_padded = nr.reduce_noise(orig_padded, sr= sr, prop_decrease=1.0,stationary=False)
    #plot_spectrogram(orig_padded[:len(data)], sr, title="Denoised Signal with NR")
###############################
    coeffs_before = pywt.swt(orig_padded, wavelet=wavelet, level=level, start_level=0, axis=-1)
    #coeffs_after = threshold(coeffs_before, thr)
    for inx,(cA, cD) in enumerate(coeffs_before,start=0):
        j= level-inx
        t = np.percentile(np.abs(cD), percentile)
        f_lo, f_hi = swt_detail_band(sr, j)
        print(f"Level {j}: D{j} ~ {f_lo:.1f} .. {f_hi:.1f} Hz")
        if j>=3:
            t=np.inf
        if j<=2:
            t=0.1
        if j==2:
            t=0.013
        print("Threshold="+str(t))
        #cD_new = np.sign(cD) * np.maximum(np.abs(cD) - t, 0.0)  # soft
        if np.isinf(t):
            cD_new = np.zeros_like(cD)
        else:
            cD_new = pywt.threshold(cD, t, mode="soft")
        coeffs_after.append((cA, cD_new))
    denoised_signal = pywt.iswt(coeffs_after, wavelet= wavelet)
    #denoised_signal_nr = nr.reduce_noise(denoised_signal,sr=sr, prop_decrease=1.0, stationary = False)
    plot_spectrogram(data[:len(data)], sr, title="original Signal")
    plot_spectrogram(denoised_signal[:len(data)], sr, title="Denoised Signal")
    #plot_spectrogram(denoised_signal_nr[:len(data)], sr, title="Denoised Signal with NR")
    plt.show()