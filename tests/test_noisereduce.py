import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np
import os
import noisereduce as nr
import pywt
import sounddevice as sd

def plot_spectrogram(y, sr, title="Spectrogram", n_fft=2048, hop_length=512, cmap="magma"):
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    S_db = librosa.amplitude_to_db(S, ref=np.max)

    plt.figure(figsize=(9, 3))
    librosa.display.specshow(S_db, sr=sr, hop_length=hop_length,
                             x_axis="time", y_axis="hz", cmap=cmap)
    plt.colorbar(format="%+2.0f dB")
    plt.title(title)
    plt.tight_layout()


if __name__ == "__main__":
    base_dir = "C:/Users/luede/Seafile/WhaleData"
    species = "Common_Dolphin"
    species2 = "Northern_Right_Whale"
    file2 ="1194_5900300B "#"0030_56018002"    
    file = "0984_5801400J"
    level = 7
    wavelet = "db4"
    thr = 0.015
    folder = os.path.join(base_dir, species)
    for filename in os.listdir(folder):
                if not filename.lower().endswith((".wav", ".flac", ".aiff", ".aif")):
                    continue

                path = os.path.join(folder, filename)
                try:
                    data, sr = librosa.load(path, sr=None, mono=True)

                    denoised_signal = nr.reduce_noise(y=data, sr=sr, prop_decrease=1.0, stationary=False)

                    plot_spectrogram(data[:len(data)], sr, title="After noisereduce (cropped)")
                    plot_spectrogram(denoised_signal[:len(data)], sr, title="Original (padded, cropped)")



                except Exception as e:
                    print(f"Error loading {path}: {e}")
                    continue

    plt.show()