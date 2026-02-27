import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np
import os
import noisereduce as nr
import pywt
import sounddevice as sd

base_dir = "C:/Users/luede/Seafile/WhaleData"
species2 = "Northern_Right_Whale"
species = "Common_Dolphin"
filename ="0466_5801402C.wav"
level = 7
wavelet = "db4"
thr = 0.005
folder = os.path.join(base_dir, species)
def reflect_padding(signal, level):
    signal = np.asarray(signal)
    pad_amount = (-len(signal)) % (2**level)
    return np.pad(signal, (0, pad_amount), mode="reflect")


path = os.path.join(folder, filename)
print(path)
data, sr = librosa.load(path, sr=None, mono=True)
data_pad = reflect_padding(data, level)
coeffs_before = pywt.swt(data_pad, wavelet=wavelet, level=level)

t = np.arange(len(data_pad)) / sr

# Matrix: (level, N) mit |cD_j|
D = np.stack([np.abs(cD) for (cA, cD) in coeffs_before], axis=0)

plt.figure(figsize=(12, 4))
plt.imshow(
    D,
    aspect="auto",
    origin="lower",
    extent=[t[0], t[-1], 1, level],   # x: Zeit [s], y: Level
    interpolation="nearest"
)
plt.colorbar(label="|cD|")
plt.xlabel("Zeit [s]")
plt.ylabel("SWT Level (Skala)")
plt.title(f"{filename} | SWT-Details (|cD|)")

# optional: Frequenzlabels (grobe Bandmitten)
# Band für Level j: [sr/2^(j+1), sr/2^j], Bandmitte ~ sr / 2^(j+0.5)
ylevels = np.arange(1, level + 1)
fmid = sr / (2 ** (ylevels + 0.5))
plt.yticks(ylevels, [f"{j} (~{fm:.0f} Hz)" for j, fm in zip(ylevels, fmid)])

plt.tight_layout()
plt.show()
