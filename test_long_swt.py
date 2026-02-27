# -*- coding: utf-8 -*-
"""
Denoise long audio by processing overlapping windows with SWT (pywt),
reconstruct with overlap-add (OLA), then plot ONLY a chosen time excerpt
(original + denoised: waveform + spectrogram).
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import pywt

# -------------------- Paths / Settings --------------------
base_dir = r"C:/Users/luede/Seafile/WhaleData/Orginial_from_Watkins"
species = "Teststreams"
filename = "longhumpback.wav"

window_length = 5.0   # seconds
hop_length = 2.5      # seconds (50% overlap)

wavelet = "db4"
level = 7
percentile = 70       # soft-threshold per detail band via percentile(abs(cD))

# Plot only an excerpt (seconds)
t_start = 0.0         # start time in seconds
t_dur = 60.0          # duration in seconds

# Spectrogram plot settings (for excerpt this is usually fine)
n_fft = 2048
hop_stft = 512

# -------------------- Load audio --------------------
folder = os.path.join(base_dir, species)
path = os.path.join(folder, filename)

data, sr = librosa.load(path, sr=None, mono=True)

# -------------------- Window/Hop in samples --------------------
win = int(round(window_length * sr))
hop = int(round(hop_length * sr))

# -------------------- Denoise per segment + Overlap-Add --------------------
y_out = np.zeros_like(data, dtype=np.float64)
w_sum = np.zeros_like(data, dtype=np.float64)

# Smooth cross-fade in overlaps (recommended)
w = np.hanning(win).astype(np.float64)

# SWT commonly requires length divisible by 2**level
block = 2**level

for start in range(0, len(data) - win + 1, hop):
    segment = data[start:start + win].astype(np.float64)

    # Pad segment so length is divisible by 2**level
    pad = (-len(segment)) % block
    if pad:
        seg_pad = np.pad(segment, (0, pad), mode="reflect")
    else:
        seg_pad = segment

    coeffs_before = pywt.swt(seg_pad, wavelet=wavelet, level=level, axis=-1)

    coeffs_after = []
    for (cA, cD) in coeffs_before:
        t = np.percentile(np.abs(cD), percentile)
        cD_new = np.sign(cD) * np.maximum(np.abs(cD) - t, 0.0)  # soft-threshold
        coeffs_after.append((cA, cD_new))

    seg_new = pywt.iswt(coeffs_after, wavelet=wavelet)[:win]  # remove padding

    # Overlap-Add reconstruction
    y_out[start:start + win] += seg_new * w
    w_sum[start:start + win] += w

# Normalize (avoid louder overlaps)
y_out = y_out / np.maximum(w_sum, 1e-12)
y_out = y_out.astype(data.dtype)

# -------------------- Select excerpt for plotting --------------------
i0 = int(round(t_start * sr))
i1 = int(round((t_start + t_dur) * sr))

data_p = data[i0:i1]
y_out_p = y_out[i0:i1]

# -------------------- Plot Waveforms (excerpt) --------------------
t = np.arange(len(data_p)) / sr + t_start

fig, ax = plt.subplots(2, 1, figsize=(14, 6), sharex=True, sharey=True)

ax[0].plot(t, data_p, linewidth=0.7)
ax[0].set_title(f"Original (Waveform)  t={t_start:.1f}..{t_start+t_dur:.1f}s")
ax[0].set_ylabel("Amplitude")

ax[1].plot(t, y_out_p, linewidth=0.7)
ax[1].set_title(f"Denoised (Waveform)  t={t_start:.1f}..{t_start+t_dur:.1f}s")
ax[1].set_xlabel("Time [s]")
ax[1].set_ylabel("Amplitude")

plt.tight_layout()

# -------------------- Plot Spectrograms (excerpt) --------------------
S_orig = librosa.amplitude_to_db(
    np.abs(librosa.stft(data_p, n_fft=n_fft, hop_length=hop_stft)),
    ref=np.max
)
S_deno = librosa.amplitude_to_db(
    np.abs(librosa.stft(y_out_p, n_fft=n_fft, hop_length=hop_stft)),
    ref=np.max
)

fig, ax = plt.subplots(2, 1, figsize=(14, 7), sharex=True, sharey=True)

img0 = librosa.display.specshow(S_orig, sr=sr, hop_length=hop_stft,
                                x_axis="time", y_axis="hz", ax=ax[0])
ax[0].set_title(f"Original (Spectrogram)  t={t_start:.1f}..{t_start+t_dur:.1f}s")
fig.colorbar(img0, ax=ax[0], format="%+2.0f dB")

img1 = librosa.display.specshow(S_deno, sr=sr, hop_length=hop_stft,
                                x_axis="time", y_axis="hz", ax=ax[1])
ax[1].set_title(f"Denoised (Spectrogram)  t={t_start:.1f}..{t_start+t_dur:.1f}s")
fig.colorbar(img1, ax=ax[1], format="%+2.0f dB")

plt.tight_layout()
plt.show()