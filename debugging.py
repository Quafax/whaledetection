# =================================================================================================================================================
# IMPORTS
# =================================================================================================================================================
import matplotlib.pyplot as plt
import librosa
import numpy as np
import pywt
# =================================================================================================================================================
# PLOT FUNCTIONS
# =================================================================================================================================================
def plot_spec(y,sr):
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
    S_db = librosa.amplitude_to_db(S, ref=np.max)
    plt.figure(figsize=(9, 3))
    librosa.display.specshow(S_db, sr=sr, hop_length=512, x_axis="time", y_axis="hz", cmap="magma")
    plt.colorbar(format="%+2.0f dB")
    plt.title("Spectrogram")
    plt.tight_layout()


def debug_plot_swt_reconstructed_details(coeffs, wavelet, sr=None,
                                        title="SWT: rekonstruierte Detail-Beiträge pro Level",
                                        normalize_ylim=True,
                                        show=True):
    """
    coeffs: Liste von (cA, cD) z.B. coeffs oder new_coeffs (nach Thresholding)
    wavelet: z.B. 'db4'
    sr: optional, wenn None -> x-Achse in Samples, sonst Zeit in Sekunden
    normalize_ylim: gleiche y-Limits für alle Subplots (besser vergleichbar)
    """
    level = len(coeffs)
    n = len(coeffs[0][0])

    x_axis = (np.arange(n) / sr) if (sr is not None) else np.arange(n)
    x_label = "Zeit [s]" if (sr is not None) else "Samples"

    # erst alle Detail-Signale berechnen (für einheitliche y-Limits)
    details = []
    for j in range(level):
        coeffs_one = []
        for i, (cA, cD) in enumerate(coeffs):
            cA0 = np.zeros_like(cA)  # Approximation weglassen
            cD0 = cD if (i == j) else np.zeros_like(cD)
            coeffs_one.append((cA0, cD0))

        detail_sig = pywt.iswt(coeffs_one, wavelet)
        details.append(detail_sig)

    # Plot
    fig, axes = plt.subplots(level, 1, sharex=True, figsize=(12, 2.2 * level))
    if level == 1:
        axes = [axes]

    if normalize_ylim:
        m = max(np.max(np.abs(d)) for d in details) + 1e-12

    for j, d in enumerate(details, start=1):
        ax = axes[j-1]
        ax.plot(x_axis, d, lw=0.8)
        ax.set_title(f"Detail-Beitrag (Level {j})")
        ax.grid(True, alpha=0.3)
        if normalize_ylim:
            ax.set_ylim(-m, m)

    axes[-1].set_xlabel(x_label)
    fig.suptitle(title)
    fig.tight_layout()
    fig.subplots_adjust(top=0.93)

    if show:
        plt.show()

    return fig, axes, details

