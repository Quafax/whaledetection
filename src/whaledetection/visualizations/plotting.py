import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from sklearn.metrics import ConfusionMatrixDisplay
from pathlib import Path

def plot_spectrogram(
    y,
    sr,
    *,
    n_fft=2048,
    hop_length=512,
    win_length=None,
    window="hann",
    center=True,
    pad_mode="reflect",
    power=2.0,
    # Anzeige
    y_axis="log",          # "log" oder "linear"
    cmap="magma",
    top_db=80.0,
    add_colorbar=True,
    colorbar_format="%+2.0f dB",
    # Plot-Layout
    title=None,
    figsize=(12, 5),
    dpi=120,
    xlim=None,             # (t_min, t_max) in Sekunden
    ylim=None,             # (f_min, f_max) in Hz (bei y_axis="linear" am sinnvollsten)
    ax=None,
):
    """
    Plottet ein STFT-Spektrogramm (Magnitude/Power) in dB.

    Parameters
    ----------
    y : np.ndarray
        Audiosignal (1D).
    sr : int
        Samplingrate.
    n_fft, hop_length, win_length, window, center, pad_mode
        STFT-Parameter.
    power : float
        1.0 = Magnitude, 2.0 = Power (üblich).
    y_axis : str
        "log" oder "linear"
    top_db : float
        Dynamikbereich in dB (höher = mehr sichtbar, niedriger = mehr Kontrast).
    xlim : tuple
        Zeitbereich in Sekunden.
    ylim : tuple
        Frequenzbereich in Hz.

    Returns
    -------
    ax : matplotlib.axes.Axes
    """
    y = np.asarray(y).reshape(-1)

    # STFT
    S = librosa.stft(
        y,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=center,
        pad_mode=pad_mode,
    )

    # Magnitude/Power -> dB
    S_mag = np.abs(S) ** power
    S_db = librosa.power_to_db(S_mag, ref=np.max, top_db=top_db)

    # Figure/Axes
    if ax is None:
        _, ax = plt.subplots(figsize=figsize, dpi=dpi)

    img = librosa.display.specshow(
        S_db,
        sr=sr,
        hop_length=hop_length,
        x_axis="time",
        y_axis=y_axis,
        cmap=cmap,
        ax=ax,
    )

    if title is not None:
        ax.set_title(title)

    if xlim is not None:
        ax.set_xlim(xlim)

    if ylim is not None:
        ax.set_ylim(ylim)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency" if y_axis == "linear" else "Frequency (log)")

    if add_colorbar:
        plt.colorbar(img, ax=ax, format=colorbar_format)

    plt.tight_layout()
    return ax

from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay


def plot_confusion_matrix(
    y_true,
    y_pred,
    class_names=None,
    title="Confusion Matrix",
    save_path=None,
    normalize=False,
):
    labels = list(range(len(class_names))) if class_names is not None else None

    fig, ax = plt.subplots(figsize=(16, 14))

    disp = ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        labels=labels,
        display_labels=class_names,
        cmap="Blues",
        xticks_rotation=90,
        normalize="true" if normalize else None,
        values_format=".2f" if normalize else "d",
        include_values=False,
        ax=ax,
        colorbar=True,
    )

    ax.set_title(title, fontsize=18, pad=20)
    ax.set_xlabel("Predicted label", fontsize=14)
    ax.set_ylabel("True label", fontsize=14)

    ax.tick_params(axis="x", labelsize=8)
    ax.tick_params(axis="y", labelsize=8)

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix_seaborn(
    y_true,
    y_pred,
    class_names,
    title="Confusion Matrix",
    save_path=None,
    normalize=False,
):
    labels = list(range(len(class_names)))

    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize="true" if normalize else None)

    fig, ax = plt.subplots(figsize=(18, 16))

    sns.heatmap(
        cm,
        cmap="Blues",
        square=True,
        cbar=True,
        xticklabels=class_names,
        yticklabels=class_names,
        annot=False,
        linewidths=0.2,
        linecolor="lightgray",
        ax=ax,
    )

    ax.set_title(title, fontsize=18, pad=20)
    ax.set_xlabel("Predicted label", fontsize=14)
    ax.set_ylabel("True label", fontsize=14)

    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
