import librosa
import numpy as np

def swt_pad_size(signal,level):
    N=len(signal)
    divisor = 2 ** level
    pad_amount = (-N) % divisor
    return N+pad_amount


def padding(signal,pad_size,mode):
    if pad_size == len(signal):
        return signal
    return librosa.util.pad_center(signal, size=pad_size, mode=mode, axis=-1)

#because librosa needs a certain length for delta and deta 2 
def pad_signal_for_delta(signal, sr, frame_length, hop_length, n_fft, min_frames=9):
    hop_samples = int(hop_length * sr)
    win_samples = int(frame_length * sr)

    # Konservativ: genug Samples für mindestens min_frames Analysepositionen
    base_window = max(win_samples, n_fft)
    min_len = base_window + hop_samples * (min_frames - 1)

    if len(signal) < min_len:
        pad_amount = min_len - len(signal)
        signal = np.pad(signal, (0, pad_amount), mode="constant")

    return signal