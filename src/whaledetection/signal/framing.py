import numpy as np
from whaledetection.signal.padding import padding
def window_signal(signal, sr , window_length, hop_length=None):
    """
    Cut signal into windows

    Parameters:
    signal: original signal
    sr: sampling rate
    window_length: wanted length of the windoes in samples
    hop_length: length the window is supposed to hop in samples
    """
    #hop_length ins percentile like 0.5 for 50%, window length in s
    #make sure hop_length and window length is right
    window_length = int(window_length*sr)
    if hop_length is None:
        hop_length = window_length
    
    #calculate how many windows are in the signal
    n_windows = int(np.ceil(max(len(signal) - window_length,0)/ hop_length) +1)

    windows = []
    for i in range(n_windows):
        start = i * hop_length
        end = start+window_length
        window=signal[start:end]
        #fill last one with zeros
        if len(window) < window_length:
            window = np.pad(window, (0, window_length - len(window)))
        windows.append(window)
    return np.array(windows)

def overlap_add(windows, hop_length, *, window=None):
    """
    Overlapp add 2D signal to 1D array

    Parameters:
    windows: 2D array of window frames
    hop_length : hop in samples
    window : window used in window function example: hann, hamming tukey etc
    """
    windows = np.asarray(windows)
    if windows.ndim != 2:
        raise ValueError("windows need to be 2D array: (n_frames, frame_length)")

    hop_length = int(hop_length)
    if hop_length <= 0:
        raise ValueError("hop_length needs to be > 0 ")

    n_frames, frame_length = windows.shape
    out_length = (n_frames - 1) * hop_length + frame_length

    # window_weights w
    if window is None:
        w = np.ones(frame_length, dtype=np.float32)

    elif isinstance(window, str):
        name = window.lower()
        if name == "hann":
            w = np.hanning(frame_length).astype(np.float32)
        elif name == "hamming":
            w = np.hamming(frame_length).astype(np.float32)
        else:
            raise ValueError("window needs to be None, 'hann', 'hamming', ('tukey', alpha) or np.ndarray.")

    elif isinstance(window, tuple):
        name = str(window[0]).lower()
        if name != "tukey":
            raise ValueError("tuple-window only for ('tukey', alpha).")
        alpha = float(window[1]) if len(window) > 1 else 0.5

        n = frame_length
        x = np.linspace(0.0, 1.0, n, endpoint=True)
        w = np.ones(n, dtype=np.float32)

        if alpha <= 0:
            w[:] = 1.0
        elif alpha >= 1:
            w = np.hanning(n).astype(np.float32)
        else:
            edge = alpha / 2.0
            left = x < edge
            right = x > (1.0 - edge)

            w[left] = 0.5 * (1.0 + np.cos(np.pi * (2.0 * x[left] / alpha - 1.0)))
            w[right] = 0.5 * (1.0 + np.cos(np.pi * (2.0 * x[right] / alpha - 2.0 / alpha + 1.0)))

    else:
        w = np.asarray(window, dtype=np.float32)
        if w.shape[0] != frame_length:
            raise ValueError("custom window needs length = frame_length")

    # adding and normilasation
    y = np.zeros(out_length, dtype=np.float32)
    norm = np.zeros(out_length, dtype=np.float32)

    for i in range(n_frames):
        start = i * hop_length
        end = start + frame_length
        y[start:end] += windows[i] * w
        norm[start:end] += w

    nonzero = norm > 1e-12
    y[nonzero] /= norm[nonzero]

    return y