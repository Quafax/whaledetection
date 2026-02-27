import librosa

def swt_pad_size(signal,level):
    N=len(signal)
    divisor = 2 ** level
    pad_amount = (-N) % divisor
    return N+pad_amount


def reflect_padding(signal,pad_size):
    if pad_size == len(signal):
        return signal
    return librosa.util.pad_center(signal, size=pad_size, mode="reflect", axis=-1)

def zero_padding(signal,pad_size):
    if pad_size == len(signal):
        return signal
    return librosa.util.pad_center(signal, size=pad_size, mode="constant", constant_values=0)

def reflect_padding(signal,pad_size):
    if pad_size == len(signal):
        return signal
    return librosa.util.pad_center(signal, size=pad_size, mode="wrap", axis=-1)
