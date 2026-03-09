import librosa

def swt_pad_size(signal,level):
    N=len(signal)
    divisor = 2 ** level
    pad_amount = (-N) % divisor
    return N+pad_amount


def padding(signal,pad_size,mode):
    if pad_size == len(signal):
        return signal
    return librosa.util.pad_center(signal, size=pad_size, mode=mode, axis=-1)

