import numpy as np
import soundfile as sf

class RingBufferMono:
    def __init__(self, capacity: int, dtype=np.float32):
        self.capacity = int(capacity)              # samples
        self.buf = np.zeros((self.capacity,), dtype=dtype)
        self.w = 0
        self.filled = 0

    def write(self, x: np.ndarray):
        x = np.asarray(x, dtype=self.buf.dtype).reshape(-1)
        n = x.size
        if n == 0:
            return

        # if block longer than buffer, keep only the last part
        if n >= self.capacity:
            x = x[-self.capacity:]
            n = x.size

        end = self.w + n
        if end <= self.capacity:
            self.buf[self.w:end] = x
        else:
            first = self.capacity - self.w
            self.buf[self.w:] = x[:first]
            self.buf[:end % self.capacity] = x[first:]

        self.w = end % self.capacity
        self.filled = min(self.capacity, self.filled + n)

    def is_full(self) -> bool:
        return self.filled == self.capacity

    def read_window(self) -> np.ndarray:
        """Oldest -> newest, length = capacity."""
        if not self.is_full():
            raise RuntimeError("RingBuffer not full yet")
        return np.concatenate([self.buf[self.w:], self.buf[:self.w]])


def iter_mono_windows_from_file(
    path: str,
    *,
    win_s: float = 2.0,
    hop_ratio: float = 0.5,
    block_size: int = 8192,
):
    """
    Streams a file, always downmixes to mono, and yields overlapping windows.

    Yields: (window, sr)
      window shape: (win_n,)
    """
    with sf.SoundFile(path, "r") as f:
        sr = f.samplerate
        win_n = int(round(win_s * sr))
        hop_n = int(round(win_n * hop_ratio))
        if hop_n <= 0 or hop_n > win_n:
            raise ValueError("hop_ratio must result in hop_n in [1, win_n]")

        rb = RingBufferMono(capacity=win_n, dtype=np.float32)
        since_last = 0

        while True:
            x = f.read(block_size, dtype="float32", always_2d=True)  # (n, ch)
            if x.shape[0] == 0:
                break

            x_mono = x.mean(axis=1)  # downmix stereo/multichannel -> mono
            rb.write(x_mono)
            since_last += x_mono.size

            while rb.is_full() and since_last >= hop_n:
                window = rb.read_window()

                # ---- HIER würdest du deine SWT machen ----
                # y = my_swt_denoise(window, sr)
                # -----------------------------------------

                yield window, sr
                since_last -= hop_n