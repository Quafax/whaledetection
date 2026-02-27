import sys
print("sys.path =", sys.path)
from whaledetection.config_loader import load_config

print(sys.path[:3])
from tests.test_ringbuffer_ai import RingBufferMono, iter_mono_windows_from_file
import os
species_list: list = ["Northern_Right_Whale"]
base_dir: str = "C:/Users/luede/Seafile/WhaleData"
filename="0074_7001800B"
wavelet = "db4"
for label, sp in enumerate(species_list):
    folder = os.path.join(base_dir, sp)
    #debugging
    if not os.path.isdir(folder):
        print(f"WARNING: folder not found: {folder}")
        continue
    path = os.path.join(folder, filename)
    try:
        for window, sr in iter_mono_windows_from_file(path, win_s=2.0, hop_ratio=0.5):
            pass
    except Exception as e:
        print(f"Error loading {path}: {e}")
        continue