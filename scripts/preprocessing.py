
from whaledetection.config.config_loader import load_config
import librosa
import os
from whaledetection.visualizations.plotting import plot_spectrogram
from whaledetection.signal.framing import window_signal, overlap_add

#später egal
import matplotlib.pyplot as plt
cfg = load_config("configs/config.yaml")

species_list = cfg.load.species_list
base_dir_in = cfg.load.base_dir_in
sr= cfg.load.sr
filename="0074_7001800B.wav"
wavelet = "db4"
window_length =2 #in s
swt_hop_length = cfg.swt.swt_hop_length
swt_frame_length = cfg.swt.swt_frame_length
print("swt hop length:"+str(swt_hop_length))
print("swt frame length:"+str(swt_frame_length))


#load signal
for label, sp in enumerate(species_list):
    folder = os.path.join(base_dir_in, sp)
    if not filename.lower().endswith((".wav", ".flac", ".aiff", ".aif")):
        continue
    #can loop with: for filename in os.listdir(folder):
    path = os.path.join(folder, filename)
    signal, sr = librosa.load(path, sr=sr, mono=True)

#window into 2s windows
windows = window_signal(signal,sr=sr,window_length=swt_frame_length, hop_length=swt_hop_length)
signal = overlap_add(windows=windows,hop_length=swt_hop_length,window=None)
plot_spectrogram(signal,sr)
plt.show()
#swt
#threshold with sureshrink
# iswt

#mfccs


