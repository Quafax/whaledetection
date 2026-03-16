# here is code to test the function separated
from swt_denoise_mfcc_extract_linear_svm import swt_denoise_level_based, swt_denoise, spectral_gating_denoise, denoise_signal_percentile, swt_denoise_pywt
import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
import sounddevice as sd

base_dir = "C:/Users/luede/Seafile/WhaleData" #base dir of laptop
#base_dir = "C:/Users/Admin/Seafile/WhaleData" #base dir of desktop
species  = "Common_Dolphin"
file = "0984_5801400J"
sr = 16384
n_mfcc=40
use_denoise = True
wavelet = 'db4'
level = 7
threshold = 0.03
mode = 'soft'
k_factors_mode = None #None, energy, or default
percentile = 80

#load a test file
folder = os.path.join(base_dir, species)
filename = os.path.join(folder, file + ".wav")
path = os.path.join(folder, filename)
data, sr = librosa.load(path, sr=None, mono=True)
x=data
#denoise the test file
print("sr:" + str(sr))

x= spectral_gating_denoise(x, sr, prop_decrease=1.0)
#x= swt_denoise(x, level, wavelet)
#x= swt_denoise_level_based(x, wavelet, level, k_factors_mode, mode)
#x= denoise_signal_percentile(x, wavelet, level, percentile)
#x= swt_denoise_pywt(x, level, wavelet, threshold, mode)

#print the spectrogramm
S = librosa.stft(x)
S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)
plt.figure(figsize=(10, 4))
librosa.display.specshow(S_db,sr=sr,x_axis='time',y_axis='hz')
plt.colorbar(format='%+2.0f dB')
plt.title('Spektrogramm')
plt.tight_layout()
plt.show()