# =================================================================================================================================================
# IMPORTS
# =================================================================================================================================================
import numpy as np
import librosa
import pywt
import noisereduce as nr
from debugging import debug_plot_swt_reconstructed_details

# =================================================================================================================================================
# PADDING FUNCTIONS
# =================================================================================================================================================

def zero_pad(x, level):
    N = len(x)
    divisor = 2 ** level
    pad_amount = (-N) % divisor
    if pad_amount > 0:
        x = np.concatenate([x, np.zeros(pad_amount)])
    return x

def reflect_pad(x, level):
    N = len(x)
    divisor = 2 ** level
    pad_amount = (-N) % divisor
    if pad_amount > 0:
        x = np.pad(x, (0, pad_amount), mode='reflect')
    return x

def edge_pad(x, level):
    N = len(x)
    divisor = 2 ** level
    pad_amount = (-N) % divisor
    if pad_amount > 0:
        x = np.pad(x, (0, pad_amount), mode='edge')
    return x

def wrap_pad(x, level):
    N = len(x)
    divisor = 2 ** level
    pad_amount = (-N) % divisor
    if pad_amount > 0:
        x = np.pad(x, (0, pad_amount), mode='wrap')
    return x

# =================================================================================================================================================
# STATIONARY WAVELET TRANSFORM FUNCTIONS
# =================================================================================================================================================
def swt_decompose(x, wavelet, level):
    coeffs = pywt.swt(x, wavelet, level=level)   # Liste von (cA, cD)
    return coeffs

def swt_reconstruct(coeffs, wavelet, level):
    recon_signal = pywt.iswt(coeffs, wavelet, level)
    return recon_signal
# =================================================================================================================================================
# MFCC EXTRACTION FUNCTIONS
# =================================================================================================================================================
#exracts the mfccs and projects it into 1D for linear SVM
def extract_mfcc_features_1D(x, sr, n_mfcc, n_fft, hop_length, n_mels):
    mfccs = librosa.feature.mfcc(y=x, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    mfcc_mean = np.mean(mfccs, axis=1)
    mfcc_std = np.std(mfccs, axis=1)
    feat = np.concatenate([mfcc_mean, mfcc_std])
    return feat

def extract_mfcc_features(x, sr, n_mfcc, n_fft, hop_length, n_mels):
    mfccs = librosa.feature.mfcc(y=x, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    return mfccs

# =================================================================================================================================================
# THRESHOLDING FUNCTIONS
# =================================================================================================================================================

def hard_threshold(coeffs, thresholds):
    if len(coeffs) != len(thresholds):
        raise ValueError("coeffs und thresholds müssen gleich lang sein")

    new_coeffs = []
    for (cA, cD), t in zip(coeffs, thresholds):
        t = float(t)
        cD = np.asarray(cD)
        cD_new = cD * (np.abs(cD) >= t)   # hard
        new_coeffs.append((cA, cD_new))   # cA unverändert
    return new_coeffs

def soft_threshold(coeffs, thresholds):
    if len(coeffs) != len(thresholds):
        raise ValueError("coeffs und thresholds müssen gleich lang sein")

    new_coeffs = []
    for (cA, cD), t in zip(coeffs, thresholds):
        t = float(t)
        cD = np.asarray(cD)
        cD_new = np.sign(cD) * np.maximum(np.abs(cD) - t, 0.0)  # soft
        new_coeffs.append((cA, cD_new))
    return new_coeffs

def garrote_threshold(coeffs, thresholds):
    if len(coeffs) != len(thresholds):
        raise ValueError("coeffs und thresholds müssen gleich lang sein")

    new_coeffs = []
    for (cA, cD), t in zip(coeffs, thresholds):
        cD = np.asarray(cD)
        t = float(t)

        abs_cD = np.abs(cD)
        mask = abs_cD > t

        cD_new = np.zeros_like(cD)

        # Garrote-Schrumpfung nur dort, wo |cD| > t
        # (t^2 / cD^2) ist safe, weil mask -> cD != 0 an der Stelle
        cD_new[mask] = cD[mask] * (1.0 - (t * t) / (cD[mask] * cD[mask]))

        new_coeffs.append((cA, cD_new))

    return new_coeffs

def pywt_threshold(coeffs, threshold):
    new_coeffs = []
    for (cA, cD) in coeffs:
        cD_denoised = pywt.threshold(cD, threshold, mode="soft")
        new_coeffs.append((cA, cD_denoised))
    return new_coeffs
# =================================================================================================================================================
# THRESHOLDING RULE FUNCTIONS
# =================================================================================================================================================
def visushrink_threshold(coeffs):
    threshold =[]
    for (cA, cD) in coeffs:
        sigma = np.median(np.abs(cD)) / 0.6745
        N = len(cD)
        t = sigma * np.sqrt(2 * np.log(N))
        threshold.append(t)
    return threshold

def percentile_threshold(coeffs, percentile):
    threshold =[]
    for (cA, cD) in (coeffs):
        t = np.percentile(np.abs(cD), percentile)
        threshold.append(t)
    return threshold
# =================================================================================================================================================
# SPECTRAL GATING FUNCTONS
# =================================================================================================================================================
def spectral_gating_denoise(x, sr, prop_decrease=1.0):
    # x is signal, sr is sampling rate, pro decrease is how much noise to reduce (1.0 is full reduction)
    reduced_noise = nr.reduce_noise(y=x, sr=sr, prop_decrease=prop_decrease)
    return reduced_noise





# =================================================================================================================================================
# =================================================================================================================================================
# =================================================================================================================================================
# =================================================================================================================================================
# MAIN FUNCTION
# =================================================================================================================================================
# =================================================================================================================================================
# =================================================================================================================================================
# =================================================================================================================================================
def main_denoise(padding, x, level, wavelet, k_factors_method, use_debugging, threshold_method, threshold_rule, fixed_threshold, percentile):#use threshold only when threshold rule = None
    #variables
    threshold = []
    new_coeffs = []
    k_factors   = []
    #padding
    if padding =="zero":
        x= zero_pad(x, level)
    elif padding =="reflect":
        x= reflect_pad(x, level)
    elif padding =="edge":
        x= edge_pad(x, level)
    elif padding =="wrap":
        x= wrap_pad(x, level)
    
    #decompose with swt
    coeffs = swt_decompose(x, wavelet, level)

    #calculate k_factors
    if k_factors_method == "default":
        k_factors = np.linspace(3.0, 1.0, level)
    elif k_factors_method == None:
        k_factors = np.ones(level)
    elif k_factors_method == "energy":
        #calculate k_factors based on energy of detail coefficients
        energies = []
        for i in range(level):
            cA, cD = coeffs[i]
            energy = np.sum(cD ** 2)
            energies.append(energy)
        total_energy = np.sum(energies)
        for energy in energies:
            k = 1.0 - (energy / total_energy)
            k_factors.append(k)
    
    #threshold rule
    if threshold_rule == "visushrink":
        threshold = visushrink_threshold(coeffs)
    elif threshold_rule == None:
        threshold = fixed_threshold
    elif threshold_rule =="percentile":
        threshold = percentile_threshold(coeffs, percentile)
    #threshold the coeffs
    if threshold_method == "hard":
        new_coeffs = hard_threshold(coeffs, threshold)
    elif threshold_method =="soft":
        new_coeffs = soft_threshold(coeffs, threshold)
    elif threshold_method == "garotte":
        new_coeffs = garrote_threshold(coeffs, threshold)
    elif threshold_method == "pywt":
        new_coeffs = pywt_threshold(coeffs,threshold)

    # debugging
    if use_debugging:
        debug_plot_swt_reconstructed_details(
        new_coeffs, wavelet, sr=sr,   # sr=None => Samples-Achse
        title="Details nach Thresholding (vor Rekonstruktion)",
        normalize_ylim=True,
        show=False  # wenn du später plt.show() machst
    )

    #recosntruct
    signal = swt_reconstruct(new_coeffs, wavelet,level)
    

    if threshold_method == "noisered":
        signal = spectral_gating_denoise(x,sr, prop_decrease=1.0)
    
    return signal




if __name__ == "__main__":
#testing
    #improtsa
    import os
    from debugging import plot_spec
    #direction
    base_dir = "C:/Users/admin/Seafile/WhaleData" #base dir of desktop
    species  = "Common_Dolphin"
    file = "0984_5801400J"
    folder = os.path.join(base_dir, species)
    filename = os.path.join(folder, file + ".wav")
    path = os.path.join(folder, filename)
    data, sr = librosa.load(path, sr=None, mono=True)
    plot_spec(data, sr)
    signal = main_denoise(padding="zero", x=data, level=7, wavelet='db4',
                          k_factors_method="energy", use_debugging=True,
                          threshold_method="soft", threshold_rule= "percentile",
                          fixed_threshold=0.03, percentile = 70)
    plot_spec(signal, sr)


    import matplotlib.pyplot as plt
    plt.show()  

    