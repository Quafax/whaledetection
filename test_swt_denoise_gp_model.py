
#############################
# Here i tested to threshold by using a gp model to model the gaussion noice and using that to calculate the threshold.
# Works quite good but is absolutly slow and in no way usable. (MAybe on a big server, have to try with uni server)

#works for higher frequebncies okay but not on lower atm. could polish but not worth the effort right now
#############################

import librosa
import librosa.display 
import matplotlib.pyplot as plt
import numpy as np
import pywt
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel




##############Variables####################
sr= None  # use original sampling rate
wavelet = 'db4' #wavelet type
level=3 #decomposition level
thresh_lambda = 0.015 #general multiplier for threshold
sample_ratio = 0.01 #ratio of samples to use for gp fitting

## padding###
def pad_to_multiple(x, m):
    """Pad x with zeros so that len(x) is a multiple of m."""
    n = len(x)
    r = n % m
    if r == 0:
        return x
    pad = m - r
    return np.pad(x, (0, pad), mode="constant")





def estimate_sigma_gp_1d(coeffs, sample_ratio=sample_ratio, random_state=0):
    """
    Estimate a global noise standard deviation using Gaussian Process Regression for 1D coefficient
    its a very lightweight gp model and not at all polished because it takes too long for my application
    random_state is the seed
    """
    # random number with fixed seed for reproducing results
    rng = np.random.RandomState(random_state)
    #make a 1d array of floats from coeffs
    flat = coeffs.ravel().astype(float)
    n = flat.size
    #samples for gp fitting either 50 or sample_ratio*n, whichever is larger
    m = max(50, int(sample_ratio * n))
    #randomly take those amount of samples
    idx = rng.choice(n, size=m, replace=False)
    X = idx[:, None]         # positions
    y = flat[idx]

    # define and fit GP model
    kernel = 1.0 * RBF(length_scale=50.0) + WhiteKernel(noise_level=1.0)
    gpr = GaussianProcessRegressor(kernel=kernel, alpha=0.0, normalize_y=True)
    gpr.fit(X, y)

    #estimate noise level
    noise_level = gpr.kernel_.k2.noise_level
    sigma = np.sqrt(noise_level)
    return sigma


def gp_soft_threshold(x, lam, sigma):
    """
    soft thresholding
    lam is the general multiplier for the threshold
    sigma from my gp model
    """
    #calculate threshold
    thr = lam * sigma
    #thresholding
    return np.sign(x) * np.maximum(np.abs(x) - thr, 0.0)


if __name__ == "__main__":
    test_sound = "C:/Users/Admin/Seafile/WhaleData/Common_Dolphin/0017_5801403R.wav"
    #test_sound = "C:/Users/luede/Seafile/WhaleData/NARW/8101301I.wav"
    #test_sound = "C:/Users/luede/Seafile/WhaleData/CommonDolphin/5801400J.wav"
    #test_sound = "C:/Users/luede/Seafile/WhaleData/SouthernRightWhale/8300200C.wav"
    signal, sr = librosa.load(test_sound, sr=sr)
    #pad
    signal_padded = pad_to_multiple(signal, 2**level)

    #make swt
    coeffs = pywt.swt(signal_padded, wavelet, level=level)
    new_coeffs = []


    #denoise by gp threshold
    for j, (cA_j, cD_j) in enumerate(coeffs, start=1):
        #estimate sigma with gp
        sigma_j = estimate_sigma_gp_1d(cD_j, sample_ratio=sample_ratio, random_state=42)
        #soft thresholding with estimated sigma
        cD_new = gp_soft_threshold(cD_j, thresh_lambda, sigma_j)
        new_coeffs.append((cA_j, cD_new))
    denoised_signal = pywt.iswt(new_coeffs, wavelet)


# --- Trim/align lengths for plotting (because SWT padding) ---
orig_plot = signal                           # unpadded original
den_plot  = denoised_signal[:len(signal)]    # trim denoised to original length

# ---------- ONE FIGURE, ALL SUBPLOTS ----------
# Rows:
# 0..level-1 : coeffs (old/new)
# level      : spectrograms (orig/den)
# level+1    : waveform (orig+den)  [optional, but included]
fig = plt.figure(figsize=(14, 2.2 * level + 8))
gs = fig.add_gridspec(
    level + 2, 2,
    height_ratios=[1]*level + [2.4, 1.4],
    hspace=0.7, wspace=0.25
)

# Coeff plots (dynamic for any level)
for i in range(level):
    cD_old = coeffs[i][1]
    cD_new = new_coeffs[i][1]

    ax_old = fig.add_subplot(gs[i, 0])
    ax_new = fig.add_subplot(gs[i, 1], sharex=ax_old, sharey=ax_old)

    ax_old.plot(cD_old, lw=0.8)
    ax_old.set_title(f"Old detail cD (Level {i+1})")
    ax_old.set_ylabel("Amp.")

    ax_new.plot(cD_new, lw=0.8)
    ax_new.set_title(f"New detail cD (Level {i+1})")

    if i == level - 1:
        ax_old.set_xlabel("Sample index")
        ax_new.set_xlabel("Sample index")
    else:
        ax_old.tick_params(labelbottom=False)
        ax_new.tick_params(labelbottom=False)

# --- Spectrograms: Original vs Denoised (side-by-side) ---
n_fft = 2048
hop_length = 512

stft_o = librosa.stft(orig_plot, n_fft=n_fft, hop_length=hop_length)
stft_d = librosa.stft(den_plot,  n_fft=n_fft, hop_length=hop_length)

# same dB reference so colors are comparable
ref_max = max(np.max(np.abs(stft_o)), np.max(np.abs(stft_d)))
S_o = librosa.amplitude_to_db(np.abs(stft_o), ref=ref_max)
S_d = librosa.amplitude_to_db(np.abs(stft_d), ref=ref_max)

ax_so = fig.add_subplot(gs[level, 0])
img1 = librosa.display.specshow(S_o, sr=sr, hop_length=hop_length,
                                x_axis='time', y_axis='hz', ax=ax_so)
ax_so.set_title("Original spectrogram")
fig.colorbar(img1, ax=ax_so, format='%+2.0f dB')

ax_sd = fig.add_subplot(gs[level, 1])
img2 = librosa.display.specshow(S_d, sr=sr, hop_length=hop_length,
                                x_axis='time', y_axis='hz', ax=ax_sd)
ax_sd.set_title("Denoised spectrogram")
fig.colorbar(img2, ax=ax_sd, format='%+2.0f dB')

# --- Waveform (optional but handy) ---
ax_wave = fig.add_subplot(gs[level + 1, :])
t = np.arange(len(orig_plot)) / sr
ax_wave.plot(t, orig_plot, lw=0.8, alpha=0.7, label="Original")
ax_wave.plot(t, den_plot,  lw=0.9, alpha=0.8, label="Denoised")
ax_wave.set_title("Waveform (time domain)")
ax_wave.set_xlabel("Time [s]")
ax_wave.set_ylabel("Amplitude")
ax_wave.legend(loc="upper right")

fig.tight_layout()
plt.show()