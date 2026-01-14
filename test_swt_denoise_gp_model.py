
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
    test_sound = "C:/Users/luede/Seafile/WhaleData/CommonDolphin/58014028.wav"
    #test_sound = "C:/Users/luede/Seafile/WhaleData/NARW/8101301I.wav"
    #test_sound = "C:/Users/luede/Seafile/WhaleData/CommonDolphin/5801400J.wav"
    #test_sound = "C:/Users/luede/Seafile/WhaleData/SouthernRightWhale/8300200C.wav"
    signal, sr = librosa.load(test_sound, sr=sr)
    #pad
    N = len(signal)
    if N % 2 != 0:
        data = np.append(signal, 0)
    #make swt
    coeffs = pywt.swt(signal, wavelet, level=level)
    new_coeffs = []


    #denoise by gp threshold
    for j, (cA_j, cD_j) in enumerate(coeffs, start=1):
        #estimate sigma with gp
        sigma_j = estimate_sigma_gp_1d(cD_j, sample_ratio=sample_ratio, random_state=42)
        #soft thresholding with estimated sigma
        cD_new = gp_soft_threshold(cD_j, thresh_lambda, sigma_j)
        new_coeffs.append((cA_j, cD_new))





    #plot the old coeffs
    for j, (cA_j, cD_j) in enumerate(coeffs, start=1):
        plt.figure()
        plt.plot(cD_j)
        plt.title("Old detail coefficients at level"+str(j))
        plt.xlabel("Sample index")
        plt.ylabel("Amplitude")
        plt.tight_layout()


    #plot the new coeffs
    for j, (cA_j, cD_j) in enumerate(new_coeffs, start=1):
        plt.figure()
        plt.plot(cD_j)
        plt.title("New detail coefficients at level"+str(j))
        plt.xlabel("Sample index")
        plt.ylabel("Amplitude")
        plt.tight_layout()

    #reconstruct signal
    denoised_signal = pywt.iswt(new_coeffs, wavelet)

    #plot the spectrogrramm
    stft = librosa.stft(denoised_signal)
    stft_db = librosa.amplitude_to_db(np.abs(stft), ref=np.max)
    plt.figure(figsize=(10, 4))
    img = librosa.display.specshow(stft_db, sr=sr, x_axis='time', y_axis='hz')
    plt.colorbar(img, format='%+2.0f dB')
    plt.title("Spectrogram of denoised signal")
    plt.tight_layout()
         
    plt.show()
