import numpy as np

def sure_threshold(coeffs,sigmas=None):
    thresholds=[]
    n=len(cD)
    if sigmas ==None:
        mads = MAD_level_based(coeffs)
        sigmas = gauss_sigma_level_based(mads)
    for j,(cA,cD) in enumerate (coeffs):
        sigma = max(sigmas[j], 1e-12)
        #normalize with sigma
        w = cD / sigma
        #get candidates for thresholds
        a = np.sort(np.abs(w))
        a2 = a ** 2

        cumsum_a2 = np.cumsum(a2)

        k = np.arange(1, n + 1)

        risks = n - 2 * k + cumsum_a2 + (n - k) * a2

        idx = np.argmin(risks)
        t_sure = a[idx]

        t_visu = np.sqrt(2 * np.log(n))

        T = sigma * min(t_sure, t_visu)
        thresholds.append(T)

    return np.array(thresholds)


def visu_threshold(sigmas,signal_length):
    thresholds=[]
    N = signal_length
    thresholds = sigmas * np.sqrt(2 * np.log(N))
    return thresholds

def bayes_threshold(coeffs):
#works atm for the first level. Possibly be optimized by level dependencies
    mads = MAD_level_based(coeffs)
    sigmas = gauss_sigma_level_based(mads)

    sigma_n = sigmas[0]
    sigma_n2 = sigma_n**2

    thresholds = []

    for j, (cA, cD) in enumerate(coeffs):

        sigma_y2 = np.mean(cD**2)

        sigma_x2 = max(sigma_y2 - sigma_n2, 0)
        sigma_x = np.sqrt(sigma_x2)

        if sigma_x == 0:
            T = np.max(np.abs(cD))
        else:
            T = sigma_n2 / sigma_x

        thresholds.append(T)
    return np.array(thresholds)

def MAD_level_based(coeffs):
    mads = []
    for j, (cA, cD) in enumerate(coeffs):
        mad_j = np.median(np.abs(cD))
        mads.append(mad_j)
    return mads

def gauss_sigma_level_based(mads):
    return np.array(mads)* (1/ 0.6745)

def sigma_level_based(mads,k):
    return np.array(mads)*k
##############currently working on level baes####################
#extract coeffs with swt, threshold and compute signal back with iswt but for every level the same threshold
def swt_denoise_level_based(x, wavelet, level, k_factors_mode, mode):
    #padding (decide wich padding strategy to use later)
    x=zero_pad(x, level)
    coeffs = pywt.swt(x, wavelet, level=level)
    denoised_coeffs = []
    N = len(x)
    k_factors = []
    if k_factors_mode == "default":
        # e.g. for level=6: [3.0, 2.7, 2.3, 1.8, 1.3, 1.0]
        k_factors = np.linspace(3.0, 1.0, level)
    elif k_factors_mode is None:
        k_factors = np.ones(level)
    elif k_factors_mode == "energy":
        for j, (cA, cD) in enumerate(coeffs):
            energy = np.sqrt(np.mean(cD ** 2))
            if energy > 0.1:
                factor = 1.0
            elif energy > 0.05:
                factor = 1.5
            else:
                factor = 2.0
            
            k_factors.append(factor)     
    k_factors = np.asarray(k_factors)
    assert k_factors.shape[0] == level, "k_factors must have length = level:" + str(level) + ", but got " + str(k_factors.shape[0])
    sigma = np.median(np.abs(coeffs)) / 0.6745 + 1e-12
    # Level-dependent universal threshold
    T = sigma * np.sqrt(2 * np.log(N))
    #k_factors[j] * sigma


 
    for j, (cA, cD) in enumerate(coeffs):
        # Noise estimate sigma from detail coefficients (MAD)
        print("Threshold:"+str(T)+" at level "+str(j+1))

        if mode == 'soft':
            cD_denoised = np.sign(cD) * np.maximum(np.abs(cD) - T, 0.0)
        elif mode == 'hard':
            cD_denoised = cD * (np.abs(cD) >= T)

        denoised_coeffs.append((cA, cD_denoised))


    # ========== PLOTTEN ==========
    # 1 Zeile für Originalsignal + 1 Zeile pro Level
    fig, axes = plt.subplots(level + 1, 1, figsize=(10, 2.5 * (level + 1)), sharex=True)

    # Falls nur ein Level (then axes ist kein Array)
    if level == 1:
        axes = [axes]  # in Liste packen, damit Indexierung konsistent ist

    # Originalsignal
    axes[0].plot(x, label='Original Signal')
    axes[0].set_title('Original Signal')
    axes[0].legend(loc='upper right')

    # Detail-Koeffizienten pro Level plotten (hier: *denoisete* Details)
    for j, (cA_d, cD_d) in enumerate(denoised_coeffs):
        axes[j + 1].plot(cD_d, label=f'Detail Coefficients (D{j+1})')
        axes[j + 1].set_title(f'SWT Detail Coefficients Level {j+1}')
        axes[j + 1].legend(loc='upper right')

    plt.tight_layout()
    # ==============================






    return pywt.iswt(denoised_coeffs, wavelet)



