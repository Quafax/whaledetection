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

