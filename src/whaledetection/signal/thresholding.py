import numpy as np

def MAD(signal):
    return np.median(np.abs(signal - np.median(signal)))

def MAD_level_based(coeffs):
    mads=[]
    for cA,cD in coeffs:
        mads.append(MAD(cD))
    return np.asarray(mads,dtype=float)

def sigma_from_MAD(mads,k=1.4826):
    return k* np.asarray(mads, dtype=float)

def visu_threshold(sigmas,signal_length):
    """
    T_j = sigma_j * sqrt(2 log N)
    """
    sigmas = np.asarray(sigmas,dtype=float)
    return sigmas * np.sqrt(2*np.log(signal_length))

def sure_threshold(sigmas, coeffs):
    thresholds=[]

    for j, (cA,cD) in enumerate(coeffs):
        sigma = max(sigmas[j], 1e-12)
        n = len(cD)

        w = np.abs(cD) / sigma
        w_sorted = np.sort(w)
        w2 = w_sorted ** 2

        cumsum_w2 = np.cumsum(w2)
        k_inx = np.arange(1, n + 1)

        risks = n - 2 * k_inx+ cumsum_w2 + (n - k_inx) * w2

        idx = np.argmin(risks)
        t_sure = w_sorted[idx]
        t_visu = np.sqrt(2 * np.log(n))

        T = sigma * min(t_sure, t_visu)

        thresholds.append(T)

    return np.asarray(thresholds)

def bayes_threshold(sigmas, coeffs):
    thresholds=[]
    #take global sigma for noise
    sigma_noise = max(sigmas[0], 1e-12)

    for cA, cD in coeffs:

        sigma_y = np.std(cD)
        sigma_x_sq = max(sigma_y**2 - sigma_noise**2, 0)
        sigma_x = np.sqrt(sigma_x_sq)

        if sigma_x < 1e-12:
            T = np.max(np.abs(cD))
        else:
            T = (sigma_noise**2) / sigma_x

        thresholds.append(T)

    return np.asarray(thresholds)