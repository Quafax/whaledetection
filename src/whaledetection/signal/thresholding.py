
import numpy as np

def MAD(signal):
    signal = np.asarray(signal, dtype=float)
    return np.median(np.abs(signal - np.median(signal)))

def MAD_level_based(coeffs):
    mads=[]
    for _,cD in coeffs:
        mads.append(MAD(cD))
    return np.asarray(mads,dtype=float)

def sigma_from_MAD(mads,k=1.4826):
    return k* np.asarray(mads, dtype=float)*0.5

def visu_threshold(sigmas,signal_length):
    sigmas = np.asarray(sigmas,dtype=float)
    return sigmas * np.sqrt(2*np.log(signal_length))

def sure_threshold(sigmas, coeffs):
    thresholds=[]

    for j, (_,cD) in enumerate(coeffs):
        sigma = max(sigmas[j], 1e-12)
        n = len(cD)
        if n == 0:
            thresholds.append(0.0)
            continue
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

    for _, cD in coeffs:

        sigma_y = np.std(cD)
        sigma_x_sq = max(sigma_y**2 - sigma_noise**2, 0)
        sigma_x = np.sqrt(sigma_x_sq)

        if sigma_x < 1e-12:
            T = np.max(np.abs(cD))
        else:
            T = (sigma_noise**2) / sigma_x

        thresholds.append(T)

    return np.asarray(thresholds)

def percentile_threshold(coeffs, percentile=95.0):
    thresholds=[]
    for _,cD in coeffs:
        T = np.percentile(np.abs(cD), percentile)
        thresholds.append(T)
    return np.asarray(thresholds, dtype=float)

def get_threshold(coeffs, mode, signal_length,k=1.4826, percentile=95.0):
    mads= MAD_level_based(coeffs)
    sigmas = sigma_from_MAD(mads,k=k)
    mode= mode.lower()
    if mode == "visu":
        return visu_threshold(sigmas=sigmas,signal_length=signal_length) 

    elif mode == "sure":
        return sure_threshold(coeffs=coeffs, sigmas=sigmas)

    elif mode == "bayes":
        return bayes_threshold(coeffs=coeffs, sigmas=sigmas)
    elif mode=="percentile":
        return percentile_threshold(coeffs=coeffs,percentile=percentile)
    else:
        raise ValueError(f"Unknown threshold_rule: {mode}, use visu, sure, bayes or percentile")
    #could do safety VALID_T_RULE