from whaledetection.config_loader import load_config
#from root_path import find_project_root
import pywt

#root = find_project_root()
#config_path=root / "configs"/"config.yaml"
#cfg = load_config(config_path)
cfg = load_config("configs/config.yaml")

def swt_deconstruct(signal, wavelet="db4", level=3, axis=0):
    coeffs=pywt.swt(signal=signal,wavelet=wavelet,level=level, axis=axis)
    return coeffs

def swt_reconstruct(coeffs, wavelet):
    signal=pywt.iswt(coeffs=coeffs, wavelet=wavelet)
    return signal