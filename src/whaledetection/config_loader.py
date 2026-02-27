import yaml
from config_types import SwtCfg, AppCfg,MfccCfg
from pathlib import Path

def load_config(path: str | Path) -> AppCfg:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    #swt loaden
    swt_raw=raw["swt"]
    swt_frame_length = swt_raw["swt_frame_length"]
    swt_hop_ratio = swt_raw["swt_hop_ratio"]
    swt_hop_length = int(swt_frame_length/swt_hop_ratio)
    swt = SwtCfg(swt_frame_length=swt_frame_length,swt_hop_ratio=swt_hop_ratio,swt_hop_length=swt_hop_length)
    #mfccs load
    mfcc_raw=raw["mfcc"]
    mfcc_frame_length=mfcc_raw["mfcc_frame_length"]
    mfcc = MfccCfg(mfcc_frame_length=mfcc_frame_length)
    return AppCfg(swt=swt,mfcc=mfcc)
