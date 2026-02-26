from dataclasses import dataclass

@dataclass
class SwtCfg:
    swt_frame_length:int
    swt_hop_ratio:int
    swt_hop_length:int
@dataclass
class MfccCfg:
    mfcc_frame_length:int

@dataclass
class AppCfg:
    swt: SwtCfg
    mfcc:MfccCfg