from dataclasses import dataclass

@dataclass
class loadSignalCfg:
    signal_base_dir_in:str
    signal_base_dir_out: str
    sr : int
    species_list: list
@dataclass
class SwtCfg:
    swt_frame_length:int
    swt_hop_ratio:int
    swt_hop_length:int
    k:float
    t_mode:str
    t_meth:str
    wavelet:str
    axis: int
    level:int
@dataclass
class padCfg:
    pad_mode: str
@dataclass
class denoiseCfg:
    method: str

@dataclass
class MfccCfg:
    mfcc_frame_length: float
    tukey_alpha:float
    hop_length: float
    n_mfcc: int
    n_fft: int
    feature_set: list[str]
@dataclass
class loadDatabaseCfg:
    database_base_dir_in: str
    database_base_dir_out: str

@dataclass
class svmCfg:
    kernel: str
    random_state: int
    model_dir_out: str
    test_size: float

@dataclass
class rfCfg:
    random_state: int
    model_dir_out: str
    test_size: float
    estimators: int

@dataclass
class featureCfg:
    feature_type: str

@dataclass
class AppCfg:
    swt: SwtCfg
    mfcc:MfccCfg
    loadSignal:loadSignalCfg
    loadDatabase: loadDatabaseCfg
    svm: svmCfg
    rf: rfCfg
    feature: featureCfg
    pad:padCfg
    denoise: denoiseCfg
