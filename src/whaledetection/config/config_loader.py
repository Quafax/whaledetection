import yaml
from pathlib import Path
from whaledetection.config.config_types import (
    SwtCfg,
    AppCfg,
    MfccCfg,
    loadSignalCfg,
    loadDatabaseCfg,
    svmCfg,
)

def load_config(path: str | Path) -> AppCfg:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    mfcc_raw = raw["mfcc"]
    mfcc_frame_length = mfcc_raw["mfcc_frame_length"]
    n_fft = mfcc_raw["n_fft"]
    n_mfcc = mfcc_raw["n_mfcc"]
    hop_length = mfcc_raw["hop_length"]
    tukey_alpha = mfcc_raw["tukey_alpha"]

    mfcc = MfccCfg(
        mfcc_frame_length=mfcc_frame_length,
        n_fft=n_fft,
        n_mfcc=n_mfcc,
        tukey_alpha=tukey_alpha,
        hop_length=hop_length,
    )

    loadSignal_raw = raw["loadSignal"]
    signal_base_dir_in = loadSignal_raw["signal_base_dir_in"]
    signal_base_dir_out = loadSignal_raw["signal_base_dir_out"]
    sr = loadSignal_raw["sr"]
    species_list = loadSignal_raw["species_list"]

    loadSignal = loadSignalCfg(
        sr=sr,
        signal_base_dir_in=signal_base_dir_in,
        signal_base_dir_out=signal_base_dir_out,
        species_list=species_list,
    )

    swt_raw = raw["swt"]
    swt_frame_length = swt_raw["swt_frame_length"]
    swt_hop_ratio = swt_raw["swt_hop_ratio"]
    swt_hop_length = int(swt_frame_length * swt_hop_ratio * sr)

    swt = SwtCfg(
        swt_frame_length=swt_frame_length,
        swt_hop_ratio=swt_hop_ratio,
        swt_hop_length=swt_hop_length,
    )

    loadDatabase_raw = raw["loadDatabase"]
    database_base_dir_in = loadDatabase_raw["database_base_dir_in"]
    database_base_dir_out = loadDatabase_raw["database_base_dir_out"]

    loadDatabase = loadDatabaseCfg(
        database_base_dir_in=database_base_dir_in,
        database_base_dir_out=database_base_dir_out,
    )

    svm_raw = raw["svm"]
    random_state = svm_raw["random_state"]
    kernel = svm_raw["kernel"]
    model_dir_out = svm_raw["model_dir_out"]
    test_size = svm_raw["test_size"]

    svm = svmCfg(
        kernel=kernel,
        random_state=random_state,
        model_dir_out=model_dir_out,
        test_size=test_size,
    )

    return AppCfg(
        swt=swt,
        mfcc=mfcc,
        loadSignal=loadSignal,
        loadDatabase=loadDatabase,
        svm=svm,
    )