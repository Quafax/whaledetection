import yaml
from whaledetection.config.config_types import mlpCfg, padCfg,featureCfg,SwtCfg, rfCfg,AppCfg,MfccCfg,loadSignalCfg, loadDatabaseCfg, svmCfg, denoiseCfg, experimentCfg
from pathlib import Path

def load_config(path: str | Path) -> AppCfg:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    #mfccs load
    mfcc_raw=raw["mfcc"]
    mfcc_frame_length=mfcc_raw["mfcc_frame_length"]
    n_fft=mfcc_raw["n_fft"]
    n_mfcc = mfcc_raw["n_mfcc"]
    hop_length=mfcc_raw["hop_length"]
    tukey_alpha=mfcc_raw["tukey_alpha"]
    feature_set=mfcc_raw["feature_set"]
    mfcc = MfccCfg(mfcc_frame_length=mfcc_frame_length,
                   n_fft=n_fft,
                   n_mfcc=n_mfcc,
                   tukey_alpha=tukey_alpha,
                   hop_length=hop_length,
                   feature_set=feature_set
                   )

    #load loader :D
    loadSignal_raw = raw["loadSignal"]
    signal_base_dir_in = loadSignal_raw["signal_base_dir_in"]
    signal_base_dir_out = loadSignal_raw["signal_base_dir_out"]
    sr = loadSignal_raw["sr"]
    species_list = loadSignal_raw["species_list"]
    loadSignal = loadSignalCfg(sr=sr,
                               signal_base_dir_in=signal_base_dir_in,
                               signal_base_dir_out=signal_base_dir_out,
                               species_list=species_list)

    #swt loaden
    swt_raw=raw["swt"]
    swt_frame_length = swt_raw["swt_frame_length"]
    swt_hop_ratio = swt_raw["swt_hop_ratio"]
    swt_hop_length = int(swt_frame_length*swt_hop_ratio*sr)
    k = swt_raw["k"]
    level=swt_raw["level"]
    wavelet = swt_raw["wavelet"]
    t_mode = swt_raw["t_mode"]
    t_meth=swt_raw["t_meth"]
    percentile=swt_raw["percentile"]
    axis=swt_raw["axis"]
    threshold_scale =swt_raw["threshold_scale"]
    swt = SwtCfg(swt_frame_length=swt_frame_length,
                 swt_hop_ratio=swt_hop_ratio,
                 swt_hop_length=swt_hop_length,
                 k=k,
                 wavelet=wavelet,
                 t_mode=t_mode,
                 t_meth=t_meth,
                 axis=axis,
                 level=level,
                 percentile=percentile,
                 threshold_scale=threshold_scale)

    loadDatabase_raw = raw["loadDatabase"]
    database_base_dir_in = loadDatabase_raw["database_base_dir_in"]
    database_base_dir_out = loadDatabase_raw["database_base_dir_out"]
    loadDatabase = loadDatabaseCfg(database_base_dir_in=database_base_dir_in,
                                    database_base_dir_out=database_base_dir_out)

    pad_raw = raw["pad"]
    pad_mode=pad_raw["pad_mode"]
    pad = padCfg(pad_mode=pad_mode)


    svm_raw = raw["svm"]
    random_state = svm_raw["random_state"]
    kernel = svm_raw["kernel"]
    model_dir_out = svm_raw["model_dir_out"]
    test_size = svm_raw["test_size"]
    svm = svmCfg(kernel=kernel,
                 random_state=random_state,
                 model_dir_out=model_dir_out,
                 test_size=test_size)

    rf_raw = raw["rf"]
    random_state = rf_raw["random_state"]
    model_dir_out = rf_raw["model_dir_out"]
    test_size = rf_raw["test_size"]
    estimators = rf_raw["estimators"]
    rf = rfCfg(random_state=random_state,
               model_dir_out=model_dir_out,
               test_size=test_size,
               estimators=estimators)
    
    feature_raw = raw["feature"]
    feature_type = feature_raw["feature_type"]
    feature = featureCfg(feature_type=feature_type)

    denoise_raw = raw["denoise"]
    method = denoise_raw["method"]
    denoise = denoiseCfg(method=method)

    mlp_raw = raw["mlp"]
    random_state=mlp_raw["random_state"]
    test_size=mlp_raw["test_size"]
    model_dir_out=mlp_raw["model_dir_out"]
    epochs=mlp_raw["epochs"]
    patience=mlp_raw["patience"]
    dropout=mlp_raw["dropout"]
    hidden_1=mlp_raw["hidden_1"]
    hidden_2=mlp_raw["hidden_2"]
    hidden_3=mlp_raw["hidden_3"]
    hidden_4=mlp_raw["hidden_4"]
    batch_size=mlp_raw["batch_size"]
    learning_rate=mlp_raw["learning_rate"]
    mlp = mlpCfg(random_state=random_state,
                 test_size=test_size,
                 model_dir_out=model_dir_out,
                 epochs=epochs,
                 patience=patience,
                 dropout=dropout,
                 hidden_1=hidden_1,
                 hidden_2=hidden_2,
                 hidden_3=hidden_3,
                 hidden_4=hidden_4,
                 batch_size=batch_size,
                 learning_rate=learning_rate,
                 )
    
    experiment_raw = raw["experiment"]
    output_dir = experiment_raw["output_dir"]
    experiment = experimentCfg(output_dir=output_dir)

    return AppCfg(pad=pad,
                  swt=swt,
                  mfcc=mfcc,
                  loadSignal=loadSignal,
                  loadDatabase=loadDatabase,
                  svm=svm,
                  rf=rf,
                  feature=feature,
                  denoise=denoise,
                  mlp=mlp,
                  experiment=experiment,)
