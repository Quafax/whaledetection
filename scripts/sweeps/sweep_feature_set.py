from copy import deepcopy
from pathlib import Path
import subprocess
import sys
import time
import yaml

BASE_CONFIG_PATH = Path("configs/config.yaml")
TMP_CONFIG_PATH = Path("configs/_feature_sweep_tmp.yaml")

WATKINS_DIR = r"C:/Users/luede/Seafile/WhaleData"
WHALEFM_DIR = r"C:/Users/luede/Seafile/WhaleFM"

FEATURE_SETS = {
    "mfcc": ["mfcc"],
    "mfcc_delta_delta2": ["mfcc", "delta", "delta2"],
}

DENOISE_SETTINGS = [
    {
        "name": "raw",
        "method": "raw",   # undenoised
        "t_mode": None,
        "t_meth": None,
    },
    {
        "name": "visu_soft",
        "method": "swt",
        "t_mode": "visu",
        "t_meth": "soft",
    },
    {
        "name": "sure_soft",
        "method": "swt",
        "t_mode": "sure",
        "t_meth": "soft",
    },
    {
        "name": "bayes_soft",
        "method": "swt",
        "t_mode": "bayes",
        "t_meth": "soft",
    },
]

DATASETS = [
    {
        "name": "watkins",
        "data_dir": WATKINS_DIR,
    },
    #if whalefm too:
    {
        "name": "whalefm",
        "data_dir": WHALEFM_DIR,
    },
]


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(path: Path, data: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def build_output_dir(dataset_name: str, feature_name: str, denoise_name: str) -> str:
    return f"results/cv/{dataset_name}_{feature_name}_{denoise_name}"


def run_job(base_cfg: dict, dataset_name: str, data_dir: str, feature_name: str, feature_set: list[str], denoise_cfg: dict) -> None:
    cfg = deepcopy(base_cfg)

    # dataset
    cfg["loadSignal"]["signal_base_dir_in"] = data_dir

    # features
    cfg["mfcc"]["feature_set"] = feature_set

    # denoising
    cfg["denoise"]["method"] = denoise_cfg["method"]
    if denoise_cfg["method"] == "swt":
        cfg["swt"]["t_mode"] = denoise_cfg["t_mode"]
        cfg["swt"]["t_meth"] = denoise_cfg["t_meth"]

    # output
    cfg["experiment"]["output_dir"] = build_output_dir(
        dataset_name=dataset_name,
        feature_name=feature_name,
        denoise_name=denoise_cfg["name"],
    )

    save_yaml(TMP_CONFIG_PATH, cfg)

    print(
        f"\n=== Running "
        f"{dataset_name} | {feature_name} | {denoise_cfg['name']} ==="
    )

    subprocess.run(
        [sys.executable, "scripts/cv_evaluate.py", str(TMP_CONFIG_PATH)],
        check=True,
    )


def run_feature_set_sweep() -> None:
    base_cfg = load_yaml(BASE_CONFIG_PATH)

    for dataset in DATASETS:
        for feature_name, feature_set in FEATURE_SETS.items():
            for denoise_cfg in DENOISE_SETTINGS:
                run_job(
                    base_cfg=base_cfg,
                    dataset_name=dataset["name"],
                    data_dir=dataset["data_dir"],
                    feature_name=feature_name,
                    feature_set=feature_set,
                    denoise_cfg=denoise_cfg,
                )
                time.sleep(2)


if __name__ == "__main__":
    run_feature_set_sweep()