import matplotlib
matplotlib.use("Agg")
from copy import deepcopy
from pathlib import Path
import yaml
import subprocess
import sys
import time

from cv_evaluate import main as cv_main

BASE_CONFIG_PATH = Path("configs/config.yaml")
TMP_CONFIG_PATH = Path("configs/_sweep_tmp.yaml")

WATKINS_DIR = r"C:/Users/luede/Seafile/WhaleData"
WHALEFM_DIR = r"C:/Users/luede/Seafile/WhaleFM"


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(path: Path, data: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def run_first_threshold_sweep() -> None:
    base_cfg = load_yaml(BASE_CONFIG_PATH)

    jobs = [
        {
            "dataset_name": "watkins",
            "data_dir": WATKINS_DIR,
            "feature_set": ["mfcc"],
            "threshold_rule": "visu",
            "output_dir": "results/cv/watkins_threshold_visu_soft",
        },
        {
            "dataset_name": "watkins",
            "data_dir": WATKINS_DIR,
            "feature_set": ["mfcc"],
            "threshold_rule": "sure",
            "output_dir": "results/cv/watkins_threshold_sure_soft",
        },
        {
            "dataset_name": "watkins",
            "data_dir": WATKINS_DIR,
            "feature_set": ["mfcc"],
            "threshold_rule": "bayes",
            "output_dir": "results/cv/watkins_threshold_bayes_soft",
        },
        {
            "dataset_name": "watkins",
            "data_dir": WATKINS_DIR,
            "feature_set": ["mfcc"],
            "threshold_rule": "percentile",
            "output_dir": "results/cv/watkins_threshold_percentile_soft",
        },
        {
            "dataset_name": "whalefm",
            "data_dir": WHALEFM_DIR,
            "feature_set": ["mfcc", "delta", "delta2"],
            "threshold_rule": "visu",
            "output_dir": "results/cv/whalefm_threshold_visu_soft",
        },
        {
            "dataset_name": "whalefm",
            "data_dir": WHALEFM_DIR,
            "feature_set": ["mfcc", "delta", "delta2"],
            "threshold_rule": "sure",
            "output_dir": "results/cv/whalefm_threshold_sure_soft",
        },
        {
            "dataset_name": "whalefm",
            "data_dir": WHALEFM_DIR,
            "feature_set": ["mfcc", "delta", "delta2"],
            "threshold_rule": "bayes",
            "output_dir": "results/cv/whalefm_threshold_bayes_soft",
        },
        {
            "dataset_name": "whalefm",
            "data_dir": WHALEFM_DIR,
            "feature_set": ["mfcc", "delta", "delta2"],
            "threshold_rule": "percentile",
            "output_dir": "results/cv/whalefm_threshold_percentile_soft",
        },
    ]

    for job in jobs:
        cfg = deepcopy(base_cfg)

        cfg["loadSignal"]["signal_base_dir_in"] = job["data_dir"]
        cfg["mfcc"]["feature_set"] = job["feature_set"]
        cfg["denoise"]["method"] = "swt"
        cfg["swt"]["t_mode"] = job["threshold_rule"]
        cfg["swt"]["t_meth"] = "soft"
        cfg["experiment"]["output_dir"] = job["output_dir"]

        save_yaml(TMP_CONFIG_PATH, cfg)
        time.sleep(10)
        print(f"\n=== Running {job['dataset_name']} | {job['threshold_rule']} | soft ===")
        subprocess.run(
            [sys.executable, "scripts/cv_evaluate.py", str(TMP_CONFIG_PATH)],
            check=True,
)


if __name__ == "__main__":
    run_first_threshold_sweep()