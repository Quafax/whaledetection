import matplotlib
matplotlib.use("Agg")

from copy import deepcopy
from pathlib import Path
import subprocess
import sys
import yaml

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


def build_jobs():
    jobs = []

    datasets = [
        {"name": "watkins", "data_dir": WATKINS_DIR},
       # {"name": "whalefm", "data_dir": WHALEFM_DIR},
    ]

    fft_settings = [
        {"n_fft": 1024, "win_length": 0.02, "hop_length": 0.01},
        {"n_fft": 2048, "win_length": 0.025,  "hop_length": 0.01},
        {"n_fft": 4096, "win_length": 0.05,   "hop_length": 0.01},
    ]

    for dataset in datasets:
        for setting in fft_settings:
            setting_tag = f"fft{setting['n_fft']}_{setting['win_length']}_hop_{setting['hop_length']}"

            # raw
            jobs.append(
                {
                    "dataset_name": dataset["name"],
                    "data_dir": dataset["data_dir"],
                    "feature_set": ["mfcc"],
                    "n_mfcc": 40,
                    "n_fft": setting["n_fft"],
                    "win_length": setting["win_length"],
                    "hop_length": setting["hop_length"],
                    "denoise_method": "none",
                    "output_dir": (
                        f"results/cv/"
                        f"{dataset['name']}_mfcc40_raw_{setting_tag}"
                    ),
                }
            )

            # sure soft
            jobs.append(
                {
                    "dataset_name": dataset["name"],
                    "data_dir": dataset["data_dir"],
                    "feature_set": ["mfcc"],
                    "n_mfcc": 40,
                    "n_fft": setting["n_fft"],
                    "win_length": setting["win_length"],
                    "hop_length": setting["hop_length"],
                    "denoise_method": "swt",
                    "threshold_rule": "sure",
                    "shrinkage_rule": "soft",
                    "output_dir": (
                        f"results/cv/"
                        f"{dataset['name']}_mfcc40_sure_soft_{setting_tag}"
                    ),
                }
            )

    return jobs


def run_fft_win_sweep() -> None:
    base_cfg = load_yaml(BASE_CONFIG_PATH)
    jobs = build_jobs()

    for job in jobs:
        cfg = deepcopy(base_cfg)

        cfg["loadSignal"]["signal_base_dir_in"] = job["data_dir"]

        cfg["mfcc"]["feature_set"] = job["feature_set"]
        cfg["mfcc"]["n_mfcc"] = job["n_mfcc"]
        cfg["mfcc"]["n_fft"] = job["n_fft"]
        cfg["mfcc"]["win_length"] = job["win_length"]
        cfg["mfcc"]["hop_length"] = job["hop_length"]

        cfg["denoise"]["method"] = job["denoise_method"]

        if job["denoise_method"] == "swt":
            cfg["swt"]["t_mode"] = job["threshold_rule"]
            cfg["swt"]["t_meth"] = job["shrinkage_rule"]

        cfg["experiment"]["output_dir"] = job["output_dir"]

        save_yaml(TMP_CONFIG_PATH, cfg)

        print(
            f"\n=== Running {job['dataset_name']} | "
            f"MFCC40 | n_fft={job['n_fft']} | "
            f"win={job['win_length']} | "
            f"hop={job['hop_length']} | "
            f"denoise={job['denoise_method']} ==="
        )
        print(f"Output: {job['output_dir']}")

        subprocess.run(
            [sys.executable, "scripts/cv_evaluate.py", str(TMP_CONFIG_PATH)],
            check=True,
        )

    print("\nDone.")


if __name__ == "__main__":
    run_fft_win_sweep()