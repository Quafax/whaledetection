import matplotlib
matplotlib.use("Agg")

from copy import deepcopy
from pathlib import Path
import subprocess
import sys
import yaml

BASE_CONFIG_PATH = Path("configs/config.yaml")
TMP_CONFIG_PATH = Path("configs/_sweep_tmp.yaml")

BASE_K = 1.4826

WATKINS_DIR = r"C:/Users/luede/Seafile/WhaleData"
WHALEFM_DIR = r"C:/Users/luede/Seafile/WhaleFM"


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(path: Path, data: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def build_jobs():
    threshold_scales = [1.0, 0.75, 0.5, 0.25, 0.1]

    datasets = [
        {
            "dataset_name": "watkins",
            "data_dir": WATKINS_DIR,
            "threshold_rules": ["visu", "bayes", "percentile"],  # sure hier skippen
        },
        {
            "dataset_name": "whalefm",
            "data_dir": WHALEFM_DIR,
            "threshold_rules": ["sure", "visu", "bayes", "percentile"],
        },
    ]

    jobs = []

    for dataset in datasets:
        for rule in dataset["threshold_rules"]:
            for scale in threshold_scales:
                scale_str = str(scale).replace(".", "_")

                jobs.append(
                    {
                        "dataset_name": dataset["dataset_name"],
                        "data_dir": dataset["data_dir"],
                        "feature_set": ["mfcc"],
                        "threshold_rule": rule,
                        "shrinkage_rule": "soft",
                        "threshold_scale": scale,
                        "k": BASE_K,
                        "output_dir": (
                            f"results/cv/"
                            f"{dataset['dataset_name']}_threshold_{rule}_soft_scale_{scale_str}"
                        ),
                    }
                )

    return jobs


def run_threshold_scale_sweep() -> None:
    base_cfg = load_yaml(BASE_CONFIG_PATH)
    jobs = build_jobs()

    for job in jobs:
        cfg = deepcopy(base_cfg)

        cfg["loadSignal"]["signal_base_dir_in"] = job["data_dir"]
        cfg["mfcc"]["feature_set"] = job["feature_set"]

        cfg["denoise"]["method"] = "swt"
        cfg["swt"]["t_mode"] = job["threshold_rule"]
        cfg["swt"]["t_meth"] = job["shrinkage_rule"]
        cfg["swt"]["k"] = job["k"]
        cfg["swt"]["threshold_scale"] = job["threshold_scale"]

        cfg["experiment"]["output_dir"] = job["output_dir"]

        save_yaml(TMP_CONFIG_PATH, cfg)

        print(
            f"\n=== Running {job['dataset_name']} | "
            f"{job['threshold_rule']} | {job['shrinkage_rule']} | "
            f"threshold_scale={job['threshold_scale']} | k={job['k']:.4f} ==="
        )
        print(f"Output: {job['output_dir']}")

        subprocess.run(
            [sys.executable, "scripts/cv_evaluate.py", str(TMP_CONFIG_PATH)],
            check=True,
        )

    print("\nDone.")


if __name__ == "__main__":
    run_threshold_scale_sweep()