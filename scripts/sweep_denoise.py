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
    jobs = []

    # HIER DEIN FINALES SETUP EINTRAGEN
    selected_n_mfcc = 40
    selected_n_fft = 2048
    selected_win_length = 0.025
    selected_hop_length = 0.01
    selected_wavelet = "db4"

    # HIER EINTRAGEN, WAS AUS DEM SCALING-VERGLEICH GEWONNEN HAT
    # option 1: "threshold_scale"
    # option 2: "k_scaling"
    scaling_mode = "threshold_scale"

    # HIER DEN GEWÄHLTEN BESTEN WERT EINTRAGEN
    selected_scale = 0.5

    datasets = [
        {"name": "watkins", "data_dir": WATKINS_DIR},
        {"name": "whalefm", "data_dir": WHALEFM_DIR},
    ]

    for dataset in datasets:
        base_tag = (
            f"{dataset['name']}_mfcc{selected_n_mfcc}"
            f"_fft{selected_n_fft}_{selected_win_length}_hop_{selected_hop_length}"
            f"_wavelet_{selected_wavelet}"
        )

        # raw
        jobs.append(
            {
                "dataset_name": dataset["name"],
                "data_dir": dataset["data_dir"],
                "feature_set": ["mfcc"],
                "n_mfcc": selected_n_mfcc,
                "n_fft": selected_n_fft,
                "win_length": selected_win_length,
                "hop_length": selected_hop_length,
                "denoise_method": "none",
                "output_dir": f"results/cv/{base_tag}_raw",
            }
        )

        # noisereduce
        jobs.append(
            {
                "dataset_name": dataset["name"],
                "data_dir": dataset["data_dir"],
                "feature_set": ["mfcc"],
                "n_mfcc": selected_n_mfcc,
                "n_fft": selected_n_fft,
                "win_length": selected_win_length,
                "hop_length": selected_hop_length,
                "denoise_method": "noisereduce",
                "output_dir": f"results/cv/{base_tag}_noisereduce",
            }
        )

        # SWT rules
        for rule in ["sure", "visu", "bayes", "percentile"]:
            rule_tag = f"{base_tag}_{rule}_soft"

            if scaling_mode == "threshold_scale":
                k_value = BASE_K
                threshold_scale = selected_scale
                rule_tag += f"_thresholdscale_{str(selected_scale).replace('.', '_')}"
            elif scaling_mode == "k_scaling":
                k_value = BASE_K * selected_scale
                threshold_scale = 1.0
                rule_tag += f"_kscaling_{str(selected_scale).replace('.', '_')}"
            else:
                raise ValueError("scaling_mode must be 'threshold_scale' or 'k_scaling'")

            jobs.append(
                {
                    "dataset_name": dataset["name"],
                    "data_dir": dataset["data_dir"],
                    "feature_set": ["mfcc"],
                    "n_mfcc": selected_n_mfcc,
                    "n_fft": selected_n_fft,
                    "win_length": selected_win_length,
                    "hop_length": selected_hop_length,
                    "denoise_method": "swt",
                    "threshold_rule": rule,
                    "shrinkage_rule": "soft",
                    "wavelet": selected_wavelet,
                    "k": k_value,
                    "threshold_scale": threshold_scale,
                    "output_dir": f"results/cv/{rule_tag}",
                }
            )

    return jobs


def run_final_denoising_sweep() -> None:
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
            cfg["swt"]["wavelet"] = job["wavelet"]
            cfg["swt"]["k"] = job["k"]
            cfg["swt"]["threshold_scale"] = job["threshold_scale"]

        cfg["experiment"]["output_dir"] = job["output_dir"]

        save_yaml(TMP_CONFIG_PATH, cfg)

        print(
            f"\n=== Running {job['dataset_name']} | "
            f"denoise={job['denoise_method']} | "
            f"n_mfcc={job['n_mfcc']} | "
            f"n_fft={job['n_fft']} | "
            f"win={job['win_length']} | "
            f"hop={job['hop_length']} ==="
        )

        if job["denoise_method"] == "swt":
            print(
                f"rule={job['threshold_rule']} | "
                f"wavelet={job['wavelet']} | "
                f"k={job['k']:.6f} | "
                f"threshold_scale={job['threshold_scale']}"
            )

        print(f"Output: {job['output_dir']}")

        subprocess.run(
            [sys.executable, "scripts/cv_evaluate.py", str(TMP_CONFIG_PATH)],
            check=True,
        )

    print("\nDone.")


if __name__ == "__main__":
    run_final_denoising_sweep()