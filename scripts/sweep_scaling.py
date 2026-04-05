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


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(path: Path, data: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def build_jobs():
    jobs = []

    # HIER SPÄTER DEIN GEWÄHLTES SETUP EINTRAGEN
    selected_n_mfcc = 40
    selected_n_fft = 2048
    selected_win_length = 0.025
    selected_hop_length = 0.01
    selected_wavelet = "db4"

    scales =  [1.0,0.9,0.8,0.7,0.6,0.5]

    for scale in scales:
        scale_str = str(scale).replace(".", "_")
        
        # threshold scaling
        jobs.append(
            {
                "dataset_name": "watkins",
                "data_dir": WATKINS_DIR,
                "feature_set": ["mfcc"],
                "n_mfcc": selected_n_mfcc,
                "n_fft": selected_n_fft,
                "win_length": selected_win_length,
                "hop_length": selected_hop_length,
                "wavelet": selected_wavelet,
                "denoise_method": "swt",
                "threshold_rule": "bayes",
                "shrinkage_rule": "soft",
                "k": BASE_K,
                "threshold_scale": scale,
                "mode_name": "thresholdscale",
                "output_dir": (
                    f"results/cv/"
                    f"watkins_visu_soft_thresholdscale_{scale_str}"
                    f"_wavelet_{selected_wavelet}"
                    f"_mfcc{selected_n_mfcc}"
                    f"_fft{selected_n_fft}_{selected_win_length}_hop_{selected_hop_length}"
                ),
            }
        )
        
        # k scaling
        jobs.append(
            {
                "dataset_name": "watkins",
                "data_dir": WATKINS_DIR,
                "feature_set": ["mfcc"],
                "n_mfcc": selected_n_mfcc,
                "n_fft": selected_n_fft,
                "win_length": selected_win_length,
                "hop_length": selected_hop_length,
                "wavelet": selected_wavelet,
                "denoise_method": "swt",
                "threshold_rule": "bayes",
                "shrinkage_rule": "soft",
                "k": BASE_K * scale,
                "threshold_scale": 1.0,
                "mode_name": "kscaling",
                "scale": scale,
                "output_dir": (
                    f"results/cv/"
                    f"watkins_visu_soft_kscaling_{scale_str}"
                    f"_wavelet_{selected_wavelet}"
                    f"_mfcc{selected_n_mfcc}"
                    f"_fft{selected_n_fft}_{selected_win_length}_hop_{selected_hop_length}"
                ),
            }
        )

    return jobs


def run_scaling_sweep() -> None:
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
        cfg["swt"]["t_mode"] = job["threshold_rule"]
        cfg["swt"]["t_meth"] = job["shrinkage_rule"]
        cfg["swt"]["wavelet"] = job["wavelet"]
        cfg["swt"]["k"] = job["k"]
        cfg["swt"]["threshold_scale"] = job["threshold_scale"]

        cfg["experiment"]["output_dir"] = job["output_dir"]

        save_yaml(TMP_CONFIG_PATH, cfg)

        print(
            f"\n=== Running {job['dataset_name']} | "
            f"{job['mode_name']} | "
            f"wavelet={job['wavelet']} | "
            f"n_mfcc={job['n_mfcc']} | "
            f"n_fft={job['n_fft']} | "
            f"win={job['win_length']} | "
            f"hop={job['hop_length']} | "
            f"k={job['k']:.6f} | "
            f"threshold_scale={job['threshold_scale']} ==="
        )
        print(f"Output: {job['output_dir']}")

        subprocess.run(
            [sys.executable, "scripts/cv_evaluate.py", str(TMP_CONFIG_PATH)],
            check=True,
        )

    print("\nDone.")


if __name__ == "__main__":
    run_scaling_sweep()