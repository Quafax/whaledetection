import matplotlib
matplotlib.use("Agg")

from copy import deepcopy
from pathlib import Path
import subprocess
import sys
import yaml

#paths
BASE_CONFIG_PATH = Path("configs/config.yaml")
TMP_CONFIG_PATH = Path("configs/_sweep_tmp.yaml")

WATKINS_DIR = r"C:/Users/luede/Seafile/WhaleData"
WHALEFM_DIR = r"C:/Users/luede/Seafile/WhaleFM"

#defaults
SELECTED_N_MFCC = 40
SELECTED_N_FFT = 2048
SELECTED_WIN_LENGTH = 0.025
SELECTED_HOP_LENGTH = 0.01

FEATURE_SET_WATKINS = ["mfcc"]
FEATURE_SET_WHALEFM = ["mfcc", "delta", "delta2"]

SELECTED_SHRINKAGE = "garrote"
SELECTED_WAVELET = "db4"

BASE_K = 1.4826
SURE_FACTORS = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
WAVELETS = ["db4", "db8", "sym8"]
THRESHOLD_RULES_FINAL = ["sure", "visu", "bayes", "percentile"]


VISU_SCALING_KIND = "threshold_scale"   # options: "threshold_scale" or "k"

#switch
RUN_WAVELET_SWEEP = True
RUN_SURE_K_SCALING = True
RUN_SURE_THRESHOLD_SCALING = True
RUN_VISU_SCALING_SANITY = True
RUN_FINAL_SWT_COMPARISON = True


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def save_yaml(path: Path, data: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)

def scale_to_str(value: float) -> str:
    return str(value).replace(".", "_")

def make_base_job(
    dataset_name: str,
    data_dir: str,
    feature_set: list[str],
    threshold_rule: str,
    wavelet: str,
    shrinkage_rule: str,
    output_dir: str,
    *,
    k: float = BASE_K,
    threshold_scale: float = 1.0,
) -> dict:
    return {
        "dataset_name": dataset_name,
        "data_dir": data_dir,
        "feature_set": feature_set,
        "n_mfcc": SELECTED_N_MFCC,
        "n_fft": SELECTED_N_FFT,
        "win_length": SELECTED_WIN_LENGTH,
        "hop_length": SELECTED_HOP_LENGTH,
        "denoise_method": "swt",
        "threshold_rule": threshold_rule,
        "shrinkage_rule": shrinkage_rule,
        "wavelet": wavelet,
        "k": k,
        "threshold_scale": threshold_scale,
        "output_dir": output_dir,
    }

def apply_job_to_config(cfg: dict, job: dict) -> dict:
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
    return cfg

def run_jobs(jobs: list[dict], phase_name: str) -> None:
    if not jobs:
        print(f"\n[{phase_name}] No jobs to run.")
        return

    base_cfg = load_yaml(BASE_CONFIG_PATH)

    print(f"\n=== {phase_name}: {len(jobs)} runs ===")
    for i, job in enumerate(jobs, start=1):
        cfg = deepcopy(base_cfg)
        cfg = apply_job_to_config(cfg, job)
        save_yaml(TMP_CONFIG_PATH, cfg)

        print(
            f"\n[{phase_name} {i}/{len(jobs)}] "
            f"{job['dataset_name']} | "
            f"rule={job['threshold_rule']} | "
            f"shrinkage={job['shrinkage_rule']} | "
            f"wavelet={job['wavelet']} | "
            f"k={job['k']:.4f} | "
            f"threshold_scale={job['threshold_scale']} | "
            f"features={job['feature_set']}"
        )
        print(f"Output: {job['output_dir']}")

        subprocess.run(
            [sys.executable, "scripts/cv_evaluate.py", str(TMP_CONFIG_PATH)],
            check=True,
        )
#jobs
def build_wavelet_jobs() -> list[dict]:
    jobs = []
    for wavelet in WAVELETS:
        jobs.append(
            make_base_job(
                dataset_name="watkins",
                data_dir=WATKINS_DIR,
                feature_set=FEATURE_SET_WATKINS,
                threshold_rule="sure",
                wavelet=wavelet,
                shrinkage_rule=SELECTED_SHRINKAGE,
                output_dir=(
                    "results/cv/"
                    f"watkins_wavelet_{wavelet}"
                    f"_sure_{SELECTED_SHRINKAGE}"
                    f"_mfcc{SELECTED_N_MFCC}"
                    f"_fft{SELECTED_N_FFT}_{SELECTED_WIN_LENGTH}"
                    f"_hop_{SELECTED_HOP_LENGTH}"
                ),
            )
        )
    return jobs

def build_sure_k_scaling_jobs() -> list[dict]:
    jobs = []
    for factor in SURE_FACTORS:
        jobs.append(
            make_base_job(
                dataset_name="watkins",
                data_dir=WATKINS_DIR,
                feature_set=FEATURE_SET_WATKINS,
                threshold_rule="sure",
                wavelet=SELECTED_WAVELET,
                shrinkage_rule=SELECTED_SHRINKAGE,
                k=BASE_K * factor,
                threshold_scale=1.0,
                output_dir=(
                    "results/cv/"
                    f"watkins_sure_{SELECTED_SHRINKAGE}"
                    f"_kscaling_{scale_to_str(factor)}"
                    f"_wavelet_{SELECTED_WAVELET}"
                    f"_mfcc{SELECTED_N_MFCC}"
                    f"_fft{SELECTED_N_FFT}_{SELECTED_WIN_LENGTH}"
                    f"_hop_{SELECTED_HOP_LENGTH}"
                ),
            )
        )
    return jobs

def build_sure_threshold_scaling_jobs() -> list[dict]:
    jobs = []
    for factor in SURE_FACTORS:
        jobs.append(
            make_base_job(
                dataset_name="watkins",
                data_dir=WATKINS_DIR,
                feature_set=FEATURE_SET_WATKINS,
                threshold_rule="sure",
                wavelet=SELECTED_WAVELET,
                shrinkage_rule=SELECTED_SHRINKAGE,
                k=BASE_K,
                threshold_scale=factor,
                output_dir=(
                    "results/cv/"
                    f"watkins_sure_{SELECTED_SHRINKAGE}"
                    f"_thresholdscale_{scale_to_str(factor)}"
                    f"_wavelet_{SELECTED_WAVELET}"
                    f"_mfcc{SELECTED_N_MFCC}"
                    f"_fft{SELECTED_N_FFT}_{SELECTED_WIN_LENGTH}"
                    f"_hop_{SELECTED_HOP_LENGTH}"
                ),
            )
        )
    return jobs

def build_visu_scaling_jobs() -> list[dict]:
    jobs = []
    for factor in SURE_FACTORS:
        if VISU_SCALING_KIND == "k":
            k_value = BASE_K * factor
            threshold_scale = 1.0
            suffix = f"kscaling_{scale_to_str(factor)}"
        else:
            k_value = BASE_K
            threshold_scale = factor
            suffix = f"thresholdscale_{scale_to_str(factor)}"

        jobs.append(
            make_base_job(
                dataset_name="watkins",
                data_dir=WATKINS_DIR,
                feature_set=FEATURE_SET_WATKINS,
                threshold_rule="visu",
                wavelet=SELECTED_WAVELET,
                shrinkage_rule=SELECTED_SHRINKAGE,
                k=k_value,
                threshold_scale=threshold_scale,
                output_dir=(
                    "results/cv/"
                    f"watkins_visu_{SELECTED_SHRINKAGE}"
                    f"_{suffix}"
                    f"_wavelet_{SELECTED_WAVELET}"
                    f"_mfcc{SELECTED_N_MFCC}"
                    f"_fft{SELECTED_N_FFT}_{SELECTED_WIN_LENGTH}"
                    f"_hop_{SELECTED_HOP_LENGTH}"
                ),
            )
        )
    return jobs

def build_final_swt_jobs() -> list[dict]:
    jobs = []

    datasets = [
        {
            "dataset_name": "watkins",
            "data_dir": WATKINS_DIR,
            "feature_set": FEATURE_SET_WATKINS,
        },
        {
            "dataset_name": "whalefm",
            "data_dir": WHALEFM_DIR,
            "feature_set": FEATURE_SET_WHALEFM,
        },
    ]

    for dataset in datasets:
        for rule in THRESHOLD_RULES_FINAL:
            jobs.append(
                make_base_job(
                    dataset_name=dataset["dataset_name"],
                    data_dir=dataset["data_dir"],
                    feature_set=dataset["feature_set"],
                    threshold_rule=rule,
                    wavelet=SELECTED_WAVELET,
                    shrinkage_rule=SELECTED_SHRINKAGE,
                    k=BASE_K,
                    threshold_scale=1.0,
                    output_dir=(
                        "results/cv/"
                        f"{dataset['dataset_name']}_final_{rule}_{SELECTED_SHRINKAGE}"
                        f"_wavelet_{SELECTED_WAVELET}"
                        f"_mfcc{SELECTED_N_MFCC}"
                        f"_fft{SELECTED_N_FFT}_{SELECTED_WIN_LENGTH}"
                        f"_hop_{SELECTED_HOP_LENGTH}"
                    ),
                )
            )
    return jobs

#main
def main() -> None:
    all_jobs = []
    if RUN_WAVELET_SWEEP:
        all_jobs.append(("Wavelet sweep", build_wavelet_jobs()))
    if RUN_SURE_K_SCALING:
        all_jobs.append(("SURE k-scaling sweep", build_sure_k_scaling_jobs()))
    if RUN_SURE_THRESHOLD_SCALING:
        all_jobs.append(("SURE threshold-scaling sweep", build_sure_threshold_scaling_jobs()))
    if RUN_VISU_SCALING_SANITY:
        all_jobs.append(("Visu scaling sanity sweep", build_visu_scaling_jobs()))
    if RUN_FINAL_SWT_COMPARISON:
        all_jobs.append(("Final SWT-method comparison", build_final_swt_jobs()))

    total_runs = sum(len(jobs) for _, jobs in all_jobs)
    print(f"Planned total runs: {total_runs}")

    for phase_name, jobs in all_jobs:
        run_jobs(jobs, phase_name)

    print("\nDone.")

if __name__ == "__main__":
    main()