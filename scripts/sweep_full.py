import csv
import sys
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path
import subprocess

import yaml


# -----------------------------
# Paths
# -----------------------------
BASE_CONFIG_PATH = Path("configs/config.yaml")
TMP_CONFIG_PATH = Path("configs/_overnight_tmp.yaml")
MANIFEST_PATH = Path("results/cv/_overnight_manifest.csv")

WATKINS_DIR = r"C:/Users/luede/Seafile/WhaleData"
WHALEFM_DIR = r"C:/Users/luede/Seafile/WhaleFM"


# -----------------------------
# Toggle what should run tonight
# -----------------------------
RUN_FINAL_METHOD_COMPARISON = True
RUN_FEATURE_ABLATION = True
RUN_SCALING_GRID = True   # turn on only if you really want a very large night batch
SKIP_COMPLETED = True
SLEEP_BETWEEN_RUNS_SEC = 2


# -----------------------------
# Chosen reference settings
# -----------------------------
SELECTED_N_MFCC = 40
SELECTED_N_FFT = 2048
SELECTED_WIN_LENGTH = 0.025
SELECTED_WAVELET = "db4"
SELECTED_SHRINKAGE = "soft"

# Keep hop length from base config unless you want to override it here.
OVERRIDE_HOP_LENGTH = None  # e.g. 0.01

# Final dataset-specific feature choices
WATKINS_FEATURE_SELECTED = ["mfcc"]
WHALEFM_FEATURE_SELECTED = ["mfcc", "delta", "delta2"]

# Optional feature-set ablation candidates
FEATURE_OPTIONS = {
    "watkins": [
        ("mfcc", ["mfcc"]),
        ("mfcc_delta_delta2", ["mfcc", "delta", "delta2"]),
    ],
    "whalefm": [
        ("mfcc", ["mfcc"]),
        ("mfcc_delta_delta2", ["mfcc", "delta", "delta2"]),
    ],
}

# SWT rules to compare in the final denoising comparison
SWT_RULES = ["sure", "visu", "bayes", "percentile"]

# Optional big scaling grid
SCALING_RULES = ["sure", "visu", "bayes", "percentile"]
K_SCALES = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
THRESHOLD_SCALES = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def ensure_manifest_header() -> None:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not MANIFEST_PATH.exists():
        with MANIFEST_PATH.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "timestamp",
                    "status",
                    "job_name",
                    "dataset",
                    "feature_tag",
                    "denoise_method",
                    "threshold_rule",
                    "wavelet",
                    "k",
                    "threshold_scale",
                    "output_dir",
                ]
            )


def append_manifest(status: str, job: dict) -> None:
    ensure_manifest_header()
    with MANIFEST_PATH.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                datetime.now().isoformat(timespec="seconds"),
                status,
                job.get("job_name", ""),
                job.get("dataset_name", ""),
                job.get("feature_tag", ""),
                job.get("denoise_method", ""),
                job.get("threshold_rule", ""),
                job.get("wavelet", ""),
                job.get("k", ""),
                job.get("threshold_scale", ""),
                job.get("output_dir", ""),
            ]
        )


def results_complete(output_dir: str) -> bool:
    out = Path(output_dir)
    required = [
        out / "svm_cv_results.txt",
        out / "rf_cv_results.txt",
        out / "mlp_cv_results.txt",
    ]
    return all(p.exists() for p in required)


def format_scale(x: float) -> str:
    return str(x).replace(".", "_")


def make_base_job(
    dataset_name: str,
    data_dir: str,
    feature_tag: str,
    feature_set: list[str],
    base_cfg: dict,
) -> dict:
    hop_length = (
        OVERRIDE_HOP_LENGTH
        if OVERRIDE_HOP_LENGTH is not None
        else base_cfg["mfcc"].get("hop_length")
    )
    return {
        "dataset_name": dataset_name,
        "data_dir": data_dir,
        "feature_tag": feature_tag,
        "feature_set": feature_set,
        "n_mfcc": SELECTED_N_MFCC,
        "n_fft": SELECTED_N_FFT,
        "win_length": SELECTED_WIN_LENGTH,
        "hop_length": hop_length,
    }


def build_jobs(base_cfg: dict) -> list[dict]:
    jobs: list[dict] = []

    datasets = [
        {
            "name": "watkins",
            "data_dir": WATKINS_DIR,
            "selected_feature_tag": "mfcc",
            "selected_feature_set": WATKINS_FEATURE_SELECTED,
        },
        {
            "name": "whalefm",
            "data_dir": WHALEFM_DIR,
            "selected_feature_tag": "mfcc_delta_delta2",
            "selected_feature_set": WHALEFM_FEATURE_SELECTED,
        },
    ]

    base_k = float(base_cfg["swt"].get("k", 1.4826))

    # -------------------------------------------------
    # 1) Final method comparison for thesis tables
    # -------------------------------------------------
    if RUN_FINAL_METHOD_COMPARISON:
        for ds in datasets:
            job_base = make_base_job(
                dataset_name=ds["name"],
                data_dir=ds["data_dir"],
                feature_tag=ds["selected_feature_tag"],
                feature_set=ds["selected_feature_set"],
                base_cfg=base_cfg,
            )
            root_tag = (
                f"{ds['name']}_{ds['selected_feature_tag']}"
                f"_nmfcc_{SELECTED_N_MFCC}"
                f"_fft_{SELECTED_N_FFT}"
                f"_win_{format_scale(SELECTED_WIN_LENGTH)}"
                f"_wavelet_{SELECTED_WAVELET}"
            )

            # raw / none
            jobs.append(
                {
                    **job_base,
                    "job_name": f"{ds['name']} | {ds['selected_feature_tag']} | raw",
                    "denoise_method": "none",
                    "output_dir": f"results/cv/{root_tag}_raw",
                }
            )

            # noisereduce / spectral-gating baseline
            jobs.append(
                {
                    **job_base,
                    "job_name": f"{ds['name']} | {ds['selected_feature_tag']} | noisereduce",
                    "denoise_method": "noisereduce",
                    "output_dir": f"results/cv/{root_tag}_noisereduce",
                }
            )

            # SWT rules with common reference scaling
            for rule in SWT_RULES:
                jobs.append(
                    {
                        **job_base,
                        "job_name": f"{ds['name']} | {ds['selected_feature_tag']} | {rule} | soft | ref",
                        "denoise_method": "swt",
                        "threshold_rule": rule,
                        "shrinkage_rule": SELECTED_SHRINKAGE,
                        "wavelet": SELECTED_WAVELET,
                        "k": base_k,
                        "threshold_scale": 1.0,
                        "output_dir": f"results/cv/{root_tag}_{rule}_soft_ref",
                    }
                )

    # -------------------------------------------------
    # 2) Feature-set ablation, only where useful
    # -------------------------------------------------
    if RUN_FEATURE_ABLATION:
        # Watkins: raw + sure are enough for the feature-set decision
        for feature_tag, feature_set in FEATURE_OPTIONS["watkins"]:
            job_base = make_base_job(
                dataset_name="watkins",
                data_dir=WATKINS_DIR,
                feature_tag=feature_tag,
                feature_set=feature_set,
                base_cfg=base_cfg,
            )
            root_tag = (
                f"watkins_{feature_tag}"
                f"_nmfcc_{SELECTED_N_MFCC}"
                f"_fft_{SELECTED_N_FFT}"
                f"_win_{format_scale(SELECTED_WIN_LENGTH)}"
                f"_wavelet_{SELECTED_WAVELET}"
            )
            jobs.append(
                {
                    **job_base,
                    "job_name": f"watkins | {feature_tag} | raw_feature_ablation",
                    "denoise_method": "none",
                    "output_dir": f"results/cv/{root_tag}_raw_feature_ablation",
                }
            )
            jobs.append(
                {
                    **job_base,
                    "job_name": f"watkins | {feature_tag} | sure_feature_ablation",
                    "denoise_method": "swt",
                    "threshold_rule": "sure",
                    "shrinkage_rule": SELECTED_SHRINKAGE,
                    "wavelet": SELECTED_WAVELET,
                    "k": base_k,
                    "threshold_scale": 1.0,
                    "output_dir": f"results/cv/{root_tag}_sure_soft_feature_ablation",
                }
            )

        # WhaleFM: run a slightly broader check
        for feature_tag, feature_set in FEATURE_OPTIONS["whalefm"]:
            job_base = make_base_job(
                dataset_name="whalefm",
                data_dir=WHALEFM_DIR,
                feature_tag=feature_tag,
                feature_set=feature_set,
                base_cfg=base_cfg,
            )
            root_tag = (
                f"whalefm_{feature_tag}"
                f"_nmfcc_{SELECTED_N_MFCC}"
                f"_fft_{SELECTED_N_FFT}"
                f"_win_{format_scale(SELECTED_WIN_LENGTH)}"
                f"_wavelet_{SELECTED_WAVELET}"
            )
            # raw
            jobs.append(
                {
                    **job_base,
                    "job_name": f"whalefm | {feature_tag} | raw_feature_ablation",
                    "denoise_method": "none",
                    "output_dir": f"results/cv/{root_tag}_raw_feature_ablation",
                }
            )
            # sure
            jobs.append(
                {
                    **job_base,
                    "job_name": f"whalefm | {feature_tag} | sure_feature_ablation",
                    "denoise_method": "swt",
                    "threshold_rule": "sure",
                    "shrinkage_rule": SELECTED_SHRINKAGE,
                    "wavelet": SELECTED_WAVELET,
                    "k": base_k,
                    "threshold_scale": 1.0,
                    "output_dir": f"results/cv/{root_tag}_sure_soft_feature_ablation",
                }
            )
            # noisereduce
            jobs.append(
                {
                    **job_base,
                    "job_name": f"whalefm | {feature_tag} | noisereduce_feature_ablation",
                    "denoise_method": "noisereduce",
                    "output_dir": f"results/cv/{root_tag}_noisereduce_feature_ablation",
                }
            )

    # -------------------------------------------------
    # 3) Optional huge scaling grid
    # -------------------------------------------------
    if RUN_SCALING_GRID:
        for ds in datasets:
            job_base = make_base_job(
                dataset_name=ds["name"],
                data_dir=ds["data_dir"],
                feature_tag=ds["selected_feature_tag"],
                feature_set=ds["selected_feature_set"],
                base_cfg=base_cfg,
            )
            root_tag = (
                f"{ds['name']}_{ds['selected_feature_tag']}"
                f"_nmfcc_{SELECTED_N_MFCC}"
                f"_fft_{SELECTED_N_FFT}"
                f"_win_{format_scale(SELECTED_WIN_LENGTH)}"
                f"_wavelet_{SELECTED_WAVELET}"
            )

            for rule in SCALING_RULES:
                for scale in K_SCALES:
                    jobs.append(
                        {
                            **job_base,
                            "job_name": f"{ds['name']} | {rule} | soft | kscale={scale}",
                            "denoise_method": "swt",
                            "threshold_rule": rule,
                            "shrinkage_rule": SELECTED_SHRINKAGE,
                            "wavelet": SELECTED_WAVELET,
                            "k": base_k * scale,
                            "threshold_scale": 1.0,
                            "output_dir": (
                                f"results/cv/{root_tag}_{rule}_soft_kscaling_{format_scale(scale)}"
                            ),
                        }
                    )

                for scale in THRESHOLD_SCALES:
                    jobs.append(
                        {
                            **job_base,
                            "job_name": f"{ds['name']} | {rule} | soft | thresholdscale={scale}",
                            "denoise_method": "swt",
                            "threshold_rule": rule,
                            "shrinkage_rule": SELECTED_SHRINKAGE,
                            "wavelet": SELECTED_WAVELET,
                            "k": base_k,
                            "threshold_scale": scale,
                            "output_dir": (
                                f"results/cv/{root_tag}_{rule}_soft_thresholdscale_{format_scale(scale)}"
                            ),
                        }
                    )

    return jobs


def apply_job_to_config(base_cfg: dict, job: dict) -> dict:
    cfg = deepcopy(base_cfg)

    cfg["loadSignal"]["signal_base_dir_in"] = job["data_dir"]

    cfg["mfcc"]["feature_set"] = job["feature_set"]
    cfg["mfcc"]["n_mfcc"] = job["n_mfcc"]
    cfg["mfcc"]["n_fft"] = job["n_fft"]
    cfg["mfcc"]["win_length"] = job["win_length"]

    if job.get("hop_length") is not None:
        cfg["mfcc"]["hop_length"] = job["hop_length"]

    cfg["denoise"]["method"] = job["denoise_method"]
    cfg["experiment"]["output_dir"] = job["output_dir"]

    if job["denoise_method"] == "swt":
        cfg["swt"]["t_mode"] = job["threshold_rule"]
        cfg["swt"]["t_meth"] = job["shrinkage_rule"]
        cfg["swt"]["wavelet"] = job["wavelet"]
        cfg["swt"]["k"] = job["k"]
        cfg["swt"]["threshold_scale"] = job["threshold_scale"]

    return cfg


def run_job(base_cfg: dict, job: dict) -> None:
    output_dir = job["output_dir"]

    if SKIP_COMPLETED and results_complete(output_dir):
        print(f"\n[SKIP] {job['job_name']}")
        print(f"       {output_dir}")
        append_manifest("SKIPPED", job)
        return

    cfg = apply_job_to_config(base_cfg, job)
    save_yaml(TMP_CONFIG_PATH, cfg)

    print(f"\n=== Running: {job['job_name']} ===")
    print(f"Output: {output_dir}")

    try:
        subprocess.run(
            [sys.executable, "scripts/cv_evaluate.py", str(TMP_CONFIG_PATH)],
            check=True,
        )
        append_manifest("OK", job)
    except subprocess.CalledProcessError:
        append_manifest("FAILED", job)
        raise


def main() -> None:
    base_cfg = load_yaml(BASE_CONFIG_PATH)
    jobs = build_jobs(base_cfg)

    print(f"Prepared {len(jobs)} jobs.")
    print(f"Manifest: {MANIFEST_PATH}")
    print(f"Skip completed: {SKIP_COMPLETED}")
    print(f"Scaling grid enabled: {RUN_SCALING_GRID}")

    for i, job in enumerate(jobs, start=1):
        print(f"\n[{i}/{len(jobs)}]")
        run_job(base_cfg, job)
        time.sleep(SLEEP_BETWEEN_RUNS_SEC)

    print("\nAll overnight jobs finished.")


if __name__ == "__main__":
    main()