from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold

from whaledetection.config.config_loader import load_config
from whaledetection.load_dataset import load_dataset
from whaledetection.model.mlp import fit_model as fit_mlp, predict as predict_mlp
from whaledetection.model.random_forest import fit_model as fit_rf, predict as predict_rf
from whaledetection.model.svm import fit_model as fit_svm, predict as predict_svm

cfg=load_config("configs/config.yaml")


def evaluate_predictions(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


def main():
    X, y, classes = load_dataset(cfg)
    print(f"Loaded {len(X)} samples with shape {X.shape}")
    print(f"Classes: {classes}")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    results = {
        "svm": [],
        "rf": [],
        "mlp": [],
    }

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
        print(f"\n=== Fold {fold} ===")

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        svm_model = fit_svm(X_train, y_train, cfg)
        svm_preds = predict_svm(svm_model, X_test)
        results["svm"].append(evaluate_predictions(y_test, svm_preds))

        rf_model = fit_rf(X_train, y_train, cfg)
        rf_preds = predict_rf(rf_model, X_test)
        results["rf"].append(evaluate_predictions(y_test, rf_preds))

        mlp_bundle = fit_mlp(X_train, y_train, cfg, X_val=X_test, y_val=y_test)
        mlp_preds = predict_mlp(mlp_bundle, X_test)
        results["mlp"].append(evaluate_predictions(y_test, mlp_preds))

    output_dir = Path("results/cv")
    output_dir.mkdir(parents=True, exist_ok=True)

    for model_name, fold_metrics in results.items():
        print(f"\n===== {model_name.upper()} =====")
        metric_names = fold_metrics[0].keys()

        lines = []
        for metric in metric_names:
            values = [fm[metric] for fm in fold_metrics]
            mean = np.mean(values)
            std = np.std(values)

            line = f"{metric}: mean={mean:.4f}, std={std:.4f}, values={values}"
            print(line)
            lines.append(line)

        out_path = output_dir / f"{model_name}_cv_results.txt"
        out_path.write_text("\n".join(lines), encoding="utf-8")
        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()