from pathlib import Path
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from whaledetection.config.config_loader import load_config
from whaledetection.load_dataset import load_dataset
from whaledetection.model.mlp import fit_model as fit_mlp, predict as predict_mlp
from whaledetection.model.random_forest import fit_model as fit_rf, predict as predict_rf
from whaledetection.model.svm import fit_model as fit_svm, predict as predict_svm
from whaledetection.visualizations.plotting import plot_confusion_matrix_seaborn, plot_mlp_history

def save_cv_confusion_matrix(y_true, y_pred, classes, model_name, output_dir):
    plot_confusion_matrix_seaborn(
        y_true,
        y_pred,
        class_names=classes,
        title=None,#f"{model_name.upper()} Confusion Matrix (CV",
        save_path=output_dir / f"{model_name}_cv_confusion_matrix_seaborn.pdf",
        normalize=True,
    )

def evaluate_predictions(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


def main(config_path: str = "configs/config.yaml"):
    cfg = load_config(config_path)
    X, y, classes = load_dataset(cfg)
    print(f"Loaded {len(X)} samples with shape {X.shape}")
    print(f"Classes: {classes}")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    results = {
        "svm": [],
        "rf": [],
        "mlp": [],
    }
    all_preds = {
        "svm": {"y_true": [], "y_pred": []},
        "rf": {"y_true": [], "y_pred": []},
        "mlp": {"y_true": [], "y_pred": []},
    }
    #"results/cv/whalefm/mfcc_no_den"
    output_dir = Path(cfg.experiment.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
        print(f"\n=== Fold {fold} ===")

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        svm_model = fit_svm(X_train, y_train, cfg)
        svm_preds = predict_svm(svm_model, X_test)
        results["svm"].append(evaluate_predictions(y_test, svm_preds))
        all_preds["svm"]["y_true"].extend(y_test.tolist())
        all_preds["svm"]["y_pred"].extend(svm_preds.tolist())

        rf_model = fit_rf(X_train, y_train, cfg)
        rf_preds = predict_rf(rf_model, X_test)
        results["rf"].append(evaluate_predictions(y_test, rf_preds))
        all_preds["rf"]["y_true"].extend(y_test.tolist())
        all_preds["rf"]["y_pred"].extend(rf_preds.tolist())

        X_train_inner, X_val, y_train_inner, y_val = train_test_split(
            X_train,
            y_train,
            test_size=cfg.mlp.test_size,
            random_state=cfg.mlp.random_state,
            stratify=y_train,
        )

        mlp_bundle = fit_mlp(
            X_train_inner,
            y_train_inner,
            cfg,
            X_val=X_val,
            y_val=y_val,
        )
        if fold == 1:
            plot_mlp_history(mlp_bundle["history"], output_dir / "mlp_fold1")

        mlp_preds = predict_mlp(mlp_bundle, X_test)
        results["mlp"].append(evaluate_predictions(y_test, mlp_preds))
        all_preds["mlp"]["y_true"].extend(y_test.tolist())
        all_preds["mlp"]["y_pred"].extend(mlp_preds.tolist())



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
        save_cv_confusion_matrix(
            y_true=np.array(all_preds[model_name]["y_true"]),
            y_pred=np.array(all_preds[model_name]["y_pred"]),
            classes=classes,
            model_name=model_name,
            output_dir=output_dir,
        )


if __name__ == "__main__":
    import sys
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/config.yaml"
    main(config_path)