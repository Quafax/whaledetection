from pathlib import Path

from whaledetection.config.config_loader import load_config
from whaledetection.model.mlp import train_model, save_model
from whaledetection.load_dataset import load_dataset
from whaledetection.visualizations.plotting import (
    plot_confusion_matrix,
    plot_confusion_matrix_seaborn,
)

cfg = load_config("configs/config.yaml")


def main():
    X, y, classes = load_dataset(cfg)

    print(f"Loaded {len(X)} samples")
    print(f"Number of classes: {len(classes)}")
    print(f"Classes: {classes}")
    print(f"Feature shape: {X.shape}")

    bundle, y_test, preds = train_model(X, y, cfg)
    """
    plot_confusion_matrix(
        y_test,
        preds,
        class_names=classes,
        title="Mlp Neural Network Confusion Matrix",
        save_path="results/mlp/mlp_confusion_matrix.png",
    )

    plot_confusion_matrix_seaborn(
        y_test,
        preds,
        class_names=classes,
        title="Mlp Neural Network Confusion Matrix (Normalized)",
        save_path="results/mlp/mlp_confusion_matrix_seaborn.png",
        normalize=True,
    )
    """
    model_dir = Path(cfg.mlp.model_dir_out)
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path, scaler_path = save_model(bundle, model_dir)

    print("Model saved to:", model_path)
    print("Scaler saved to:", scaler_path)


if __name__ == "__main__":
    main()