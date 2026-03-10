from pathlib import Path

from whaledetection.config.config_loader import load_config
from whaledetection.model.random_forest import train_model, save_model
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

    model, y_test, preds = train_model(X, y, cfg)

    plot_confusion_matrix(
        y_test,
        preds,
        class_names=classes,
        title="Random Forest Confusion Matrix",
        save_path="results/rf/rf_confusion_matrix.png",
    )

    plot_confusion_matrix_seaborn(
        y_test,
        preds,
        class_names=classes,
        title="Random Forest Confusion Matrix (Normalized)",
        save_path="results/rf/rf_confusion_matrix_seaborn.png",
        normalize=True,
    )

    model_path = Path("models/rf/random_forest_model.joblib")
    model_path.parent.mkdir(parents=True, exist_ok=True)

    save_model(model, model_path)

    print("Model saved to:", model_path)


if __name__ == "__main__":
    main()