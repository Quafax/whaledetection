from whaledetection.config.config_loader import load_config
from whaledetection.model.svm import train_model, save_model
from whaledetection.load_dataset import load_dataset
from pathlib import Path
from whaledetection.visualizations.plotting import plot_confusion_matrix,plot_confusion_matrix_seaborn
cfg = load_config("configs/config.yaml")
def main():

    X, y, classes = load_dataset(cfg)

    print(f"Loaded {len(X)} samples")
    print(f"Number of classes: {len(classes)}")
    print(f"Classes: {classes}")
    print(f"Feature shape: {X.shape}")



    model,y_test,preds = train_model(X, y, cfg)
    plot_confusion_matrix(
    y_test,
    preds,
    class_names=classes,
    title="SVM Confusion Matrix",
    save_path="results/svm/svm_confusion_matrix_mfcc_swt.png"
)
    
    plot_confusion_matrix_seaborn(
    y_test,
    preds,
    class_names=classes,
    title="SVM Confusion Matrix (Normalized)",
    save_path="results/svm/svm_confusion_matrix_seaborn_mfcc_swt.png",
    normalize=True,
)
#make sure path exists
    model_path = Path(cfg.svm.model_dir_out)
    if model_path.suffix == "":
        model_path = model_path / "svm_mffc_swt_model.joblib"
    model_path.parent.mkdir(parents=True, exist_ok=True)

    save_model(model, model_path)

    print("Model saved to:",model_path)


if __name__ == "__main__":
    main()