from pathlib import Path

import joblib
import numpy as np
import tensorflow as tf

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def build_model(input_dim: int, num_classes: int, cfg):
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(input_dim,)),
            tf.keras.layers.Dense(cfg.mlp.hidden_1, activation="relu"),
            tf.keras.layers.Dropout(cfg.mlp.dropout),

            tf.keras.layers.Dense(cfg.mlp.hidden_2, activation="relu"),
            tf.keras.layers.Dropout(cfg.mlp.dropout),

            tf.keras.layers.Dense(cfg.mlp.hidden_3, activation="relu"),
            tf.keras.layers.Dropout(cfg.mlp.dropout),

            tf.keras.layers.Dense(cfg.mlp.hidden_4, activation="relu"),
            tf.keras.layers.Dropout(cfg.mlp.dropout),

            tf.keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=cfg.mlp.learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def train_model(X, y, cfg):
    test_size = cfg.mlp.test_size
    random_state = cfg.mlp.random_state

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    num_classes = len(np.unique(y))
    input_dim = X_train_scaled.shape[1]

    model = build_model(input_dim=input_dim, num_classes=num_classes, cfg=cfg)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=cfg.mlp.patience,
            restore_best_weights=True,
        )
    ]

    history = model.fit(
        X_train_scaled,
        y_train,
        validation_data=(X_test_scaled, y_test),
        epochs=cfg.mlp.epochs,
        batch_size=cfg.mlp.batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    probs = model.predict(X_test_scaled, verbose=0)
    preds = np.argmax(probs, axis=1)

    print(classification_report(y_test, preds))

    bundle = {
        "model": model,
        "scaler": scaler,
        "history": history.history,
    }

    return bundle, y_test, preds


def save_model(bundle, model_dir):
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / "mlp_model.keras"
    scaler_path = model_dir / "mlp_scaler.joblib"

    bundle["model"].save(model_path)
    joblib.dump(bundle["scaler"], scaler_path)

    return model_path, scaler_path


def load_model(model_dir):
    model_dir = Path(model_dir)

    model_path = model_dir / "mlp_model.keras"
    scaler_path = model_dir / "mlp_scaler.joblib"

    model = tf.keras.models.load_model(model_path)
    scaler = joblib.load(scaler_path)

    return {
        "model": model,
        "scaler": scaler,
    }


def predict(bundle, features):
    features = np.asarray(features, dtype=np.float32)

    if features.ndim == 1:
        features = features.reshape(1, -1)

    features_scaled = bundle["scaler"].transform(features)
    probs = bundle["model"].predict(features_scaled, verbose=0)
    preds = np.argmax(probs, axis=1)

    return preds