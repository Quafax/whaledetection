import joblib

from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def train_model(X, y,cfg):
    test_size= cfg.svm.test_size
    random_state=cfg.svm.random_state
    kernel = cfg.svm.kernel

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel=kernel, probability=True))
    ])

    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)

    print(classification_report(y_test, preds))

    return pipeline, y_test, preds

def save_model(model, path):

    joblib.dump(model, path)


def load_model(path):

    return joblib.load(path)

def predict(model, features):

    if features.ndim == 1:
        features = features.reshape(1, -1)

    return model.predict(features)