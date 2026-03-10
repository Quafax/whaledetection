import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def train_model(X, y,cfg):
    test_size= cfg.rf.test_size
    random_state=cfg.rf.random_state

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    pipeline = Pipeline([
        ("scaler", StandardScaler()), #useless but for comparability
        ("rf", RandomForestClassifier(
            n_estimators=cfg.rf.estimators,
            max_depth=None,
            random_state=random_state,
            n_jobs=-1 
        ))
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