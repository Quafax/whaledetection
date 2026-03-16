import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def build_model(cfg):
    return Pipeline([("scaler", StandardScaler()),
                     ("rf", RandomForestClassifier(n_estimators=cfg.rf.estimators,
                                                   max_depth=None,
                                                   random_state=cfg.rf.random_state,
                                                   n_jobs=-1,),),])

def fit_model(X_train,y_train,cfg):
    model=build_model(cfg)
    model.fit(X_train,y_train)
    return model

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

    model = fit_model(X_train,y_train,cfg)
    preds = model.predict(X_test)

    print(classification_report(y_test, preds))

    return model, y_test, preds

def save_model(model, path):

    joblib.dump(model, path)


def load_model(path):

    return joblib.load(path)

def predict(model, features):

    if features.ndim == 1:
        features = features.reshape(1, -1)

    return model.predict(features)