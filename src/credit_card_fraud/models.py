"""Model zoo: two supervised families x four ways of handling imbalance, plus an unsupervised baseline."""
import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from lightgbm import LGBMClassifier
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

IMBALANCE = ["none", "class_weight", "smote", "undersample"]


def resample(X, y, how: str, seed: int):
    if how == "smote":
        return SMOTE(random_state=seed).fit_resample(X, y)
    if how == "undersample":  # keep every fraud, 10 legit per fraud
        return RandomUnderSampler(sampling_strategy=0.1, random_state=seed).fit_resample(X, y)
    return X, y


def make(family: str, how: str, seed: int):
    weight = "balanced" if how == "class_weight" else None
    if family == "logreg":
        return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000, class_weight=weight))
    if family == "lightgbm":
        return LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=31, min_child_samples=50,
                              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                              class_weight=weight, random_state=seed, n_jobs=4, verbose=-1)
    raise ValueError(family)


def fit_score(family: str, how: str, seed: int, X_tr, y_tr, X_ev) -> np.ndarray:
    """Fit on (possibly resampled) training data and return fraud scores for X_ev."""
    if family == "isolation_forest":
        m = IsolationForest(n_estimators=300, random_state=seed, n_jobs=4).fit(X_tr)
        return -m.score_samples(X_ev)
    Xr, yr = resample(X_tr, y_tr, how, seed)
    m = make(family, how, seed).fit(Xr, yr)
    return m.predict_proba(X_ev)[:, 1]
