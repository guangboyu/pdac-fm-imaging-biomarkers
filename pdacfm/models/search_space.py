"""Feature selectors, classifier heads and their Optuna search space.

The search space is shared by both arms. Optuna parameter names and the
order of ``suggest_*`` calls are part of the sampler state, so they are kept
exactly as used for the published results.
"""
from __future__ import annotations

from functools import partial

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.linear_model import ElasticNetCV, LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier

SEED = 42
SELECTORS = ("pca", "mi", "elnet")
HEADS = ("LogReg", "SVC", "RF", "XGB")

# Optuna parameter name for each head hyperparameter: {head: [(arg, optuna_name)]}.
HEAD_PARAM_NAMES = {
    "LogReg": [("C", "lr_C"), ("penalty", "lr_penalty")],
    "SVC": [("C", "svm_C"), ("gamma", "svm_gamma")],
    "RF": [("n_estimators", "rf_n_estimators"), ("max_depth", "rf_max_depth"),
           ("min_samples_leaf", "rf_min_leaf")],
    "XGB": [("n_estimators", "xgb_n_estimators"), ("learning_rate", "xgb_lr"),
            ("max_depth", "xgb_max_depth"), ("reg_alpha", "xgb_alpha"),
            ("reg_lambda", "xgb_lambda")],
}


def suggest_selector(trial, n_features: int) -> dict:
    """Sample a selector and its size (PCA components or MI top-k, 5 to 60)."""
    method = trial.suggest_categorical("sel_method", list(SELECTORS))
    params = {"sel_method": method}
    hi = max(1, min(60, n_features))
    lo = min(5, hi)
    if method == "pca":
        params["pca_n"] = trial.suggest_int("pca_n", lo, hi)
    elif method == "mi":
        params["mi_k"] = trial.suggest_int("mi_k", lo, hi)
    return params


def suggest_head(trial) -> dict:
    """Sample a classifier head and its hyperparameters."""
    name = trial.suggest_categorical("head", list(HEADS))
    p = {"head": name}
    if name == "LogReg":
        p["C"] = trial.suggest_float("lr_C", 1e-3, 100.0, log=True)
        p["penalty"] = trial.suggest_categorical("lr_penalty", ["l1", "l2"])
    elif name == "SVC":
        p["C"] = trial.suggest_float("svm_C", 0.01, 100.0, log=True)
        p["gamma"] = trial.suggest_float("svm_gamma", 1e-4, 10.0, log=True)
    elif name == "RF":
        p["n_estimators"] = trial.suggest_int("rf_n_estimators", 50, 500)
        p["max_depth"] = trial.suggest_int("rf_max_depth", 2, 12)
        p["min_samples_leaf"] = trial.suggest_int("rf_min_leaf", 2, 20)
    elif name == "XGB":
        p["n_estimators"] = trial.suggest_int("xgb_n_estimators", 50, 500)
        p["learning_rate"] = trial.suggest_float("xgb_lr", 0.01, 0.3, log=True)
        p["max_depth"] = trial.suggest_int("xgb_max_depth", 2, 8)
        p["reg_alpha"] = trial.suggest_float("xgb_alpha", 1e-6, 1.0, log=True)
        p["reg_lambda"] = trial.suggest_float("xgb_lambda", 1e-6, 1.0, log=True)
    return p


def build_head(name: str, params: dict):
    """Instantiate a classifier head; missing hyperparameters use defaults."""
    if name == "LogReg":
        penalty = params.get("penalty", "l2")
        return LogisticRegression(
            C=params.get("C", 1.0), penalty=penalty,
            solver="liblinear" if penalty == "l1" else "lbfgs",
            max_iter=2000, random_state=SEED)
    if name == "SVC":
        return SVC(kernel="rbf", C=params.get("C", 1.0), gamma=params.get("gamma", "scale"),
                   probability=True, random_state=SEED)
    if name == "RF":
        return RandomForestClassifier(
            n_estimators=params.get("n_estimators", 200),
            max_depth=params.get("max_depth", None),
            min_samples_leaf=params.get("min_samples_leaf", 2),
            class_weight="balanced", random_state=SEED, n_jobs=1)
    if name == "XGB":
        return XGBClassifier(
            n_estimators=params.get("n_estimators", 200),
            learning_rate=params.get("learning_rate", 0.1),
            max_depth=params.get("max_depth", 3),
            reg_alpha=params.get("reg_alpha", 0.0),
            reg_lambda=params.get("reg_lambda", 1.0),
            eval_metric="logloss", random_state=SEED, n_jobs=1, verbosity=0)
    raise ValueError(f"unknown head: {name}")


class Selector:
    """Supervised feature selector fitted on a training fold.

    ``pca``: principal components; ``mi``: top-k mutual information;
    ``elnet``: non-zero ElasticNetCV coefficients (top quartile by |coef| if
    fewer than two survive).
    """

    def __init__(self, params: dict):
        self.method = params["sel_method"]
        self.params = params
        self._model = None
        self._mask = None

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        if self.method == "pca":
            n = max(1, min(self.params["pca_n"], X.shape[0] - 1, X.shape[1]))
            self._model = PCA(n_components=n, random_state=SEED)
            return self._model.fit_transform(X)
        if self.method == "mi":
            k = min(self.params["mi_k"], X.shape[1])
            score = partial(mutual_info_classif, random_state=SEED)
            self._model = SelectKBest(score, k=k).fit(X, y)
            return self._model.transform(X)
        if self.method == "elnet":
            en = ElasticNetCV(cv=3, max_iter=5000, random_state=SEED, n_jobs=1)
            en.fit(X, y.astype(float))
            coef = np.abs(en.coef_)
            mask = coef > 0
            if mask.sum() < 2:
                mask = coef >= np.percentile(coef, 75)
            if mask.sum() < 1:
                mask = np.ones(X.shape[1], dtype=bool)
            self._mask = mask
            return X[:, mask]
        raise ValueError(f"unknown selector: {self.method}")

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.method == "elnet":
            return X[:, self._mask]
        return self._model.transform(X)


def selector_params(best_params: dict) -> dict:
    """Extract the selector part of an Optuna ``best_params`` dict."""
    out = {"sel_method": best_params["sel_method"]}
    for key in ("pca_n", "mi_k"):
        if key in best_params:
            out[key] = best_params[key]
    return out


def head_params_from_trial(head: str, trial_params: dict) -> dict:
    """Map raw Optuna parameter names back to classifier arguments."""
    out = {"head": head}
    for arg, name in HEAD_PARAM_NAMES[head]:
        if name in trial_params:
            out[arg] = trial_params[name]
    return out
