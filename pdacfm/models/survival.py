"""Time-to-event model for human overall survival.

All patients enter with censoring. Inside each training fold the features
are standardised and reduced by PCA (2, 3, 4 or 6 components); an Optuna
study (TPE, seed 42, 50 trials) chooses the PCA size and one of Cox
proportional hazards, random survival forest or gradient-boosted survival
by the mean Harrell C over a 5-fold StratifiedGroupKFold (stratified on the
event, grouped by patient). The chosen configuration is refitted fold by
fold; each fold's risk scores are z-scored and pooled into one
out-of-fold risk per patient.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import optuna
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sksurv.ensemble import GradientBoostingSurvivalAnalysis, RandomSurvivalForest
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import concordance_index_censored, concordance_index_ipcw, cumulative_dynamic_auc
from sksurv.util import Surv

optuna.logging.set_verbosity(optuna.logging.WARNING)

SEED = 42
N_FOLDS = 5
N_TRIALS = 50
PCA_CHOICES = (2, 3, 4, 6)
TD_TIMES = (365.0, 730.0)


def suggest_model(trial):
    """Sample a survival model family, its hyperparameters and the PCA size."""
    family = trial.suggest_categorical("family", ["coxph", "rsf", "gbsa"])
    n_pc = trial.suggest_categorical("pca", list(PCA_CHOICES))
    if family == "coxph":
        model = CoxPHSurvivalAnalysis(alpha=trial.suggest_float("coxph_alpha", 1e-2, 1e2, log=True))
    elif family == "rsf":
        model = RandomSurvivalForest(
            n_estimators=trial.suggest_categorical("rsf_n_est", [100, 200, 300]),
            min_samples_leaf=trial.suggest_int("rsf_leaf", 3, 15),
            max_depth=trial.suggest_int("rsf_depth", 2, 4),
            random_state=SEED, n_jobs=1)
    else:
        model = GradientBoostingSurvivalAnalysis(
            n_estimators=trial.suggest_categorical("gbsa_n_est", [100, 200]),
            learning_rate=trial.suggest_float("gbsa_lr", 0.01, 0.2, log=True),
            max_depth=trial.suggest_int("gbsa_depth", 1, 3),
            subsample=0.8, random_state=SEED)
    return n_pc, model


def _fit_predict(model, n_pc, X_tr, y_tr, X_te):
    sc = StandardScaler().fit(X_tr)
    pca = PCA(n_components=min(n_pc, X_tr.shape[1], X_tr.shape[0] - 1),
              random_state=SEED).fit(sc.transform(X_tr))
    model.fit(pca.transform(sc.transform(X_tr)), y_tr)
    return model.predict(pca.transform(sc.transform(X_te)))


def cross_validate(make_model, X: np.ndarray, y, groups, event):
    """Per-fold Harrell C and pooled out-of-fold risk (z-scored within fold)."""
    cv = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    per_fold, risk = [], np.full(len(X), np.nan)
    for tr, te in cv.split(X, event, groups):
        n_pc, model = make_model()
        try:
            r = _fit_predict(model, n_pc, X[tr], y[tr], X[te])
        except Exception:
            return None, None
        per_fold.append(concordance_index_censored(event[te].astype(bool), y["time"][te], r)[0])
        risk[te] = (r - r.mean()) / (r.std() + 1e-9)
    return per_fold, risk


def bootstrap_c_ci(event, time, risk, groups, n_boot: int = 1000, seed: int = SEED):
    """95% CI of Harrell C resampling patients."""
    rng = np.random.default_rng(seed)
    units = np.unique(groups)
    index = {u: np.where(groups == u)[0] for u in units}
    boots = []
    for _ in range(n_boot):
        idx = np.concatenate([index[u] for u in rng.choice(units, len(units), replace=True)])
        ev = event[idx].astype(bool)
        if ev.sum() < 2 or (~ev).sum() < 1:
            continue
        try:
            boots.append(concordance_index_censored(ev, time[idx], risk[idx])[0])
        except Exception:
            pass
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


@dataclass
class SurvivalResult:
    params: dict
    c_index: float          # pooled out-of-fold Harrell C
    c_index_ci: tuple
    c_index_per_fold: float
    c_uno: float
    td_auc: dict
    risk: np.ndarray


def search(X: np.ndarray, y, groups, event, n_trials: int = N_TRIALS) -> dict:
    """Optuna search over model family, hyperparameters and PCA size."""
    def objective(trial):
        per_fold, _ = cross_validate(lambda: suggest_model(trial), X, y, groups, event)
        return -1.0 if per_fold is None else float(np.mean(per_fold))

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def run(X: pd.DataFrame, time, event, groups, params: dict | None = None,
        n_trials: int = N_TRIALS) -> SurvivalResult:
    """Cross-validate ``params`` (Optuna parameter names), or search for them if None."""
    X = X.to_numpy(float)
    time, event, groups = np.asarray(time, float), np.asarray(event, int), np.asarray(groups)
    y = Surv.from_arrays(event=event.astype(bool), time=time)
    best = params or search(X, y, groups, event, n_trials)

    per_fold, risk = cross_validate(lambda: suggest_model(optuna.trial.FixedTrial(best)),
                                    X, y, groups, event)
    ok = np.isfinite(risk)
    c = concordance_index_censored(event[ok].astype(bool), time[ok], risk[ok])[0]
    tau = float(np.percentile(time[event == 1], 90))
    c_uno = concordance_index_ipcw(y[ok], y[ok], risk[ok], tau=tau)[0]
    auc, _ = cumulative_dynamic_auc(y[ok], y[ok], risk[ok], list(TD_TIMES))
    return SurvivalResult(
        params=best, c_index=float(c),
        c_index_ci=bootstrap_c_ci(event[ok], time[ok], risk[ok], groups[ok]),
        c_index_per_fold=float(np.mean(per_fold)), c_uno=float(c_uno),
        td_auc={int(t): float(a) for t, a in zip(TD_TIMES, np.atleast_1d(auc))},
        risk=risk)
