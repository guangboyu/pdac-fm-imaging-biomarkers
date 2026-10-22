"""Optuna selector-and-head search with grouped cross-validation.

One Optuna study (TPE, seed 42) jointly samples a feature selector
(PCA, mutual information or ElasticNet) and a classifier head (logistic
regression, RBF SVC, random forest or XGBoost) and maximises the mean
grouped cross-validation AUROC. The selector of the best trial is then
shared by all four heads, each head keeping the hyperparameters of its own
best trial, and every head is refitted fold by fold to produce out-of-fold
probabilities. Scaling, imputation and the selector are fitted inside each
training fold.

The two arms differ only in the folds and in how a cell is summarised:

* ``run_mouse``: the search uses 5-fold GroupKFold over all animals, the
  evaluation uses leave-one-mouse-out, and each head is summarised by its
  pooled out-of-fold AUROC. The reported head is the one with the highest
  out-of-fold AUROC.
* ``run_human``: search and evaluation share one 5-fold
  StratifiedGroupKFold (shuffled, seed 42) grouped by patient. The reported
  head is the head of the best Optuna trial and its AUROC is the mean of
  the five per-fold AUROCs.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import optuna
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

from pdacfm.models.search_space import (
    HEADS,
    SEED,
    Selector,
    build_head,
    head_params_from_trial,
    selector_params,
    suggest_head,
    suggest_selector,
)
from pdacfm.stats.metrics import binary_summary, per_fold_auroc

optuna.logging.set_verbosity(optuna.logging.WARNING)
# scikit-learn 1.8 deprecates the ``penalty`` argument used here; results are unchanged.
warnings.filterwarnings("ignore", module=r"sklearn\.")

N_TRIALS = 100
N_FOLDS = 5


@dataclass
class CVResult:
    """Outcome of one modelling cell."""

    best_head: str
    auroc: float                      # headline AUROC of ``best_head``
    selector: dict                    # shared selector configuration
    head_params: dict                 # {head: hyperparameters}
    oof: pd.DataFrame                 # columns: row, group, fold, head, y_true, y_prob
    per_head: pd.DataFrame            # one row per head
    study: optuna.Study = field(repr=False)


def _mean_group_scores(y_true, y_prob, groups):
    """Average row predictions within each group (one score per animal)."""
    df = pd.DataFrame({"g": groups, "y": y_true, "p": y_prob})
    agg = df.groupby("g", sort=True).agg(y=("y", "first"), p=("p", "mean"))
    return agg["y"].to_numpy(), agg["p"].to_numpy(), agg.index.to_numpy()


def _scale(X_tr, X_te):
    sc = StandardScaler()
    return np.nan_to_num(sc.fit_transform(X_tr)), np.nan_to_num(sc.transform(X_te))


def tune(X: np.ndarray, y: np.ndarray, groups: np.ndarray, splits: list,
         n_trials: int = N_TRIALS, seed: int = SEED,
         aggregate_groups: bool = False) -> optuna.Study:
    """Maximise mean fold AUROC over the joint selector-and-head space.

    ``X`` must not contain missing values. With ``aggregate_groups`` the
    AUROC of a fold is computed after averaging row predictions per group
    (used for the per-week ``weekwise`` representation of the mouse arm).
    """
    def objective(trial):
        params = {**suggest_selector(trial, X.shape[1]), **suggest_head(trial)}
        aucs = []
        for tr, te in splits:
            if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
                continue
            X_tr, X_te = _scale(X[tr], X[te])
            try:
                sel = Selector(params)
                X_tr, X_te = sel.fit_transform(X_tr, y[tr]), sel.transform(X_te)
            except Exception:
                continue
            if X_tr.shape[1] < 1:
                continue
            try:
                clf = build_head(params["head"], params).fit(X_tr, y[tr])
                prob = clf.predict_proba(X_te)[:, 1]
            except Exception:
                continue
            if aggregate_groups:
                yt, yp, _ = _mean_group_scores(y[te], prob, groups[te])
                if len(np.unique(yt)) < 2:
                    continue
                aucs.append(roc_auc_score(yt, yp))
            else:
                aucs.append(roc_auc_score(y[te], prob))
        return float(np.mean(aucs)) if aucs else 0.0

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study


def best_head_params(study: optuna.Study) -> dict:
    """Hyperparameters of the best completed trial for each head."""
    out = {}
    for head in HEADS:
        trials = [t for t in study.trials
                  if t.state == optuna.trial.TrialState.COMPLETE and t.params.get("head") == head]
        if trials:
            best = max(trials, key=lambda t: t.value if t.value is not None else 0.0)
            out[head] = head_params_from_trial(head, best.params)
        else:
            out[head] = {"head": head}
    return out


def predict_folds(X: pd.DataFrame, y: np.ndarray, groups: np.ndarray, splits: list,
                  selector: dict, head_params: dict, impute: bool = True,
                  aggregate_groups: bool = False) -> pd.DataFrame:
    """Refit selector and every head in each training fold; return OOF rows."""
    has_nan = bool(X.isna().any().any())
    rows = []
    for fold, (tr, te) in enumerate(splits, start=1):
        if len(np.unique(y[tr])) < 2:
            continue
        X_tr, X_te = X.iloc[tr].to_numpy(), X.iloc[te].to_numpy()
        if impute and has_nan:
            imp = SimpleImputer(strategy="median")
            X_tr, X_te = imp.fit_transform(X_tr), imp.transform(X_te)
        X_tr, X_te = _scale(X_tr, X_te)
        try:
            sel = Selector(selector)
            X_tr, X_te = sel.fit_transform(X_tr, y[tr]), sel.transform(X_te)
        except Exception:
            continue
        if X_tr.shape[1] < 1:
            continue
        for head in HEADS:
            try:
                clf = build_head(head, head_params[head]).fit(X_tr, y[tr])
                prob = clf.predict_proba(X_te)[:, 1]
            except Exception:
                continue
            if aggregate_groups:
                yt, yp, g = _mean_group_scores(y[te], prob, groups[te])
                idx = np.full(len(g), -1)
            else:
                yt, yp, g, idx = y[te], prob, groups[te], te
            rows += [{"row": int(i), "group": gg, "fold": fold, "head": head,
                      "y_true": int(a), "y_prob": float(b)}
                     for i, gg, a, b in zip(idx, g, yt, yp)]
    return pd.DataFrame(rows)


def _impute_all(X: pd.DataFrame) -> np.ndarray:
    if X.isna().any().any():
        return SimpleImputer(strategy="median").fit_transform(X)
    return X.to_numpy()


def run_mouse(X: pd.DataFrame, y, groups, n_trials: int = N_TRIALS,
              aggregate_groups: bool = False, seed: int = SEED) -> CVResult:
    """Mouse protocol: GroupKFold search, leave-one-mouse-out evaluation.

    Missing values are median-imputed on the full matrix for the search
    only; the leave-one-mouse-out refits impute inside each training fold.
    """
    X = X.reset_index(drop=True)
    y = np.asarray(y).astype(int)
    groups = np.asarray(groups)
    Xg = _impute_all(X)
    n_groups = len(np.unique(groups))
    inner = list(GroupKFold(n_splits=min(N_FOLDS, n_groups)).split(Xg, y, groups=groups))
    study = tune(Xg, y, groups, inner, n_trials, seed, aggregate_groups)

    selector = selector_params(study.best_params)
    heads = best_head_params(study)
    outer = list(LeaveOneGroupOut().split(X, y, groups=groups))
    oof = predict_folds(X, y, groups, outer, selector, heads,
                        impute=True, aggregate_groups=aggregate_groups)

    rows = []
    for head in HEADS:
        sub = oof[oof["head"] == head]
        if sub.empty or sub["y_true"].nunique() < 2:
            rows.append({"head": head, "auroc": np.nan, "_sort": -1})
            continue
        s = binary_summary(sub["y_true"].to_numpy(), sub["y_prob"].to_numpy(), seed=seed)
        rows.append({"head": head, **s, "_sort": s["auroc"]})
    per_head = (pd.DataFrame(rows).sort_values("_sort", ascending=False)
                .drop(columns="_sort").reset_index(drop=True))
    best = per_head.iloc[0]
    return CVResult(best_head=best["head"], auroc=float(best["auroc"]), selector=selector,
                    head_params=heads, oof=oof, per_head=per_head, study=study)


def run_human(X: pd.DataFrame, y, groups, n_trials: int = N_TRIALS,
              seed: int = SEED) -> CVResult:
    """Human protocol: one shuffled StratifiedGroupKFold for search and evaluation."""
    X = X.reset_index(drop=True)
    y = np.asarray(y).astype(int)
    groups = np.asarray(groups)
    X = pd.DataFrame(_impute_all(X), columns=X.columns)
    n_groups = len(np.unique(groups))
    cv = StratifiedGroupKFold(n_splits=min(N_FOLDS, n_groups), shuffle=True, random_state=seed)
    splits = list(cv.split(X.to_numpy(), y, groups=groups))
    study = tune(X.to_numpy(), y, groups, splits, n_trials, seed)

    selector = selector_params(study.best_params)
    heads = best_head_params(study)
    best_head = study.best_params["head"]
    oof = predict_folds(X, y, groups, splits, selector, heads, impute=False)

    rows = []
    for head in HEADS:
        sub = oof[oof["head"] == head]
        if sub.empty:
            rows.append({"head": head, "auroc": np.nan})
            continue
        folds = [(g["y_true"].to_numpy(), g["y_prob"].to_numpy()) for _, g in sub.groupby("fold")]
        pooled = binary_summary(sub["y_true"].to_numpy(), sub["y_prob"].to_numpy(), seed=seed)
        rows.append({"head": head, "auroc": per_fold_auroc(folds),
                     "auroc_pooled": pooled["auroc"]})
    per_head = pd.DataFrame(rows)
    auroc = float(per_head.loc[per_head["head"] == best_head, "auroc"].iloc[0])
    return CVResult(best_head=best_head, auroc=auroc, selector=selector,
                    head_params=heads, oof=oof, per_head=per_head, study=study)
