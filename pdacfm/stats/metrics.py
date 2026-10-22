"""Discrimination, operating-point and continuous-association metrics."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve

N_BOOT = 1000
SEED = 42


def youden_threshold(y_true, y_prob) -> float:
    """Probability cut-off maximising Youden's J (clamped to [0, 1])."""
    if len(np.unique(y_true)) < 2:
        return 0.5
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    return float(min(1.0, max(0.0, thr[int(np.argmax(tpr - fpr))])))


def bootstrap_auroc_ci(y_true, y_prob, n_boot: int = N_BOOT, seed: int = SEED):
    """Percentile 95% CI of AUROC from ``n_boot`` row resamples."""
    y_true, y_prob = np.asarray(y_true), np.asarray(y_prob)
    rng = np.random.default_rng(seed)
    n = len(y_true)
    boot = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        boot.append(roc_auc_score(y_true[idx], y_prob[idx]))
    if not boot:
        return np.nan, np.nan
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def bootstrap_threshold_metric_ci(y_true, y_prob, metric, n_boot: int = N_BOOT,
                                  seed: int = SEED):
    """Percentile 95% CI of a thresholded metric, re-deriving Youden's cut-off per resample."""
    y_true, y_prob = np.asarray(y_true), np.asarray(y_prob)
    rng = np.random.default_rng(seed)
    n = len(y_true)
    boot = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        thr = youden_threshold(y_true[idx], y_prob[idx])
        boot.append(metric(y_true[idx], (y_prob[idx] >= thr).astype(int)))
    if not boot:
        return np.nan, np.nan
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def binary_summary(y_true, y_prob, seed: int = SEED) -> dict:
    """Pooled AUROC with bootstrap CI, plus F1 and accuracy at Youden's cut-off."""
    y_true, y_prob = np.asarray(y_true).astype(int), np.asarray(y_prob).astype(float)
    auc = roc_auc_score(y_true, y_prob)
    lo, hi = bootstrap_auroc_ci(y_true, y_prob, seed=seed)
    thr = youden_threshold(y_true, y_prob)
    pred = (y_prob >= thr).astype(int)
    f1 = lambda a, b: f1_score(a, b, zero_division=0)  # noqa: E731
    return {
        "auroc": float(auc), "auroc_ci_lo": lo, "auroc_ci_hi": hi,
        "youden_threshold": thr,
        "f1": float(f1(y_true, pred)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "f1_ci": bootstrap_threshold_metric_ci(y_true, y_prob, f1, seed=seed),
        "accuracy_ci": bootstrap_threshold_metric_ci(y_true, y_prob, accuracy_score, seed=seed),
        "n": int(len(y_true)),
    }


def per_fold_auroc(folds) -> float:
    """Mean AUROC over folds; folds with a single class are ignored."""
    aucs = [roc_auc_score(yt, yp) if len(np.unique(yt)) > 1 else np.nan for yt, yp in folds]
    return float(np.nanmean(aucs))


def per_fold_bootstrap_ci(y_true, y_prob, fold, rng, n_boot: int = N_BOOT):
    """95% CI of the per-fold mean AUROC, resampling rows (one row per patient).

    Each resample keeps every row's fold assignment and recomputes the mean
    of the per-fold AUROCs. Pass one ``numpy.random.Generator`` and reuse it
    across endpoints to reproduce a sequence of intervals.
    """
    df = pd.DataFrame({"y": np.asarray(y_true), "p": np.asarray(y_prob), "fold": np.asarray(fold)})

    def mean_auc(d):
        aucs = [roc_auc_score(g["y"], g["p"]) for _, g in d.groupby("fold") if g["y"].nunique() > 1]
        return float(np.mean(aucs)) if aucs else np.nan

    idx = np.arange(len(df))
    boots = [mean_auc(df.iloc[rng.choice(idx, size=len(idx), replace=True)]) for _ in range(n_boot)]
    boots = [b for b in boots if not np.isnan(b)]
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def spearman(x, y) -> tuple[float, float]:
    """Spearman rho and two-sided p on finite pairs."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    keep = np.isfinite(x) & np.isfinite(y)
    if keep.sum() < 5:
        return np.nan, np.nan
    res = stats.spearmanr(x[keep], y[keep])
    return float(res.statistic), float(res.pvalue)
