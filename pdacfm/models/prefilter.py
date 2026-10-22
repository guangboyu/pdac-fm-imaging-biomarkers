"""Label-free feature pre-filter.

Both arms drop near-constant features and one member of each highly
correlated pair before model selection. The filter uses no labels and is
applied once to the full feature matrix of each modelling cell.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

VARIANCE_THRESHOLD = 0.01
CORRELATION_THRESHOLD = 0.90


def filter_variance(X: pd.DataFrame, threshold: float = VARIANCE_THRESHOLD) -> pd.DataFrame:
    """Keep columns whose population variance exceeds ``threshold`` (NaN-aware)."""
    variances = X.var(ddof=0, skipna=True)
    return X[variances[variances > threshold].index]


def filter_correlation(X: pd.DataFrame, threshold: float = CORRELATION_THRESHOLD) -> pd.DataFrame:
    """Drop the later column of every pair with |Pearson r| > ``threshold``."""
    if X.shape[1] < 2:
        return X
    corr = X.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = {c for c in upper.columns if (upper[c] > threshold).any()}
    return X[[c for c in X.columns if c not in to_drop]]


def prefilter(X: pd.DataFrame) -> pd.DataFrame:
    """Variance filter followed by correlation filter.

    Falls back to the unfiltered input if either step would remove every
    column.
    """
    X_var = filter_variance(X)
    if X_var.shape[1] == 0:
        return X
    X_corr = filter_correlation(X_var)
    return X_corr if X_corr.shape[1] > 0 else X
