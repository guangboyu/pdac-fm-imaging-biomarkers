"""Aggregation of weekly mouse embeddings into one feature row per animal.

Input is the long table written by ``scripts/extract_mouse_embeddings.py``:
one row per (animal, week) with stream blocks ``s0_emb_*`` (T2w),
``s1_emb_*`` (T1w) and, when ADC is used, ``adc_emb_*`` plus
``adc_week_missing``. Every representation returns ``mouse_id``, numbered
feature columns ``emb_*`` and the indicator ``_adc_missing`` (1 when the
animal has no ADC at any week).

* ``absolute_pooled``: per-stream mean over the available weeks
* ``moments``: per-stream mean and standard deviation over weeks
* ``delta``: week 4 minus week 2
* ``slope``: least-squares slope against week number
* ``concat_weeks``: weeks 2, 3 and 4 side by side, with ``_weekN_missing``
  flags for absent weeks (their features are left missing)

``absolute_pooled``, ``moments`` and ``slope`` use every available
post-treatment week (weeks 2 to 5).
"""
from __future__ import annotations

import pandas as pd

STREAMS = ("s0_emb_", "s1_emb_", "adc_emb_")
REPRESENTATIONS = ("absolute_pooled", "moments", "delta", "slope", "concat_weeks")


def stream_columns(long_df: pd.DataFrame) -> list[str]:
    """Embedding columns ordered by stream (T2w, T1w, ADC), then by index."""
    cols = []
    for prefix in STREAMS:
        block = [c for c in long_df.columns if c.startswith(prefix)]
        cols += sorted(block, key=lambda c: int(c.rsplit("_", 1)[1]))
    return cols


def adc_missing(long_df: pd.DataFrame) -> pd.Series:
    if "adc_week_missing" not in long_df.columns:
        return pd.Series(0, index=pd.Index(long_df["mouse_id"].unique()))
    return long_df.groupby("mouse_id")["adc_week_missing"].min().astype(int)


def _finish(X: pd.DataFrame, long_df: pd.DataFrame) -> pd.DataFrame:
    """Number feature columns and attach ``mouse_id`` and ``_adc_missing``."""
    X = X.copy()
    X.columns = [f"emb_{i}" for i in range(X.shape[1])]
    X["mouse_id"] = X.index
    X["_adc_missing"] = adc_missing(long_df).reindex(X.index).fillna(0).astype(int).values
    return X.reset_index(drop=True)


def absolute_pooled(long_df: pd.DataFrame) -> pd.DataFrame:
    g = long_df.groupby("mouse_id")
    return _finish(g[stream_columns(long_df)].mean(), long_df)


def moments(long_df: pd.DataFrame) -> pd.DataFrame:
    cols = stream_columns(long_df)
    g = long_df.groupby("mouse_id")
    return _finish(pd.concat([g[cols].mean(), g[cols].std(ddof=0).fillna(0.0)], axis=1), long_df)


def delta(long_df: pd.DataFrame, early: str = "week2", late: str = "week4") -> pd.DataFrame:
    cols = stream_columns(long_df)
    e = long_df[long_df["week"] == early].groupby("mouse_id")[cols].mean()
    l_ = long_df[long_df["week"] == late].groupby("mouse_id")[cols].mean()
    common = e.index.intersection(l_.index)
    return _finish(l_.loc[common] - e.loc[common], long_df)


def slope(long_df: pd.DataFrame) -> pd.DataFrame:
    cols = stream_columns(long_df)
    rows, ids = [], []
    for mid, m in long_df.groupby("mouse_id", sort=True):
        if m["week"].nunique() < 2:
            continue
        w = m["week"].str.replace("week", "").astype(float).to_numpy()
        wc = w - w.mean()
        F = m[cols].to_numpy(float)
        rows.append((wc[:, None] * (F - F.mean(axis=0))).sum(axis=0) / (wc ** 2).sum())
        ids.append(mid)
    return _finish(pd.DataFrame(rows, index=ids), long_df)


def concat_weeks(long_df: pd.DataFrame, weeks=("week2", "week3", "week4")) -> pd.DataFrame:
    cols = stream_columns(long_df)
    mice = long_df["mouse_id"].unique()
    blocks, flags = [], {}
    for wk in weeks:
        sub = long_df[long_df["week"] == wk].set_index("mouse_id")
        blocks.append(sub.reindex(mice)[cols].set_axis([f"{wk}_{c}" for c in cols], axis=1))
        flags[f"_{wk}_missing"] = (~pd.Index(mice).isin(sub.index)).astype(int)
    X = _finish(pd.concat(blocks, axis=1), long_df)
    for name, values in flags.items():
        X[name] = values
    return X


def derive(long_df: pd.DataFrame, representation: str) -> pd.DataFrame:
    try:
        return globals()[representation](long_df)
    except KeyError:
        raise ValueError(f"unknown representation: {representation}") from None


def feature_columns(X: pd.DataFrame) -> list[str]:
    """Model inputs: embedding columns followed by the missingness indicators."""
    flags = [c for c in ("_adc_missing", "_week2_missing", "_week3_missing", "_week4_missing")
             if c in X.columns]
    return [c for c in X.columns if c.startswith("emb_")] + flags
