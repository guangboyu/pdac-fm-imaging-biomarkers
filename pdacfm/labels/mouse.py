"""Mouse endpoints from the per-animal pathology and survival table.

Expected columns (one row per animal): ``mouse_id``, ``cd8_density`` and
``cd68_density`` (positive cells per mm2), ``cd68_pct_positive``,
``ki67_pct_positive``, ``collagen_frac`` (trichrome), ``necrosis_frac``
(H&E), ``survival_days`` and ``euthanised`` (1 for animals euthanised for
tissue collection rather than followed to the survival endpoint).

Each continuous endpoint is split at its cohort median (values at the
median are labelled high); animals without a value are left out.
"""
from __future__ import annotations

import pandas as pd

ENDPOINTS = ("os", "cd8", "collagen", "cd8_cd68", "treatment_response")


def zscore(s: pd.Series) -> pd.Series:
    """Population z-score over the non-missing values."""
    s = s.astype(float)
    sd = s.std(ddof=0)
    return s * 0.0 if not sd or pd.isna(sd) else (s - s.mean()) / sd


def continuous_endpoints(df: pd.DataFrame) -> pd.DataFrame:
    """Continuous value of every endpoint, indexed by ``mouse_id``.

    CD68 records with a positive-cell percentage above 100% are set to
    missing before any derived value is computed.
    """
    df = df.set_index("mouse_id").copy()
    bad = df["cd68_pct_positive"] > 100
    df.loc[bad, ["cd68_density", "cd68_pct_positive"]] = float("nan")

    out = pd.DataFrame(index=df.index)
    out["cd8"] = df["cd8_density"]
    out["collagen"] = df["collagen_frac"]
    out["cd8_cd68"] = df["cd8_density"] / df["cd68_density"].clip(lower=0.1)
    out["treatment_response"] = (zscore(df["necrosis_frac"]) + zscore(df["collagen_frac"])
                                 + zscore(df["cd68_density"]) - zscore(df["ki67_pct_positive"]))
    out["os"] = df["survival_days"].where(df["euthanised"] == 0)
    return out


def median_split(values: pd.Series) -> pd.Series:
    """1 if value >= cohort median, 0 otherwise; missing values dropped."""
    v = values.dropna()
    return (v >= v.median()).astype(int)


def binary_endpoints(df: pd.DataFrame) -> dict[str, pd.Series]:
    cont = continuous_endpoints(df)
    return {name: median_split(cont[name]) for name in ENDPOINTS}
