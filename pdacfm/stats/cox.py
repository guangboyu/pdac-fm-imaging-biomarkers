"""Multivariable Cox analysis of the human imaging risk score.

The out-of-fold risk score from ``pdacfm.models.survival`` is entered with
age, AJCC stage (ordinal I to IV) and tumour grade (ordinal G1 to G3); all
four are standardised so hazard ratios are per standard deviation. The
three models are fitted on the same complete-case set, so their
concordance indices are apparent (same-cohort) values.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from scipy import stats

CLINICAL = ["age", "stage", "grade"]


def stage_ordinal(stage) -> float:
    """'Stage IIB' -> 2; unknown or 'Not reported' -> NaN."""
    if not isinstance(stage, str) or "Not" in stage:
        return np.nan
    m = re.search(r"Stage (IV|III|II|I)", stage)
    return {"I": 1, "II": 2, "III": 3, "IV": 4}.get(m.group(1)) if m else np.nan


def grade_ordinal(grade) -> float:
    return {"G1": 1, "G2": 2, "G3": 3}.get(grade, np.nan)


def multivariable_cox(df: pd.DataFrame) -> dict:
    """``df`` has ``time``, ``event``, ``risk``, ``age``, ``stage`` and ``grade``
    (ordinal), one row per patient."""
    cols = CLINICAL + ["risk"]
    df = df.copy()
    for c in cols:
        df[c] = (df[c] - df[c].mean()) / df[c].std()
    data = df[["time", "event"] + cols].dropna()

    fit = lambda covs: CoxPHFitter().fit(data[["time", "event"] + covs], "time", "event")  # noqa: E731
    clinical, imaging, combined = fit(CLINICAL), fit(["risk"]), fit(cols)

    lr = 2 * (combined.log_likelihood_ - clinical.log_likelihood_)
    s = combined.summary.loc["risk"]
    return {
        "n": len(data), "events": int(data["event"].sum()),
        "c_clinical": clinical.concordance_index_,
        "c_imaging": imaging.concordance_index_,
        "c_combined": combined.concordance_index_,
        "hr": float(np.exp(s["coef"])),
        "hr_ci": (float(np.exp(s["coef lower 95%"])), float(np.exp(s["coef upper 95%"]))),
        "p": float(s["p"]),
        "lr_chi2": float(lr), "lr_p": float(stats.chi2.sf(lr, df=1)),
        "summary": combined.summary,
    }
