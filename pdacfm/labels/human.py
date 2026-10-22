"""Human (CPTAC-PDA) cohort selection and endpoints.

The scan table has one row per CT scan with ``scan_id``, ``patient_id``,
``series`` (the series folder name) and patient-level values:
``os_days``, ``os_event``, ``CD8A_prot``, ``COL1A1_prot``, ``CYT`` and
``CD8_macrophage`` (from ``cd8_macrophage_score``). Protein values are the TMT
quantifications of cBioPortal study ``paad_cptac_2021`` (profile
``paad_cptac_2021_protein_quantification``); ``CYT`` is the cytolytic
activity score, the geometric mean of GZMA and PRF1 expression.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PHASE_ORDER = ("VENOUS", "ARTERIAL", "CE_PV_LIKE", "DELAYED", "CE_UNKNOWN")
MACROPHAGE_PROTEINS = ("AIF1", "CSF1R", "C1QA", "C1QB", "CD14", "ITGAM", "MSR1")

ENDPOINTS = {
    "cd8a_protein": "CD8A_prot",
    "col1a1_protein": "COL1A1_prot",
    "cd8_macrophage_protein": "CD8_macrophage",
    "cytolytic_activity": "CYT",
}

_PHASE_KEYS = {
    "ARTERIAL": ["ARTERIAL", "_ART_", "ART_ST", "ART_THINS", "PANCREATIC", "TETNICZA",
                 "CTA_AP", "I_FAZA", "AORTA"],
    "VENOUS": ["VENOUS", "_PV_", "PORTAL", "ZYLNA", "F.ZYLNA", "ZYLNA_F"],
    "DELAYED": ["DELAY", "5_MIN"],
    "CE_PV_LIKE": ["ABDPEL", "AXIAL_ABD", "ABDOPELVIS", "AXIALS", "3.75MM_ABDPELVIS",
                   "KLPJBM3D", "ABDPANC", "ABDOMEN"],
}


def series_phase(series: str) -> str:
    """Contrast phase from the series folder name (first matching rule wins)."""
    name = str(series).upper()
    for phase, keys in _PHASE_KEYS.items():
        if any(k in name for k in keys):
            return phase
    return "CE_UNKNOWN"


def deduplicate(scans: pd.DataFrame) -> pd.DataFrame:
    """Keep one scan per patient, preferring portal-venous series.

    Ties within a phase keep the first scan in table order.
    """
    out = scans.copy()
    out["phase"] = out["series"].map(series_phase)
    rank = {p: i for i, p in enumerate(PHASE_ORDER)}
    out["_rank"] = out["phase"].map(rank)
    out = out.sort_values(["patient_id", "_rank"], kind="mergesort")
    return out.groupby("patient_id", sort=False).head(1).drop(columns="_rank").reset_index(drop=True)


def attach_embeddings(scans: pd.DataFrame, embeddings: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Join scan-level embeddings and keep one scan per patient."""
    features = [c for c in embeddings.columns if c.startswith("emb_")]
    merged = scans.merge(embeddings[["scan_id"] + features], on="scan_id",
                         how="inner", validate="one_to_one")
    return deduplicate(merged), features


def zscore(s: pd.Series) -> pd.Series:
    return (s - s.mean()) / s.std(ddof=0)


def cd8_macrophage_score(protein: pd.DataFrame) -> pd.Series:
    """z(CD8A) minus the mean z-score of seven pan-macrophage proteins.

    ``protein`` has one row per CPTAC sample and one column per gene; the
    z-scores use every sample in the CPTAC-PDA proteome, not only the imaged
    patients. CD163 and MRC1 are excluded so that the composite stays
    pan-macrophage, like CD68 immunohistochemistry.
    """
    macrophage = pd.DataFrame({g: zscore(protein[g]) for g in MACROPHAGE_PROTEINS})
    return zscore(protein["CD8A"]) - macrophage.mean(axis=1, skipna=True)


def patient_median_split(scans: pd.DataFrame, column: str) -> pd.Series:
    """Median split at the patient level, broadcast back to scans.

    The cut-off is the median over patients (not scans) with a value;
    patients at the median are labelled high.
    """
    values = scans.groupby("patient_id")[column].first().dropna()
    cutoff = values.median()
    label = scans["patient_id"].map((values >= cutoff).astype(int))
    return label


def os_binary(scans: pd.DataFrame) -> pd.Series:
    """Overall survival above the median among patients who died.

    Censored patients have no binary label; they enter the time-to-event
    analysis (``pdacfm.models.survival``) instead.
    """
    died = (scans["os_days"] > 0) & (scans["os_event"] == 1)
    sub = scans[died]
    values = sub.groupby("patient_id")["os_days"].first()
    label = pd.Series(np.nan, index=scans.index)
    label[died] = sub["patient_id"].map((values >= values.median()).astype(int))
    return label
