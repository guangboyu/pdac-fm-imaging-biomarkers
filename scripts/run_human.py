"""Fit and evaluate the human molecular endpoints with patient-grouped 5-fold CV.

Uses one scan per patient (portal-venous preferred). The reported AUROC is
the mean of the five per-fold AUROCs; its 95% CI resamples patients with
the folds held fixed (1000 resamples, one generator seeded 42 and shared
across endpoints in the order given). Writes
<results>/human/<endpoint>/summary.json and oof.csv.

    python scripts/run_human.py --endpoint all
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from pdacfm.config import load_configurations, load_paths
from pdacfm.labels.human import ENDPOINTS, attach_embeddings, patient_median_split
from pdacfm.models.classification import run_human
from pdacfm.models.prefilter import prefilter
from pdacfm.stats.metrics import per_fold_bootstrap_ci, spearman


def run_endpoint(name: str, cfg: dict, paths: dict, rng, n_trials: int) -> tuple[dict, pd.DataFrame]:
    embeddings = pd.read_csv(Path(paths["embeddings"]) / f"{cfg['encoder']}_{cfg['pooling']}.csv")
    cohort, features = attach_embeddings(pd.read_csv(paths["scans"]), embeddings)
    column = ENDPOINTS[name]
    label = patient_median_split(cohort, column)
    data = cohort[label.notna()].reset_index(drop=True)
    y = label.dropna().astype(int).to_numpy()

    X = prefilter(data[features])
    result = run_human(X, y, data["patient_id"].to_numpy(), n_trials=n_trials)
    oof = result.oof.rename(columns={"group": "patient_id"})
    best = oof[oof["head"] == result.best_head]
    lo, hi = per_fold_bootstrap_ci(best["y_true"], best["y_prob"], best["fold"], rng)
    rho = spearman(data.loc[best["row"].to_numpy(), column], best["y_prob"])[0]
    return {
        "endpoint": name, **cfg, "n": len(data), "n_features": X.shape[1],
        "best_head": result.best_head, "auroc": result.auroc, "auroc_ci": [lo, hi],
        "spearman": rho, "selector": result.selector, "head_params": result.head_params,
        "per_head": result.per_head.to_dict("records"),
    }, oof


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--endpoint", default="all", choices=["all", *ENDPOINTS])
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--paths")
    args = ap.parse_args()

    paths = load_paths(args.paths)
    configs = load_configurations()["human"]
    names = list(ENDPOINTS) if args.endpoint == "all" else [args.endpoint]
    rng = np.random.default_rng(42)
    for name in names:
        summary, oof = run_endpoint(name, configs[name], paths["human"], rng, args.trials)
        out = Path(paths["results"]) / "human" / name
        out.mkdir(parents=True, exist_ok=True)
        oof.to_csv(out / "oof.csv", index=False)
        (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
        lo, hi = summary["auroc_ci"]
        print(f"{name}: {summary['best_head']} AUROC {summary['auroc']:.3f} [{lo:.3f}, {hi:.3f}] "
              f"rho {summary['spearman']:+.2f} (n={summary['n']})")


if __name__ == "__main__":
    main()
