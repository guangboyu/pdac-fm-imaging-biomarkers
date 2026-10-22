"""Human overall survival: cross-validated risk score and multivariable Cox model.

Uses one scan per patient and every patient with follow-up (censored
included). By default the published model configuration in
configs/endpoints.yaml is cross-validated; --search reruns the Optuna search
instead. Writes <results>/human/os/summary.json and risk.csv.

    python scripts/run_human_survival.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from pdacfm.config import load_configurations, load_paths
from pdacfm.labels.human import attach_embeddings
from pdacfm.models import survival
from pdacfm.models.prefilter import prefilter
from pdacfm.stats.cox import grade_ordinal, multivariable_cox, stage_ordinal


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--search", action="store_true", help="rerun the Optuna search")
    ap.add_argument("--trials", type=int, default=survival.N_TRIALS)
    ap.add_argument("--paths")
    args = ap.parse_args()

    paths = load_paths(args.paths)
    cfg = load_configurations()["human"]["os"]
    embeddings = pd.read_csv(Path(paths["human"]["embeddings"]) / f"{cfg['encoder']}_{cfg['pooling']}.csv")
    cohort, features = attach_embeddings(pd.read_csv(paths["human"]["scans"]), embeddings)
    cohort = cohort[(cohort["os_days"] > 0) & cohort["os_event"].notna()].reset_index(drop=True)

    X = prefilter(cohort[features])
    params = None if args.search else cfg["params"]
    res = survival.run(X, cohort["os_days"], cohort["os_event"], cohort["patient_id"],
                       params=params, n_trials=args.trials)

    risk = cohort[["scan_id", "patient_id", "os_days", "os_event"]].assign(risk=res.risk)
    summary = {
        "endpoint": "os", "encoder": cfg["encoder"], "pooling": cfg["pooling"],
        "n": len(cohort), "events": int(cohort["os_event"].sum()), "params": res.params,
        "c_index": res.c_index, "c_index_ci": res.c_index_ci, "c_index_per_fold": res.c_index_per_fold,
        "c_uno": res.c_uno, "td_auc": res.td_auc,
    }

    if {"age", "stage", "grade"} <= set(cohort.columns):
        clin = pd.DataFrame({
            "time": cohort["os_days"], "event": cohort["os_event"], "risk": res.risk,
            "age": cohort["age"], "stage": cohort["stage"].map(stage_ordinal),
            "grade": cohort["grade"].map(grade_ordinal),
        })
        cox = multivariable_cox(clin)
        summary["cox"] = {k: v for k, v in cox.items() if k != "summary"}
        print(cox["summary"][["exp(coef)", "exp(coef) lower 95%", "exp(coef) upper 95%", "p"]].round(3))

    out = Path(paths["results"]) / "human" / "os"
    out.mkdir(parents=True, exist_ok=True)
    risk.to_csv(out / "risk.csv", index=False)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    lo, hi = res.c_index_ci
    print(f"os: C {res.c_index:.3f} [{lo:.3f}, {hi:.3f}], Uno C {res.c_uno:.3f}, "
          f"2-year AUC {res.td_auc.get(730, float('nan')):.3f}")


if __name__ == "__main__":
    main()
