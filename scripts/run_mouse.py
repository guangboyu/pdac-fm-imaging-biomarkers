"""Fit and evaluate one mouse endpoint with leave-one-mouse-out cross-validation.

Uses the configuration in configs/endpoints.yaml unless overridden. Writes
<results>/mouse/<endpoint>/summary.json and oof.csv.

    python scripts/run_mouse.py --endpoint cd8
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from lifelines.utils import concordance_index

from pdacfm.config import load_configurations, load_paths
from pdacfm.features import temporal
from pdacfm.labels.mouse import ENDPOINTS, continuous_endpoints, median_split
from pdacfm.models.classification import run_mouse
from pdacfm.models.prefilter import prefilter
from pdacfm.stats.metrics import spearman


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--endpoint", required=True, choices=ENDPOINTS)
    ap.add_argument("--encoder")
    ap.add_argument("--streams")
    ap.add_argument("--pooling")
    ap.add_argument("--aggregation", choices=temporal.REPRESENTATIONS)
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--paths")
    args = ap.parse_args()

    cfg = dict(load_configurations()["mouse"][args.endpoint])
    cfg.update({k: getattr(args, k) for k in ("encoder", "streams", "pooling", "aggregation")
                if getattr(args, k)})
    paths = load_paths(args.paths)

    long_df = pd.read_csv(Path(paths["mouse"]["embeddings"])
                          / f"{cfg['encoder']}_{cfg['streams']}_{cfg['pooling']}.csv")
    long_df["mouse_id"] = long_df["mouse_id"].astype(str)
    features = temporal.derive(long_df, cfg["aggregation"])

    pathology = pd.read_csv(paths["mouse"]["pathology"], dtype={"mouse_id": str})
    biology = continuous_endpoints(pathology)[args.endpoint]
    labels = median_split(biology)
    data = features[features["mouse_id"].isin(labels.index)].reset_index(drop=True)
    y = data["mouse_id"].map(labels).astype(int).to_numpy()

    X = prefilter(data[temporal.feature_columns(data)])
    result = run_mouse(X, y, data["mouse_id"].to_numpy(), n_trials=args.trials)

    oof = result.oof.rename(columns={"group": "mouse_id"})
    best = oof[oof["head"] == result.best_head]
    summary = {
        "endpoint": args.endpoint, **cfg, "n": len(data), "n_features": X.shape[1],
        "best_head": result.best_head, "auroc": result.auroc,
        "selector": result.selector, "head_params": result.head_params,
        "per_head": result.per_head.to_dict("records"),
    }
    value = best["mouse_id"].map(biology).to_numpy()
    if args.endpoint == "os":
        summary["c_index"] = concordance_index(value, best["y_prob"].to_numpy())
    else:
        summary["spearman"] = spearman(value, best["y_prob"].to_numpy())[0]

    out = Path(paths["results"]) / "mouse" / args.endpoint
    out.mkdir(parents=True, exist_ok=True)
    oof.to_csv(out / "oof.csv", index=False)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"{args.endpoint}: {result.best_head} AUROC {result.auroc:.3f} "
          f"[{result.per_head.iloc[0]['auroc_ci_lo']:.3f}, {result.per_head.iloc[0]['auroc_ci_hi']:.3f}]")


if __name__ == "__main__":
    main()
