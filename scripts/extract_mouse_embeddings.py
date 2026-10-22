"""Extract frozen-encoder embeddings for every mouse and post-treatment week.

Writes one long table per (encoder, streams, pooling):
<embeddings>/<encoder>_<streams>_<pooling>.csv with columns mouse_id, week,
s0_emb_* (T2w), s1_emb_* (T1w) and, for t1t2adc, adc_week_missing and adc_emb_*.

    python scripts/extract_mouse_embeddings.py --encoder dinov3_s --streams t1t2adc --pooling msc
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from pdacfm.config import load_paths
from pdacfm.features.encoders import load_timm_encoder
from pdacfm.features.mri import WEEKS, embed_session, prepare_session
from pdacfm.features.pooling import VARIANTS, WIDTH


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--encoder", required=True,
                    choices=["dinov3_s", "dinov3_b", "dinov3_l", "efficientnet_b0"])
    ap.add_argument("--streams", default="t1t2adc", choices=["t1t2", "t1t2adc"])
    ap.add_argument("--pooling", default="avg", choices=VARIANTS)
    ap.add_argument("--paths", help="paths YAML (default configs/paths.yaml)")
    args = ap.parse_args()

    if args.pooling != "avg" and not args.encoder.startswith("dinov3"):
        ap.error("patch-token pooling needs a DINOv3 encoder")
    paths = load_paths(args.paths)["mouse"]
    manifest = json.loads(Path(paths["manifest"]).read_text())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net, dim = load_timm_encoder(args.encoder)
    net = net.to(device)
    width = dim * WIDTH[args.pooling]
    use_adc = args.streams == "t1t2adc"

    rows = []
    for mouse_id, weeks in tqdm(manifest.items(), desc="mice"):
        for week in WEEKS:
            w = weeks.get(week)
            if not w or not all(w.get(k) and Path(w[k]).exists() for k in ("t2w", "t1w", "mask")):
                continue
            session = prepare_session(w["t2w"], w["t1w"], w["mask"],
                                      w.get("adc") if use_adc else None,
                                      w.get("adc_mask") if use_adc else None)
            emb = embed_session(net, session, args.pooling, device)
            row = {"mouse_id": mouse_id, "week": week}
            row.update({f"s0_emb_{i}": v for i, v in enumerate(emb["s0"])})
            row.update({f"s1_emb_{i}": v for i, v in enumerate(emb["s1"])})
            if use_adc:
                adc = emb["adc"] if emb["adc"] is not None else np.full(width, np.nan)
                row["adc_week_missing"] = int(emb["adc"] is None)
                row.update({f"adc_emb_{i}": v for i, v in enumerate(adc)})
            rows.append(row)

    out = Path(paths["embeddings"]) / f"{args.encoder}_{args.streams}_{args.pooling}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"wrote {out} ({len(rows)} sessions)")


if __name__ == "__main__":
    main()
