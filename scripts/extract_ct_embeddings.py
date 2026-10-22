"""Extract one frozen-encoder embedding per CT scan.

Writes <embeddings>/<encoder>_<pooling>.csv with columns scan_id, patient_id,
emb_*. DINOv3 inputs are ImageNet-normalised; MedSAM applies its own
normalisation; EfficientNet-B0 receives [0, 1] intensities.

    python scripts/extract_ct_embeddings.py --encoder dinov3_l --pooling attn
    python scripts/extract_ct_embeddings.py --encoder medsam
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

from pdacfm.config import load_paths
from pdacfm.features.ct import load_roi, stacks_full, stacks_tumour
from pdacfm.features.encoders import ImageNetNormalize, MedSAMEncoder, load_timm_encoder
from pdacfm.features.pooling import VARIANTS, pool_stack


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--encoder", required=True,
                    choices=["dinov3_s", "dinov3_b", "dinov3_l", "medsam", "efficientnet_b0"])
    ap.add_argument("--pooling", default="avg", choices=VARIANTS)
    ap.add_argument("--paths")
    args = ap.parse_args()

    dinov3 = args.encoder.startswith("dinov3")
    if args.pooling != "avg" and not dinov3:
        ap.error("patch-token pooling needs a DINOv3 encoder")
    paths = load_paths(args.paths)["human"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.encoder == "medsam":
        net = MedSAMEncoder(paths["medsam_checkpoint"])
    else:
        net, _ = load_timm_encoder(args.encoder)
    net = net.to(device)
    norm = ImageNetNormalize().to(device)

    scans = pd.read_csv(paths["scans"])
    rows = []
    for _, row in tqdm(scans.iterrows(), total=len(scans), desc="scans"):
        roi = Path(paths["roi"]) / str(row["scan_id"])
        if not (roi / "image.nii.gz").exists():
            continue
        img, mask = load_roi(roi / "image.nii.gz", roi / "label.nii.gz")
        if args.pooling == "avg":
            x, m = stacks_full(img), None
        else:
            x, m = stacks_tumour(img, mask)
        x = torch.from_numpy(x).float().to(device)
        if dinov3:
            x = norm(x)
        with torch.no_grad():
            if args.pooling == "avg":
                emb = net(x).mean(dim=0)
            else:
                emb = pool_stack(net, x, args.pooling,
                                 torch.from_numpy(m).to(device) if m is not None else None)
        rows.append({"scan_id": row["scan_id"], "patient_id": row["patient_id"],
                     **{f"emb_{i}": float(v) for i, v in enumerate(emb.cpu().numpy())}})

    out = Path(paths["embeddings"]) / f"{args.encoder}_{args.pooling}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"wrote {out} ({len(rows)} scans)")


if __name__ == "__main__":
    main()
