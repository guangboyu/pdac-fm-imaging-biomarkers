"""Crop every CT scan to the tumour region used for embedding extraction.

Reads ct_path and mask_path from the scan table and writes
<roi>/<scan_id>/image.nii.gz and label.nii.gz.

    python scripts/build_ct_roi.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import SimpleITK as sitk
from tqdm import tqdm

from pdacfm.config import load_paths
from pdacfm.features.ct import build_roi


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--paths")
    args = ap.parse_args()
    paths = load_paths(args.paths)["human"]
    scans = pd.read_csv(paths["scans"])
    root = Path(paths["roi"])
    for _, row in tqdm(scans.iterrows(), total=len(scans), desc="scans"):
        roi = build_roi(row["ct_path"], row["mask_path"])
        if roi is None:
            print(f"skipped {row['scan_id']}: empty mask")
            continue
        out = root / str(row["scan_id"])
        out.mkdir(parents=True, exist_ok=True)
        sitk.WriteImage(roi[0], str(out / "image.nii.gz"))
        sitk.WriteImage(roi[1], str(out / "label.nii.gz"))


if __name__ == "__main__":
    main()
