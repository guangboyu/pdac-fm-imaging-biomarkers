"""Mouse MRI preprocessing and per-week embedding extraction.

For every animal and post-treatment week, the T2-weighted tumour mask
defines one in-plane crop (bounding box of the tumour over all tumour
slices plus a margin of max(20 voxels, 20% of the box side)) that is shared
by the T2-weighted and T1-weighted images and resized to 128 x 128. The ADC
map is cropped the same way from its own diffusion-space mask. Each volume
is clipped to the 1st and 99th percentile of the tumour voxels, scaled to
[0, 1], and background voxels are dimmed to 10%. Each stream is then
sampled as 32 evenly spaced three-slice stacks over the tumour slices,
embedded with a frozen encoder and pooled into one vector.

Intensities are passed to the encoder in [0, 1] without ImageNet
normalisation, as in the published runs.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
from scipy.ndimage import zoom

from pdacfm.features.pooling import pool_stack

WEEKS = ("week2", "week3", "week4", "week5")
CROP_SIZE = 128
N_STACKS = 32
BACKGROUND = 0.1


def read_volume(path) -> np.ndarray:
    """NIfTI to float32 array in (z, y, x) order."""
    return sitk.GetArrayFromImage(sitk.ReadImage(str(path))).astype(np.float32)


def normalise(vol: np.ndarray, mask: np.ndarray, background: float = BACKGROUND) -> np.ndarray:
    """Clip to the tumour 1st/99th percentiles, scale to [0, 1], dim the background."""
    fg = vol[mask > 0]
    if fg.size < 10:
        fg = vol.ravel()
    lo, hi = np.percentile(fg, 1), np.percentile(fg, 99)
    scaled = (np.clip(vol, lo, hi) - lo) / (hi - lo if hi - lo >= 1e-6 else 1.0)
    return (scaled * mask + scaled * (1.0 - mask) * background).astype(np.float32)


def crop_box(mask: np.ndarray, margin_min: int = 20, margin_frac: float = 0.20):
    """In-plane crop bounds and the tumour-bearing slice indices."""
    binary = mask > 0
    z_idx = list(np.where(binary.sum(axis=(1, 2)) > 0)[0]) or list(range(mask.shape[0]))
    proj = binary[z_idx].max(axis=0)
    rows, cols = np.where(proj.max(axis=1))[0], np.where(proj.max(axis=0))[0]
    if rows.size == 0 or cols.size == 0:
        rows, cols = np.arange(mask.shape[1]), np.arange(mask.shape[2])
    y_min, y_max, x_min, x_max = rows[0], rows[-1], cols[0], cols[-1]
    my = max(margin_min, int(margin_frac * max(y_max - y_min + 1, 1)))
    mx = max(margin_min, int(margin_frac * max(x_max - x_min + 1, 1)))
    H, W = mask.shape[1:]
    box = (max(0, y_min - my), min(H, y_max + my + 1), max(0, x_min - mx), min(W, x_max + mx + 1))
    return box, z_idx


def crop(vol: np.ndarray, box, z_idx, size: int = CROP_SIZE) -> np.ndarray:
    """Crop tumour slices and resize in plane to ``size`` x ``size`` (bilinear)."""
    y0, y1, x0, x1 = box
    z = [i for i in z_idx if i < vol.shape[0]] or list(range(min(5, vol.shape[0])))
    patch = vol[z, y0:y1, x0:x1]
    h, w = patch.shape[1:]
    if (h, w) != (size, size):
        patch = zoom(patch, (1.0, size / max(h, 1), size / max(w, 1)), order=1)
    return patch.astype(np.float32)


def stacks(vol: np.ndarray, n: int = N_STACKS) -> np.ndarray:
    """``n`` evenly spaced (z-1, z, z+1) stacks; returns (n, 3, H, W)."""
    nz = vol.shape[0]
    centres = np.arange(1, nz - 1) if nz >= 3 else np.arange(nz)
    centres = centres[np.linspace(0, len(centres) - 1, n, dtype=int)]
    return np.stack([vol[[max(0, c - 1), c, min(nz - 1, c + 1)]] for c in centres])


def prepare_session(t2w, t1w, mask, adc=None, dwi_mask=None) -> dict:
    """Load one weekly session and return cropped, normalised stacks."""
    m = (read_volume(mask) > 0).astype(np.float32)
    box, z_idx = crop_box(m)
    out = {
        "s0": stacks(crop(normalise(read_volume(t2w), m), box, z_idx)),
        "s1": stacks(crop(normalise(read_volume(t1w), m), box, z_idx)),
        "mask": stacks(crop(m, box, z_idx))[:, 1],
        "adc": None,
    }
    if adc and dwi_mask and Path(adc).exists() and Path(dwi_mask).exists():
        dm = (read_volume(dwi_mask) > 0).astype(np.float32)
        dbox, dz = crop_box(dm)
        out["adc"] = stacks(crop(normalise(read_volume(adc), dm), dbox, dz))
    return out


def embed_session(net, session: dict, variant: str, device) -> dict:
    """Pool each stream of one session; T2w and T1w share the T2w tumour mask."""
    t = lambda a: torch.from_numpy(a).to(device)  # noqa: E731
    mask = t(session["mask"])
    out = {
        "s0": pool_stack(net, t(session["s0"]), variant, mask, attn_eps=1e-8),
        "s1": pool_stack(net, t(session["s1"]), variant, mask, attn_eps=1e-8),
        "adc": None,
    }
    if session["adc"] is not None:
        out["adc"] = pool_stack(net, t(session["adc"]), variant, None, attn_eps=1e-8)
    return {k: (v.cpu().numpy() if v is not None else None) for k, v in out.items()}
