"""Human CT preprocessing and slice-stack sampling.

``build_roi`` turns a CT volume and its tumour mask into a fixed region of
interest: the CT is clipped to a soft-tissue window (level 50, width 400 HU,
so [-150, 250] HU), min-max scaled to [0, 1] over the volume, resampled to
1 x 1 x 3 mm (x, y, z) and cropped to 96 x 160 x 160 voxels (z, y, x)
centred on the tumour bounding box.

Embeddings are computed from 48 three-slice stacks per scan with the
background dimmed to 10%. Two samplings were used:

* ``full``: stacks evenly spaced over all 96 slices of the region
  (used for MedSAM and EfficientNet-B0)
* ``tumour``: stacks centred on evenly spaced tumour-bearing slices
  (used for the DINOv3 patch-token pooling variants)
"""
from __future__ import annotations

import nibabel as nib
import numpy as np
import SimpleITK as sitk

SPACING_XYZ = (1.0, 1.0, 3.0)
CROP_ZYX = (96, 160, 160)
WINDOW = (50.0, 400.0)
N_STACKS = 48
BACKGROUND = 0.1


def _resample(img: sitk.Image, spacing, is_label: bool, reference: sitk.Image) -> sitk.Image:
    size = [max(1, int(round(n * s / t))) for n, s, t in zip(img.GetSize(), img.GetSpacing(), spacing)]
    f = sitk.ResampleImageFilter()
    f.SetReferenceImage(reference)
    f.SetOutputSpacing(tuple(float(s) for s in spacing))
    f.SetSize(size)
    f.SetOutputDirection(reference.GetDirection())
    f.SetOutputOrigin(reference.GetOrigin())
    if is_label:
        f.SetInterpolator(sitk.sitkNearestNeighbor)
        f.SetDefaultPixelValue(0)
    else:
        f.SetInterpolator(sitk.sitkLinear)
        f.SetDefaultPixelValue(float(sitk.GetArrayViewFromImage(img).min()))
    return f.Execute(img)


def _crop_pad(arr: np.ndarray, centre, size, pad_value) -> np.ndarray:
    size = np.array(size)
    start = np.asarray(centre) - size // 2
    src0, src1 = np.maximum(start, 0), np.minimum(start + size, arr.shape)
    dst0 = src0 - start
    dst1 = dst0 + (src1 - src0)
    out = np.full(tuple(size), pad_value, dtype=arr.dtype)
    out[dst0[0]:dst1[0], dst0[1]:dst1[1], dst0[2]:dst1[2]] = \
        arr[src0[0]:src1[0], src0[1]:src1[1], src0[2]:src1[2]]
    return out


def build_roi(ct_path, mask_path):
    """Return (image, mask) SimpleITK images of the cropped region, or None if the mask is empty."""
    ct = sitk.ReadImage(str(ct_path))
    mask = sitk.Cast(sitk.ReadImage(str(mask_path)), sitk.sitkUInt8)
    if sitk.GetArrayViewFromImage(mask).max() == 0:
        return None

    level, width = WINDOW
    ct = sitk.Cast(sitk.Clamp(ct, lowerBound=level - width / 2, upperBound=level + width / 2),
                   sitk.sitkFloat32)
    arr = sitk.GetArrayFromImage(ct)
    scaled = sitk.GetImageFromArray((arr - arr.min()) / max(float(arr.max() - arr.min()), 1e-8))
    scaled.CopyInformation(ct)

    ct_rs = _resample(scaled, SPACING_XYZ, False, scaled)
    mask_rs = _resample(mask, SPACING_XYZ, True, scaled)
    mask_rs = sitk.Resample(mask_rs, ct_rs, sitk.Transform(), sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)

    ct_arr = sitk.GetArrayFromImage(ct_rs).astype(np.float32)
    m_arr = sitk.GetArrayFromImage(mask_rs).astype(np.uint8)
    coords = np.argwhere(m_arr > 0)
    if coords.size == 0:
        return None
    centre = np.round((coords.min(axis=0) + coords.max(axis=0)) / 2.0).astype(int)

    out_img = sitk.GetImageFromArray(_crop_pad(ct_arr, centre, CROP_ZYX, float(ct_arr.min())))
    out_msk = sitk.GetImageFromArray((_crop_pad(m_arr, centre, CROP_ZYX, 0) > 0).astype(np.uint8))
    for im in (out_img, out_msk):
        im.SetSpacing(ct_rs.GetSpacing())
        im.SetDirection(ct_rs.GetDirection())
    return out_img, out_msk


def load_roi(image_path, mask_path):
    """Read a cached region as (z, y, x) arrays with the background dimmed."""
    img = nib.load(str(image_path)).get_fdata().astype(np.float32).transpose(2, 1, 0)
    mask = nib.load(str(mask_path)).get_fdata().astype(np.float32).transpose(2, 1, 0)
    return img * mask + img * (1 - mask) * BACKGROUND, mask


def stacks_full(img: np.ndarray, n: int = N_STACKS) -> np.ndarray:
    """``n`` stacks (z, z+1, z+2) evenly spaced over the whole region."""
    nz = img.shape[0]
    starts = np.linspace(0, max(0, nz - 3), n, dtype=int)
    return np.stack([img[min(z, nz - 3):min(z, nz - 3) + 3] for z in starts])


def stacks_tumour(img: np.ndarray, mask: np.ndarray, n: int = N_STACKS):
    """``n`` stacks (z-1, z, z+1) centred on evenly spaced tumour slices.

    Returns the stacks (n, 3, H, W) and the centre-slice masks (n, H, W).
    """
    nz = img.shape[0]
    z_tumour = np.where((mask > 0).sum(axis=(1, 2)) > 0)[0]
    if z_tumour.size == 0:
        z_tumour = np.arange(nz)
    centres = z_tumour[np.linspace(0, z_tumour.size - 1, n).astype(int)]
    x = np.stack([img[[max(0, c - 1), c, min(nz - 1, c + 1)]] for c in centres])
    return x, np.stack([mask[c] for c in centres])
