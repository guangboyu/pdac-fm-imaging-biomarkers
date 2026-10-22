"""Pooling of DINOv3 patch tokens into one vector per image stack.

A stack of S slices gives S x P patch tokens of width D after the final
layer norm (the class token and four register tokens are dropped). Pooling
runs over all S x P tokens:

* ``avg``  : timm average pooling per slice, then mean over slices (D)
* ``ms``   : patch-token mean and standard deviation (2D)
* ``msc``  : ``ms`` plus the class token averaged over slices (3D)
* ``mask`` : ``ms`` over patches that overlap the tumour mask only (2D)
* ``attn`` : mean and standard deviation weighted by last-block attention
             from the class token to each patch, averaged over heads (2D)
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

VARIANTS = ("avg", "ms", "msc", "mask", "attn")
WIDTH = {"avg": 1, "ms": 2, "msc": 3, "mask": 2, "attn": 2}


def forward_with_attention(net, x):
    """Run ``net.forward_features`` and capture the last block's attention.

    DINOv3 in timm uses rotary position embeddings with fused scaled
    dot-product attention, so the weights are recovered by wrapping
    ``F.scaled_dot_product_attention`` (which receives the rotated q and k).
    The last call, from the final block, is kept.
    """
    captured = {}
    original = F.scaled_dot_product_attention

    def capture(q, k, v, *args, **kwargs):
        scale = kwargs.get("scale") or q.shape[-1] ** -0.5
        captured["attn"] = ((q.float() @ k.float().transpose(-2, -1)) * scale).softmax(-1)
        return original(q, k, v, *args, **kwargs)

    F.scaled_dot_product_attention = capture
    try:
        tokens = net.forward_features(x)
    finally:
        F.scaled_dot_product_attention = original
    if "attn" not in captured:
        raise RuntimeError("attention weights were not captured")
    return tokens, captured["attn"]


def tumour_patch_mask(mask: torch.Tensor, n_patches: int) -> torch.Tensor:
    """Flag patches overlapping the tumour; ``mask`` is (S, H, W). Returns (S * P,)."""
    g = int(round(n_patches ** 0.5))
    pooled = F.adaptive_avg_pool2d(mask.float().unsqueeze(1), (g, g))
    return (pooled.reshape(mask.shape[0], -1) > 0).reshape(-1)


@torch.no_grad()
def pool_stack(net, x: torch.Tensor, variant: str, mask: torch.Tensor | None = None,
               attn_eps: float = 0.0) -> torch.Tensor:
    """Pool a stack ``x`` of shape (S, 3, H, W) into one feature vector.

    ``mask`` (S, H, W) is the tumour mask of each stack's centre slice and is
    needed only for ``mask``; without it ``mask`` falls back to ``ms``.
    ``attn_eps`` is added under the square root of the weighted variance
    (1e-8 in the mouse arm, 0 in the human arm).
    """
    if variant == "avg":
        return net(x).mean(dim=0)

    n_prefix = net.num_prefix_tokens
    if variant == "attn":
        tokens, attn = forward_with_attention(net, x)
        flat = tokens[:, n_prefix:, :].reshape(-1, tokens.shape[-1])
        w = attn[:, :, 0, n_prefix:].mean(dim=1).reshape(-1)
        w = w / w.sum().clamp(min=1e-8)
        mean = (w[:, None] * flat).sum(dim=0)
        var = (w[:, None] * (flat - mean) ** 2).sum(dim=0).clamp(min=0)
        return torch.cat([mean, torch.sqrt(var + attn_eps)])

    tokens = net.forward_features(x)
    patches = tokens[:, n_prefix:, :]
    flat = patches.reshape(-1, patches.shape[-1])
    if variant == "mask" and mask is not None:
        keep = tumour_patch_mask(mask.to(flat.device), patches.shape[1])
        if keep.any():
            flat = flat[keep]
    stats = [flat.mean(dim=0), flat.std(dim=0)]
    if variant == "msc":
        stats.append(tokens[:, 0, :].mean(dim=0))
    if variant not in ("ms", "msc", "mask"):
        raise ValueError(f"unknown pooling variant: {variant}")
    return torch.cat(stats)
