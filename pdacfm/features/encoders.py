"""Frozen pretrained image encoders.

All encoders are used as fixed feature extractors: weights are never
updated and no labels are seen. Inputs are 2.5D stacks of three adjacent
axial slices passed as the three colour channels, with intensities in [0, 1].
"""
from __future__ import annotations

import types

import timm
import torch
import torch.nn.functional as F
from torch import nn

TIMM_ENCODERS = {
    "dinov3_s": ("vit_small_patch16_dinov3", 384),
    "dinov3_b": ("vit_base_patch16_dinov3", 768),
    "dinov3_l": ("vit_large_patch16_dinov3", 1024),
    "efficientnet_b0": ("efficientnet_b0", 1280),
}

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _freeze(model: nn.Module) -> nn.Module:
    for p in model.parameters():
        p.requires_grad_(False)
    return model.eval()


def load_timm_encoder(name: str) -> tuple[nn.Module, int]:
    """DINOv3 or EfficientNet-B0 from timm, with average pooling of the last layer."""
    timm_name, dim = TIMM_ENCODERS[name]
    model = timm.create_model(timm_name, pretrained=True, num_classes=0, global_pool="avg")
    return _freeze(model), dim


class ImageNetNormalize(nn.Module):
    """Apply ImageNet mean/std normalisation to [0, 1] inputs."""

    def __init__(self):
        super().__init__()
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, x):
        return (x - self.mean) / self.std


class MedSAMEncoder(nn.Module):
    """MedSAM ViT-B image encoder (``segment_anything`` checkpoint).

    Slices are resized to 128 x 128, the positional embedding is
    bicubically interpolated to the smaller patch grid, and the 256-channel
    neck output is averaged over space.
    """

    image_size = 128
    dim = 256

    def __init__(self, checkpoint: str):
        super().__init__()
        from segment_anything import sam_model_registry

        self.encoder = sam_model_registry["vit_b"](checkpoint=checkpoint).image_encoder
        self.encoder.forward = types.MethodType(self._forward_interp, self.encoder)
        self.register_buffer("mean", torch.tensor([123.675, 116.28, 103.53]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([58.395, 57.12, 57.375]).view(1, 3, 1, 1))
        _freeze(self)

    @staticmethod
    def _forward_interp(enc, x):
        x = enc.patch_embed(x)
        ph, pw = x.shape[1], x.shape[2]
        pos = enc.pos_embed
        if (ph, pw) != (pos.shape[1], pos.shape[2]):
            pos = F.interpolate(pos.permute(0, 3, 1, 2).float(), size=(ph, pw),
                                mode="bicubic", align_corners=False).permute(0, 2, 3, 1).to(x.dtype)
        x = x + pos
        for blk in enc.blocks:
            x = blk(x)
        return enc.neck(x.permute(0, 3, 1, 2))

    def forward(self, x):
        if x.shape[-2:] != (self.image_size, self.image_size):
            x = F.interpolate(x, size=(self.image_size, self.image_size),
                              mode="bilinear", align_corners=False)
        x = (x * 255.0 - self.mean) / self.std
        return self.encoder(x).flatten(2).mean(-1)
