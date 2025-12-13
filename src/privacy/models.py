# src/privacy/models.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from src.models.model_usfl import USFLBackbone


def _reinit_weights(module: nn.Module) -> None:
    """
    Re-initialize a module tree to avoid "leaking" victim weights.
    """
    for m in module.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))  # type: ignore[name-defined]
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
            if hasattr(m, "weight") and m.weight is not None:
                nn.init.ones_(m.weight)
            if hasattr(m, "bias") and m.bias is not None:
                nn.init.zeros_(m.bias)


@dataclass
class LatentSpec:
    c: int
    h: int
    w: int

    @staticmethod
    def from_tensor(z: torch.Tensor) -> "LatentSpec":
        if z.dim() != 4:
            raise ValueError(f"Expected 4D smashed activations, got shape={tuple(z.shape)}")
        return LatentSpec(c=int(z.size(1)), h=int(z.size(2)), w=int(z.size(3)))


class ShadowEncoder(nn.Module):
    """
    Shadow encoder E_theta with the same *architecture* as victim client-front f1 (up to cut1),
    but with independently initialized weights.

    NOTE:
      We accept a 'front_module' (nn.Module) as a template, then re-initialize it.
    """
    def __init__(self, front_module: nn.Module, reinit: bool = True):
        super().__init__()
        self.front = front_module
        if reinit:
            # re-init to avoid using victim weights
            for m in self.front.modules():
                if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.Linear):
                    nn.init.kaiming_uniform_(m.weight, a=5 ** 0.5)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
                    if getattr(m, "weight", None) is not None:
                        nn.init.ones_(m.weight)
                    if getattr(m, "bias", None) is not None:
                        nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.front(x)


class LabelHead(nn.Module):
    """
    Inference head H_phi that maps smashed activations Z (4D feature maps) to logits.
    """
    def __init__(self, z_spec: LatentSpec, num_classes: int, hidden_dim: int = 256, dropout: float = 0.0):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(z_spec.c, hidden_dim)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.dim() != 4:
            raise ValueError(f"LabelHead expects 4D z, got {tuple(z.shape)}")
        h = self.pool(z).flatten(1)
        h = F.relu(self.fc1(h), inplace=True)
        h = self.drop(h)
        return self.fc2(h)


class MirrorDecoder(nn.Module):
    """
    Decoder D_psi: maps smashed activations Z (Cz,Hz,Wz) back to image (C,H,W).

    This is a generic ConvTranspose-based upsampler. For strict "mirror of ResNet blocks",
    implement a specialized decoder per backbone; interface remains the same.
    """
    def __init__(
        self,
        z_spec: LatentSpec,
        img_shape: Tuple[int, int, int],
        channel_min: int = 32,
        out_act: str = "sigmoid",
    ):
        super().__init__()
        c_img, h_img, w_img = img_shape
        self.img_shape = img_shape

        # how many x2 upsample steps are needed
        steps_h = int(round(math.log2(h_img / z_spec.h))) if z_spec.h > 0 else 0  # type: ignore[name-defined]
        steps_w = int(round(math.log2(w_img / z_spec.w))) if z_spec.w > 0 else 0  # type: ignore[name-defined]
        steps = max(0, min(6, max(steps_h, steps_w)))  # safety cap

        layers = []
        in_ch = z_spec.c
        out_ch = max(channel_min, in_ch // 2)

        for _ in range(steps):
            layers += [
                nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ]
            in_ch = out_ch
            out_ch = max(channel_min, out_ch // 2)

        # final projection to RGB (or grayscale)
        layers += [
            nn.Conv2d(in_ch, c_img, kernel_size=3, stride=1, padding=1),
        ]
        if out_act.lower() == "sigmoid":
            layers += [nn.Sigmoid()]
        else:
            layers += [nn.Identity()]

        self.net = nn.Sequential(*layers)

        # If the computed steps do not perfectly match target size, we will interpolate at runtime.

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.net(z)
        c, h, w = self.img_shape
        if x.size(2) != h or x.size(3) != w:
            x = F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False)
        return x


class SAESLAttacker(nn.Module):
    """
    SAE-SL attacker with:
      - ShadowEncoder E_theta (train-time only)
      - LabelHead H_phi
      - MirrorDecoder D_psi

    Train-time forward:
      logits, x_hat, z = attacker.forward_aux(x_aux)

    Victim-time forward (bypass encoder):
      logits, x_hat = attacker.forward_victim(A_front)
    """
    def __init__(
        self,
        encoder: ShadowEncoder,
        label_head: LabelHead,
        decoder: MirrorDecoder,
    ):
        super().__init__()
        self.encoder = encoder
        self.label_head = label_head
        self.decoder = decoder

    def forward_aux(self, x_aux: torch.Tensor):
        z = self.encoder(x_aux)
        logits = self.label_head(z)
        x_hat = self.decoder(z)
        return logits, x_hat, z

    @torch.no_grad()
    def forward_victim(self, A_front: torch.Tensor):
        logits = self.label_head(A_front)
        x_hat = self.decoder(A_front)
        return logits, x_hat


def build_front_template_from_backbone(backbone: USFLBackbone, cut1: int) -> nn.Module:
    """
    Build a *module template* for victim f1, assuming backbone has .layers: nn.ModuleList.

    For your ResNet18USFLBackbone and MobileNetV2USFLBackbone this holds (per your code).
    """
    layers = backbone.layers
    if cut1 < 0 or cut1 >= len(layers):
        raise ValueError(f"cut1={cut1} out of range for backbone.layers (len={len(layers)})")

    # f1 includes layers[0..cut1] inclusive
    front_modules = [layers[i] for i in range(cut1 + 1)]
    return nn.Sequential(*front_modules)
