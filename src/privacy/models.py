# src/privacy/models.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, List, Dict, Any

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

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
            nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
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
            _reinit_weights(self.front)

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
        h = F.relu(self.fc1(h), inplace=False)
        h = self.drop(h)
        return self.fc2(h)


class _ResizeTo(nn.Module):
    """
    Lightweight inverse of pooling / size mismatch: force feature map to a target spatial size.
    """
    def __init__(self, target_hw: Tuple[int, int], mode: str = "bilinear"):
        super().__init__()
        self.target_hw = (int(target_hw[0]), int(target_hw[1]))
        self.mode = mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(-2) == self.target_hw[0] and x.size(-1) == self.target_hw[1]:
            return x
        return F.interpolate(x, size=self.target_hw, mode=self.mode, align_corners=False)


class MirrorDecoder(nn.Module):
    """
    STRICT mirror decoder for AutoEncoderNN-style MIA.

    Design goal:
      - "Mirror" victim client-front f1 (up to cut1) by:
        (i) tracing the *leaf* ops (Conv2d / Pool / BN / ReLU) with a dummy forward,
        (ii) building a reverse stack that mirrors the same op sequence:
             Conv2d -> ConvTranspose2d (with output_padding solved from traced shapes),
             MaxPool/AvgPool -> ResizeTo (restoring the pre-pool spatial size),
             BN/ReLU -> BN/ReLU (same order reversed).

    Notes:
      - For composite blocks (e.g., ResNet BasicBlock), we mirror their internal leaf ops,
        i.e., this decoder is a strict mirror of the *leaf-op sequence*, not a symbolic inverse.
      - This keeps changes minimal while matching the paper's "mirror the client architecture"
        requirement at the granularity that matters for reconstruction capacity.
    """
    def __init__(self, net: nn.Sequential, img_shape: Tuple[int, int, int], out_act: str = "sigmoid"):
        super().__init__()
        self.img_shape = (int(img_shape[0]), int(img_shape[1]), int(img_shape[2]))
        self.net = net

        # final activation is appended by factory; keep here for safety
        if out_act.lower() == "sigmoid":
            self.out_act = nn.Sigmoid()
        else:
            self.out_act = nn.Identity()

    @staticmethod
    def _convtranspose_output_padding(
        in_hw: Tuple[int, int],
        out_hw: Tuple[int, int],
        conv: nn.Conv2d,
    ) -> Tuple[int, int]:
        """
        Compute output_padding (h, w) so that ConvTranspose2d mirrors Conv2d size transform.
        If unsatisfied (due to constraints), return (0,0) and rely on ResizeTo as a last resort.
        """
        k_h, k_w = conv.kernel_size if isinstance(conv.kernel_size, tuple) else (conv.kernel_size, conv.kernel_size)
        s_h, s_w = conv.stride if isinstance(conv.stride, tuple) else (conv.stride, conv.stride)
        p_h, p_w = conv.padding if isinstance(conv.padding, tuple) else (conv.padding, conv.padding)
        d_h, d_w = conv.dilation if isinstance(conv.dilation, tuple) else (conv.dilation, conv.dilation)

        # convtranspose formula:
        # out = (in-1)*stride - 2*pad + dilation*(k-1) + output_padding + 1
        def solve(in_len: int, target: int, k: int, s: int, p: int, d: int) -> int:
            base = (in_len - 1) * s - 2 * p + d * (k - 1) + 1
            op = target - base
            if op < 0 or op >= s:
                return 0
            return int(op)

        op_h = solve(in_hw[0], out_hw[0], k_h, s_h, p_h, d_h)
        op_w = solve(in_hw[1], out_hw[1], k_w, s_w, p_w, d_w)
        return op_h, op_w

    @classmethod
    def from_encoder(
        cls,
        encoder_front: nn.Module,
        z_spec: LatentSpec,
        img_shape: Tuple[int, int, int],
        out_act: str = "sigmoid",
    ) -> "MirrorDecoder":
        """
        Build a strict mirror decoder by tracing encoder_front leaf ops on a dummy image.
        """
        device = next(encoder_front.parameters(), torch.empty(0)).device
        x_dummy = torch.zeros(1, *img_shape, device=device)

        # --- trace leaf ops ---
        records: List[Dict[str, Any]] = []
        hooks = []

        def _register(m: nn.Module):
            # record only leaf ops; containers will be expanded by .modules()
            if isinstance(m, nn.Conv2d):
                def hook(mod, inp, out):
                    x_in = inp[0]
                    records.append({
                        "type": "conv2d",
                        "module": mod,
                        "in_shape": tuple(x_in.shape),
                        "out_shape": tuple(out.shape),
                    })
                hooks.append(m.register_forward_hook(hook))
            elif isinstance(m, (nn.MaxPool2d, nn.AvgPool2d)):
                def hook(mod, inp, out):
                    x_in = inp[0]
                    records.append({
                        "type": "pool",
                        "module": mod,
                        "in_shape": tuple(x_in.shape),
                        "out_shape": tuple(out.shape),
                    })
                hooks.append(m.register_forward_hook(hook))
            elif isinstance(m, nn.BatchNorm2d):
                def hook(mod, inp, out):
                    x_in = inp[0]
                    records.append({
                        "type": "bn2d",
                        "module": mod,
                        "in_shape": tuple(x_in.shape),
                        "out_shape": tuple(out.shape),
                    })
                hooks.append(m.register_forward_hook(hook))
            elif isinstance(m, nn.ReLU):
                def hook(mod, inp, out):
                    x_in = inp[0]
                    records.append({
                        "type": "relu",
                        "module": mod,
                        "in_shape": tuple(x_in.shape),
                        "out_shape": tuple(out.shape),
                    })
                hooks.append(m.register_forward_hook(hook))
            elif isinstance(m, nn.Identity):
                def hook(mod, inp, out):
                    x_in = inp[0]
                    records.append({
                        "type": "identity",
                        "module": mod,
                        "in_shape": tuple(x_in.shape),
                        "out_shape": tuple(out.shape),
                    })
                hooks.append(m.register_forward_hook(hook))

        for m in encoder_front.modules():
            # skip the root module itself if it's a container; leaf ops are what we want
            if m is encoder_front:
                continue
            _register(m)

        was_training = encoder_front.training
        encoder_front.eval()
        with torch.no_grad():
            z_dummy = encoder_front(x_dummy)

        # cleanup hooks
        for h in hooks:
            h.remove()
        if was_training:
            encoder_front.train()

        # sanity check against victim latent spec
        if z_dummy.dim() != 4:
            raise ValueError(f"encoder_front must output 4D features, got {tuple(z_dummy.shape)}")
        z_chk = LatentSpec.from_tensor(z_dummy)
        if (z_chk.c, z_chk.h, z_chk.w) != (z_spec.c, z_spec.h, z_spec.w):
            raise ValueError(
                "LatentSpec mismatch between encoder_front(dummy) and victim A_front.\n"
                f"  encoder_front: (c,h,w)=({z_chk.c},{z_chk.h},{z_chk.w})\n"
                f"  victim z_spec : (c,h,w)=({z_spec.c},{z_spec.h},{z_spec.w})\n"
                "Check that cut1 and img_shape match the victim-side privacy_samples."
            )

        # --- build inverse stack (reverse order) ---
        inv_layers: List[nn.Module] = []
        cur_c = z_spec.c
        cur_hw = (z_spec.h, z_spec.w)

        for rec in reversed(records):
            rtype = rec["type"]
            if rtype == "conv2d":
                conv: nn.Conv2d = rec["module"]
                in_shape = rec["in_shape"]   # before conv
                out_shape = rec["out_shape"] # after conv
                # We invert: out -> in
                target_hw = (int(in_shape[2]), int(in_shape[3]))
                in_hw = (int(out_shape[2]), int(out_shape[3]))
                op_h, op_w = cls._convtranspose_output_padding(in_hw=in_hw, out_hw=target_hw, conv=conv)

                # mirror channels: out_channels -> in_channels
                deconv = nn.ConvTranspose2d(
                    in_channels=int(out_shape[1]),
                    out_channels=int(in_shape[1]),
                    kernel_size=conv.kernel_size,
                    stride=conv.stride,
                    padding=conv.padding,
                    output_padding=(op_h, op_w),
                    dilation=conv.dilation,
                    groups=conv.groups,
                    bias=(conv.bias is not None),
                )
                inv_layers.append(deconv)

                # if output_padding cannot perfectly restore size, enforce exact size
                inv_layers.append(_ResizeTo(target_hw))
                cur_c = int(in_shape[1])
                cur_hw = target_hw

            elif rtype == "pool":
                in_shape = rec["in_shape"]
                target_hw = (int(in_shape[2]), int(in_shape[3]))
                inv_layers.append(_ResizeTo(target_hw))
                cur_hw = target_hw

            elif rtype == "bn2d":
                # Use current decoder channel count to build BN to avoid
                # running_mean/var size mismatch when channel ordering differs.
                # Mirror behavior approximately by creating a BN with cur_c features.
                inv_layers.append(nn.BatchNorm2d(cur_c))

            elif rtype == "relu":
                inv_layers.append(nn.ReLU(inplace=False))

            elif rtype == "identity":
                inv_layers.append(nn.Identity())

        # If channels mismatch image channels, project at the end.
        c_img, h_img, w_img = (int(img_shape[0]), int(img_shape[1]), int(img_shape[2]))
        if cur_c != c_img:
            inv_layers.append(nn.Conv2d(cur_c, c_img, kernel_size=1, stride=1, padding=0))
            cur_c = c_img

        # enforce final spatial shape exactly
        inv_layers.append(_ResizeTo((h_img, w_img)))

        # final activation
        if out_act.lower() == "sigmoid":
            inv_layers.append(nn.Sigmoid())
        else:
            inv_layers.append(nn.Identity())

        net = nn.Sequential(*inv_layers)
        _reinit_weights(net)

        return cls(net=net, img_shape=img_shape, out_act=out_act)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.net(z)
        c, h, w = self.img_shape
        # safety: final enforce (should already match via ResizeTo)
        if x.size(1) != c:
            x = x[:, :c, ...] if x.size(1) > c else F.pad(x, (0, 0, 0, 0, 0, c - x.size(1)))
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

    def forward_latent(self, z: torch.Tensor):
        """Train-time forward when smashed activations are already available (no encoder).

        This is used for victim unlabeled batches where only A_front (latent Z) is observed.
        """
        logits = self.label_head(z)
        x_hat = self.decoder(z)
        return logits, x_hat

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
    front_modules = []
    for layer in layers[: cut1 + 1]:
        _reinit_weights(layer)
        front_modules.append(layer)

    return nn.Sequential(*front_modules)
