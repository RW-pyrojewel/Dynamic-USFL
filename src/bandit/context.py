# src/graph/context.py
from dataclasses import dataclass
from typing import Dict, List

import torch
import torch.nn as nn


@dataclass
class LayerProfile:
    """
    每一个 backbone block 的静态 profile 信息（单样本维度）。
    """
    conv_macs: float     # Conv 层的 MAC 数（单样本）
    linear_macs: float   # Linear 层的 MAC 数（单样本）
    conv_params: int     # Conv 层的参数量
    linear_params: int   # Linear 层的参数量
    act_numel: int       # block 输出 activation 的元素数（单样本）


def _conv_macs(module: nn.Conv2d, inp: torch.Tensor, out: torch.Tensor) -> int:
    """
    估算 Conv 的 MAC 数：
      MAC ≈ H_out * W_out * C_out * (C_in/groups * K_h * K_w)
    这里按“乘加算 1 次 MAC”来算（不乘以 2）。
    """
    # inp: [B, C_in, H_in, W_in]
    # out: [B, C_out, H_out, W_out]
    if inp.ndim != 4 or out.ndim != 4:
        return 0

    b, c_in, h_in, w_in = inp.shape
    b, c_out, h_out, w_out = out.shape
    k_h, k_w = module.kernel_size
    groups = module.groups

    macs_per_position = (c_in // groups) * k_h * k_w
    macs = h_out * w_out * c_out * macs_per_position
    return int(macs)


def _linear_macs(module: nn.Linear, inp: torch.Tensor, out: torch.Tensor) -> int:
    """
    估算 Linear 的 MAC 数：
      MAC ≈ in_features * out_features
    """
    if inp.ndim != 2 or out.ndim != 2:
        # 先简单拍平成 [B, -1] 处理
        b = inp.shape[0]
        inp = inp.view(b, -1)
        out = out.view(b, -1)

    b, in_f = inp.shape
    b, out_f = out.shape
    macs = in_f * out_f
    return int(macs)


def profile_backbone_layers(
    backbone: nn.Module,
    img_size: int,
    in_channels: int = 3,
    device: str = "cpu",
) -> List[LayerProfile]:
    """
    针对 USFLBackbone 的 .layers 做一次静态 profile，得到每个 block 的
    (macs, params, act_numel)。

    参数
    ----
    backbone : nn.Module
        需要具备 .layers: nn.ModuleList 属性。
    img_size : int
        输入图像的 H=W（假设方形图像）。
    in_channels : int
        输入图像通道数（通常为 3）。
    device : str
        profile 时使用的设备（"cpu" 足够）。

    返回
    ----
    profiles : List[LayerProfile]
        每个 block 一个条目，按 backbone.layers 的顺序排列。
    """
    backbone = backbone.to(device)
    backbone.eval()

    dummy = torch.zeros(1, in_channels, img_size, img_size, device=device)

    # 确保有 .layers
    if not hasattr(backbone, "layers"):
        raise ValueError("backbone must have attribute 'layers' (nn.ModuleList).")

    profiles: List[LayerProfile] = []

    x = dummy
    # 逐 block 运行，并用 forward hook 统计 Conv / Linear 的 MAC 与参数
    for block in backbone.layers:
        stats = {"conv_macs": 0.0, "linear_macs": 0.0, "conv_params": 0, "linear_params": 0}
        hooks = []

        def make_conv_hook():
            def hook(mod: nn.Conv2d, inp, out):
                macs = _conv_macs(mod, inp[0], out)
                stats["conv_macs"] += float(macs)

                params = mod.weight.numel()
                if mod.bias is not None:
                    params += mod.bias.numel()
                stats["conv_params"] += int(params)
            return hook

        def make_linear_hook():
            def hook(mod: nn.Linear, inp, out):
                macs = _linear_macs(mod, inp[0], out)
                stats["linear_macs"] += float(macs)

                params = mod.weight.numel()
                if mod.bias is not None:
                    params += mod.bias.numel()
                stats["linear_params"] += int(params)
            return hook

        # 注册 block 内 Conv/Linear 的 hook
        for m in block.modules():
            if isinstance(m, nn.Conv2d):
                hooks.append(m.register_forward_hook(make_conv_hook()))
            elif isinstance(m, nn.Linear):
                hooks.append(m.register_forward_hook(make_linear_hook()))

        with torch.no_grad():
            out = block(x)

        for h in hooks:
            h.remove()

        act_numel = out.numel()

        profiles.append(
            LayerProfile(
                conv_macs=stats["conv_macs"],
                linear_macs=stats["linear_macs"],
                conv_params=stats["conv_params"],
                linear_params=stats["linear_params"],
                act_numel=act_numel,
            )
        )

        x = out

    return profiles


def build_static_context_for_cut(
    profiles: List[LayerProfile],
    cut1: int,
    cut2: int,
    batch_size: int,
    bytes_per_elem: int,
) -> Dict[str, float]:
    """
    基于已 profile 的各 block 信息，构造一个 (cut1, cut2) 对应的静态上下文特征。

    假设：
      - 前端段: blocks [0 .. cut1]
      - 中间段: blocks [cut1+1 .. cut2]
      - 后端段: blocks [cut2+1 .. L-1]
      - 客户端 = 前端 + 后端
      - 服务器 = 中间
    """
    L = len(profiles)
    if not (0 <= cut1 <= cut2 < L - 1):
        raise ValueError(f"Invalid cuts: cut1={cut1}, cut2={cut2}, L={L}")

    server_idx = list(range(cut1 + 1, cut2 + 1))

    def sum_over(idxs, key: str) -> float:
        if key == "conv_macs":
            return float(sum(profiles[i].conv_macs for i in idxs))
        elif key == "linear_macs":
            return float(sum(profiles[i].linear_macs for i in idxs))
        elif key == "conv_params":
            return float(sum(profiles[i].conv_params for i in idxs))
        elif key == "linear_params":
            return float(sum(profiles[i].linear_params for i in idxs))
        elif key == "act_numel":
            return float(sum(profiles[i].act_numel for i in idxs))
        else:
            raise KeyError(key)

    # 按请求返回 server 端的细分统计（per batch for MACs）以及 activation bytes
    server_conv_macs = batch_size * sum_over(server_idx, "conv_macs")
    server_linear_macs = batch_size * sum_over(server_idx, "linear_macs")
    server_act_macs = batch_size * sum_over(server_idx, "act_numel")

    server_conv_params = int(sum_over(server_idx, "conv_params"))
    server_linear_params = int(sum_over(server_idx, "linear_params"))
    # 激活层通常无参数，保留字段但设为 0
    server_act_params = 0

    # IR 大小（per batch）
    act_front_numel = profiles[cut1].act_numel
    act_back_numel = profiles[cut2].act_numel

    act_front_bytes = batch_size * act_front_numel * bytes_per_elem
    act_back_bytes = batch_size * act_back_numel * bytes_per_elem

    context = {
        "server_conv_macs": float(server_conv_macs),
        "server_linear_macs": float(server_linear_macs),
        "server_act_macs": float(server_act_macs),
        "act_front_bytes": act_front_bytes,
        "act_back_bytes": act_back_bytes,
        "server_conv_params": server_conv_params,
        "server_linear_params": server_linear_params,
        "server_act_params": server_act_params,
    }
    return context
