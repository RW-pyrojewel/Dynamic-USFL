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
    macs: float          # Conv + Linear 的 MAC 数（不区分前向/反向）
    params: int          # Conv + Linear 的参数量
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
        stats = {"macs": 0.0, "params": 0}
        hooks = []

        def make_conv_hook():
            def hook(mod: nn.Conv2d, inp, out):
                macs = _conv_macs(mod, inp[0], out)
                stats["macs"] += float(macs)

                params = mod.weight.numel()
                if mod.bias is not None:
                    params += mod.bias.numel()
                stats["params"] += int(params)
            return hook

        def make_linear_hook():
            def hook(mod: nn.Linear, inp, out):
                macs = _linear_macs(mod, inp[0], out)
                stats["macs"] += float(macs)

                params = mod.weight.numel()
                if mod.bias is not None:
                    params += mod.bias.numel()
                stats["params"] += int(params)
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
                macs=stats["macs"],
                params=stats["params"],
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
    if not (0 <= cut1 < cut2 < L):
        raise ValueError(f"Invalid cuts: cut1={cut1}, cut2={cut2}, L={L}")

    front_idx = list(range(0, cut1 + 1))
    mid_idx = list(range(cut1 + 1, cut2 + 1))
    back_idx = list(range(cut2 + 1, L))

    edge_idx = front_idx + back_idx
    server_idx = mid_idx

    def sum_over(idxs, key: str) -> float:
        if key == "macs":
            return float(sum(profiles[i].macs for i in idxs))
        elif key == "params":
            return float(sum(profiles[i].params for i in idxs))
        elif key == "act_numel":
            return float(sum(profiles[i].act_numel for i in idxs))
        else:
            raise KeyError(key)

    # MACs 总量（per batch）
    edge_macs = batch_size * sum_over(edge_idx, "macs")
    server_macs = batch_size * sum_over(server_idx, "macs")

    # 参数量
    edge_params = sum_over(edge_idx, "params")
    server_params = sum_over(server_idx, "params")

    # IR 大小（per batch）
    act_front_numel = profiles[cut1].act_numel
    act_back_numel = profiles[cut2].act_numel

    act_front_bytes = batch_size * act_front_numel * bytes_per_elem
    act_back_bytes = batch_size * act_back_numel * bytes_per_elem

    context = {
        "edge_macs": edge_macs,
        "server_macs": server_macs,
        "act_front_bytes": act_front_bytes,
        "act_back_bytes": act_back_bytes,
        "edge_params": edge_params,
        "server_params": server_params,
    }
    return context
