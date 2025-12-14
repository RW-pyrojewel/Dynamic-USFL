# src/privacy/data.py

from dataclasses import dataclass
from typing import Optional

import os
import torch
from torch.utils.data import DataLoader

from .config import PrivacyConfig


@dataclass
class VictimBatch:
    A_front: torch.Tensor           # [N, ...]
    y: torch.Tensor                 # [N]
    x: Optional[torch.Tensor] = None  # [N, C, H, W] or None

    @property
    def num_samples(self) -> int:
        return int(self.A_front.shape[0])


def build_aux_loader(cfg, priv_cfg: PrivacyConfig) -> DataLoader:
    """
    构建辅助数据集的 DataLoader，输入的是 (x_aux, y_aux) 原始样本。
    具体数据集与 label 映射由各自的数据模块负责。
    """
    name = priv_cfg.aux.name.lower()

    if name == "derm7pt":
        from src.data.derm7pt import build_derm7pt_dataloader

        loader = build_derm7pt_dataloader(
            root=priv_cfg.aux.root,
            split=priv_cfg.aux.split,
            img_size=getattr(cfg.data, "img_size", 224) if hasattr(cfg, "data") else 224,
            batch_size=priv_cfg.aux.batch_size,
            num_workers=priv_cfg.aux.num_workers,
            shuffle=True,
        )
        return loader
    if name == "cinic10":
        from src.data.cinic10 import build_cinic10_dataloader

        loader = build_cinic10_dataloader(
            root=priv_cfg.aux.root,
            split=priv_cfg.aux.split,
            img_size=getattr(cfg.data, "img_size", 32) if hasattr(cfg, "data") else 32,
            batch_size=priv_cfg.aux.batch_size,
            num_workers=priv_cfg.aux.num_workers,
            shuffle=True,
        )
        return loader

    raise NotImplementedError(f"Aux dataset '{name}' is not implemented in privacy.data.")


def load_victim_batch(
    cut_dir: str,
    device: str = "cuda",
) -> Optional[VictimBatch]:
    """
    从 cut_dir 下加载训练阶段记录的隐私样本。

    约定:
      - 文件名: privacy_samples.pt
      - 内容: dict with keys:
          "A_front": Tensor [N, ...]
          "y":       Tensor [N]
          "x":       Tensor [N, C, H, W] (可选)
    """
    path = os.path.join(cut_dir, "privacy_samples.pt")
    if not os.path.exists(path):
        return None

    data = torch.load(path, map_location=device)
    A_front = data["A_front"].to(device)
    y = data["y"].to(device)
    x = data.get("x", None)
    if x is not None:
        x = x.to(device)

    return VictimBatch(A_front=A_front, y=y, x=x)
