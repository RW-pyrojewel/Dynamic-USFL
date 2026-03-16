# src/privacy/data.py

from dataclasses import dataclass
from typing import Optional

import os
import torch
from torch.utils.data import DataLoader, Subset

from .config import PrivacyConfig


@dataclass
class VictimBatch:
    A_front: torch.Tensor                       # [N, ...]
    y: torch.Tensor                             # [N]
    x: Optional[torch.Tensor] = None            # [N, C, H, W] or None
    grad_A_back: Optional[torch.Tensor] = None  # [N, ...]

    @property
    def num_samples(self) -> int:
        return int(self.A_front.shape[0])


def _get_seed(cfg, default: int = 42) -> int:
    """Best-effort seed extraction (kept local to avoid extra deps)."""
    for key in (("seed", "torch"), ("seed", "master")):
        cur = cfg
        ok = True
        for k in key:
            if hasattr(cur, k):
                cur = getattr(cur, k)
            elif isinstance(cur, dict) and k in cur:
                cur = cur[k]
            else:
                ok = False
                break
        if ok and isinstance(cur, int):
            return int(cur)
    return int(default)


def _maybe_limit_dataset(ds, max_samples: Optional[int], seed: int):
    """Limit dataset size deterministically if max_samples is set."""
    if max_samples is None:
        return ds
    try:
        max_samples = int(max_samples)
    except Exception:
        return ds
    if max_samples <= 0:
        return ds
    n = len(ds)
    if n <= max_samples:
        return ds

    g = torch.Generator()
    g.manual_seed(int(seed))
    idx = torch.randperm(n, generator=g)[:max_samples].tolist()
    return Subset(ds, idx)


def build_aux_loader(cfg, priv_cfg: PrivacyConfig) -> DataLoader:
    """
    构建辅助数据集的 DataLoader，输入的是 (x_aux, y_aux) 原始样本。

    兼容两种来源：
      (A) 规律挖掘实验（fixed-ratio aux split）：
          - aux 数据来自 *与 victim 同域的数据集*，按 r_aux 固定比例划分（seed.torch 固定）
          - 直接复用 src.data 的 build_aux_dataloader(cfg)
      (B) 历史兼容（外部 aux 数据集）：
          - 例如 derm7pt / cinic10，仍沿用原有各自的数据模块构建函数

    说明：
      - 不改变本函数接口；
      - 当 priv_cfg.aux.name 与 cfg.data.dataset 相同时，会优先走 (A)；
      - 否则退回 (B)。
    """
    name = priv_cfg.aux.name.lower()
    main_name = getattr(getattr(cfg, "data", object()), "dataset", "").lower()

    # ---------- (A) Fixed-ratio aux split: delegate to src.data ----------
    try:
        from src.data import build_aux_dataloader as _build_aux_dataloader
    except Exception:
        _build_aux_dataloader = None

    if _build_aux_dataloader is not None and name == main_name:
        loader = _build_aux_dataloader(cfg)

        # Optional deterministic truncation (per-cut max_samples)
        seed = _get_seed(cfg)
        max_samples = priv_cfg.aux.max_samples
        ds = _maybe_limit_dataset(loader.dataset, max_samples=max_samples, seed=seed + 12345)

        if ds is loader.dataset:
            return loader

        batch_size = priv_cfg.aux.batch_size
        num_workers = priv_cfg.aux.num_workers
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
        )

    # ---------- (B) Legacy external aux datasets ----------
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

    raise NotImplementedError(
        f"Aux dataset '{name}' is not implemented in privacy.data. "
        f"Supported ratio-split datasets: cifar10/ham10000/eurosat/neu; "
        f"legacy datasets: derm7pt/cinic10."
    )


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
    grad_A_back = data.get("grad_A_back", None)
    if x is not None:
        x = x.to(device)
    if grad_A_back is not None:
        grad_A_back = grad_A_back.to(device)

    return VictimBatch(A_front=A_front, y=y, x=x, grad_A_back=grad_A_back)
