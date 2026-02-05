from __future__ import annotations

from typing import Dict, Tuple

from torch.utils.data import DataLoader
from torchvision import datasets

from src.data.splits import (
    get_default_r_aux,
    get_default_train_val_split,
    get_split_seed,
    two_stage_split_indices,
)
from src.data.subset import TransformedSubset
from src.data.transforms import build_transforms


def build_neu_datasets(cfg) -> Dict[str, object]:
    """Build NEU surface defect classification datasets.

    Expected directory layout under cfg.data.root:
      root/
        class_0_name/
          xxx.jpg
          ...
        class_1_name/
          ...
        ...

    We use ImageFolder and perform deterministic aux/train/val splitting.
    """
    root = cfg.data.root
    train_tfm, val_tfm = build_transforms(cfg)

    base = datasets.ImageFolder(root=root, transform=None)

    seed = get_split_seed(cfg)
    r_aux = get_default_r_aux(cfg)
    train_ratio, val_ratio = get_default_train_val_split(cfg)

    split = two_stage_split_indices(
        total=len(base),
        r_aux=r_aux,
        train_val_split=(train_ratio, val_ratio),
        seed=seed,
    )

    return {
        "train": TransformedSubset(base, split.train, transform=train_tfm),
        "val": TransformedSubset(base, split.val, transform=val_tfm),
        "aux": TransformedSubset(base, split.aux, transform=val_tfm),
        "meta": {
            "r_aux": r_aux,
            "seed": seed,
            "train_val_split": (train_ratio, val_ratio),
            "classes": base.classes,
        },
    }


def build_neu_dataloaders(cfg) -> Tuple[DataLoader, DataLoader]:
    batch_size = cfg.data.batch_size
    num_workers = getattr(cfg.data, "num_workers", 4)

    ds = build_neu_datasets(cfg)
    train_ds = ds["train"]
    val_ds = ds["val"]

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    return train_loader, val_loader


def build_neu_aux_dataloader(cfg) -> DataLoader:
    aux_cfg = getattr(getattr(cfg, "privacy", object()), "aux_dataset", object())
    batch_size = getattr(aux_cfg, "batch_size", 64)
    num_workers = getattr(aux_cfg, "num_workers", getattr(cfg.data, "num_workers", 4))

    ds = build_neu_datasets(cfg)["aux"]
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
