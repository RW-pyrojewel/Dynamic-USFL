# src/data/cifar10.py
from typing import Tuple
from torch.utils.data import DataLoader
from torchvision import datasets

from src.data.transforms import build_transforms


def build_cifar10_dataloaders(cfg) -> Tuple[DataLoader, DataLoader]:
    root = cfg.data.root
    batch_size = cfg.data.batch_size
    num_workers = getattr(cfg.data, "num_workers", 4)

    train_tfm, val_tfm = build_transforms(cfg)

    train_ds = datasets.CIFAR10(
        root=root,
        train=True,
        download=True,
        transform=train_tfm,
    )
    val_ds = datasets.CIFAR10(
        root=root,
        train=False,
        download=True,
        transform=val_tfm,
    )

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
