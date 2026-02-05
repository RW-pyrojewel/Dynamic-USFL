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


def build_cifar10_datasets(cfg) -> Dict[str, object]:
    """Build CIFAR-10 datasets following the project's split convention.

    If cfg.data.train_val_split is provided, we will:
      - build a base training pool from CIFAR-10 train split
      - split it into aux / train / val using r_aux and train_val_split

    Otherwise, we fallback to the classic CIFAR-10 train/test split:
      - train = CIFAR-10 train
      - val   = CIFAR-10 test
      - aux   = a split from CIFAR-10 train (still supported)
    """
    root = cfg.data.root

    train_tfm, val_tfm = build_transforms(cfg)

    # Base pools without transform; transforms are applied by TransformedSubset.
    train_pool = datasets.CIFAR10(
        root=root,
        train=True,
        download=True,
        transform=None,
    )

    tv = getattr(cfg.data, "train_val_split", None)
    seed = get_split_seed(cfg)
    r_aux = get_default_r_aux(cfg)

    if tv is None:
        # Fallback: use official test split as val.
        test_pool = datasets.CIFAR10(
            root=root,
            train=False,
            download=True,
            transform=None,
        )

        aux_idx, train_idx, _ = two_stage_split_indices(
            total=len(train_pool),
            r_aux=r_aux,
            train_val_split=(0.999, 0.001),  # tiny placeholder, val comes from test_pool
            seed=seed,
        )

        out = {
            "train": TransformedSubset(train_pool, train_idx, transform=train_tfm),
            "val": TransformedSubset(test_pool, range(len(test_pool)), transform=val_tfm),
            "aux": TransformedSubset(train_pool, aux_idx, transform=val_tfm),
            "meta": {
                "use_official_test_as_val": True,
                "r_aux": r_aux,
                "seed": seed,
            },
        }
        return out

    train_ratio, val_ratio = get_default_train_val_split(cfg)
    split = two_stage_split_indices(
        total=len(train_pool),
        r_aux=r_aux,
        train_val_split=(train_ratio, val_ratio),
        seed=seed,
    )

    return {
        "train": TransformedSubset(train_pool, split.train, transform=train_tfm),
        "val": TransformedSubset(train_pool, split.val, transform=val_tfm),
        "aux": TransformedSubset(train_pool, split.aux, transform=val_tfm),
        "meta": {
            "use_official_test_as_val": False,
            "r_aux": r_aux,
            "seed": seed,
            "train_val_split": (train_ratio, val_ratio),
        },
    }


def build_cifar10_dataloaders(cfg) -> Tuple[DataLoader, DataLoader]:
    """Return (train_loader, val_loader)."""
    batch_size = cfg.data.batch_size
    num_workers = getattr(cfg.data, "num_workers", 4)

    ds = build_cifar10_datasets(cfg)
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


def build_cifar10_aux_dataloader(cfg) -> DataLoader:
    """Build the auxiliary labeled dataloader used by SAE-USL attacks."""
    batch_size = getattr(getattr(cfg, "privacy", object()), "aux_dataset", object())
    batch_size = getattr(batch_size, "batch_size", 64)
    num_workers = getattr(getattr(cfg, "privacy", object()), "aux_dataset", object())
    num_workers = getattr(num_workers, "num_workers", getattr(cfg.data, "num_workers", 4))

    ds = build_cifar10_datasets(cfg)["aux"]
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
