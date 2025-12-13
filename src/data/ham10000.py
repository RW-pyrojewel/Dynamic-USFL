# src/data/ham10000.py
from glob import glob
import os
from typing import Tuple, Optional

import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader, random_split

from src.data.transforms import build_transforms


HAM_LABELS = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
LABEL2IDX = {l: i for i, l in enumerate(HAM_LABELS)}


class HAM10000Dataset(Dataset):
    """
    HAM10000 dataset loader.

    Expected structure (typical):
      root/
        HAM10000_metadata.csv
        images/
          ISIC_0027419.jpg
          ...

    CSV minimal columns:
      - image_id (or "image" / "image_id")
      - dx (label)
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform=None,
        metadata_csv: Optional[str] = None,
        image_dir: Optional[str] = None,
        label_col: str = "dx",
        id_col: str = "image_id",
    ):
        super().__init__()
        self.root = root
        self.split = split
        self.transform = transform

        # resolve paths
        if metadata_csv is None:
            metadata_csv = os.path.join(root, "HAM10000_metadata.csv")
        if image_dir is None:
            image_dir = os.path.join(root, "HAM10000_images_part_*")

        if not os.path.isfile(metadata_csv):
            raise FileNotFoundError(f"Metadata CSV not found: {metadata_csv}")
        if not glob(image_dir):
            raise FileNotFoundError(f"Image dir not found: {image_dir}")

        df = pd.read_csv(metadata_csv)

        # basic checks
        if id_col not in df.columns:
            raise KeyError(f"ID column '{id_col}' not found in CSV.")
        if label_col not in df.columns:
            raise KeyError(f"Label column '{label_col}' not found in CSV.")

        # store paths / labels
        self.image_ids = df[id_col].astype(str).tolist()
        raw_labels = df[label_col].astype(str).tolist()

        self.labels = [LABEL2IDX.get(l, -1) for l in raw_labels]
        if any(l == -1 for l in self.labels):
            unknown = sorted(set([raw_labels[i] for i, l in enumerate(self.labels) if l == -1]))
            raise ValueError(f"Unknown labels found in CSV: {unknown}.\n"
                             f"Please update HAM_LABELS / LABEL2IDX.")

        self.image_dir = image_dir

    def __len__(self):
        return len(self.image_ids)

    def _resolve_img_path(self, image_id: str) -> str:
        ext = '.jpg'
        for base in glob(self.image_dir):
            p = os.path.join(base, image_id + ext)
            if os.path.isfile(p):
                return p
            # 最后兜底：直接当 full name
            p = os.path.join(base, image_id)
            if os.path.isfile(p):
                return p
        raise FileNotFoundError(f"Image file not found for id={image_id}")

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        label = self.labels[idx]
        img_path = self._resolve_img_path(image_id)

        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)

        return img, label


def build_ham10000_dataloaders(cfg) -> Tuple[DataLoader, DataLoader]:
    """
    Build train/val DataLoaders for HAM10000 using cfg.data.
    """
    root = cfg.data.root
    batch_size = cfg.data.batch_size
    num_workers = getattr(cfg.data, "num_workers", 4)
    split_ratio = getattr(cfg.data, "train_val_split", [0.8, 0.2])
    seed = getattr(cfg, "seed", 42)

    train_tfm, val_tfm = build_transforms(cfg)

    full_ds = HAM10000Dataset(
        root=root,
        split="full",
        transform=None,  # split 后再分别设置
    )

    n_total = len(full_ds)
    n_train = int(n_total * split_ratio[0])
    n_val = n_total - n_train

    g = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(full_ds, [n_train, n_val], generator=g)

    # 给 split 后的 subset 赋 transform
    train_ds.dataset.transform = train_tfm
    val_ds.dataset.transform = val_tfm

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
