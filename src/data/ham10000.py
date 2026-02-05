from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image
from torch.utils.data import DataLoader, Dataset

from src.data.splits import (
    get_default_r_aux,
    get_default_train_val_split,
    get_split_seed,
    two_stage_split_indices,
)
from src.data.subset import TransformedSubset
from src.data.transforms import build_transforms


@dataclass(frozen=True)
class HAMRecord:
    path: str
    label: int


class HAM10000Dataset(Dataset):
    """HAM10000 classification dataset loader.

    Expected layout (flexible):
      root/
        HAM10000_metadata.csv
        (one or more image folders, e.g.)
        HAM10000_images_part_1/*.jpg
        HAM10000_images_part_2/*.jpg

    The metadata CSV must contain at least:
      - image_id : string (filename stem)
      - dx       : class label string
    """

    def __init__(self, root: str, metadata_csv: Optional[str] = None) -> None:
        self.root = Path(root)
        self.metadata_csv = Path(metadata_csv) if metadata_csv is not None else self.root / "HAM10000_metadata.csv"
        if not self.metadata_csv.exists():
            raise FileNotFoundError(
                f"HAM10000 metadata CSV not found: {self.metadata_csv}. "
                f"Please set cfg.data.metadata_csv or place HAM10000_metadata.csv under cfg.data.root."
            )

        id_to_path = self._index_images(self.root)

        rows = self._read_metadata(self.metadata_csv)
        dx_labels = sorted({r["dx"] for r in rows})
        self.classes = dx_labels
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        records: List[HAMRecord] = []
        missing = 0
        for r in rows:
            image_id = r["image_id"]
            dx = r["dx"]
            p = id_to_path.get(image_id)
            if p is None:
                missing += 1
                continue
            records.append(HAMRecord(path=p, label=self.class_to_idx[dx]))

        if len(records) == 0:
            raise RuntimeError(
                "No HAM10000 images matched metadata. "
                "Check that image filenames match metadata image_id and are under cfg.data.root."
            )
        if missing > 0:
            # Soft warning via exception message is too aggressive; keep as meta info.
            # Users can inspect dataset length to notice mismatch.
            pass

        self.records = records

    @staticmethod
    def _read_metadata(csv_path: Path) -> List[Dict[str, str]]:
        with csv_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            required = {"image_id", "dx"}
            if not required.issubset(set(reader.fieldnames or [])):
                raise ValueError(
                    f"HAM10000 metadata CSV must contain columns {sorted(required)}; got {reader.fieldnames}"
                )
            return [row for row in reader]

    @staticmethod
    def _index_images(root: Path) -> Dict[str, str]:
        """Index images by filename stem under root."""
        exts = {".jpg", ".jpeg", ".png"}
        mapping: Dict[str, str] = {}
        for dirpath, _, filenames in os.walk(root):
            for fn in filenames:
                ext = Path(fn).suffix.lower()
                if ext not in exts:
                    continue
                stem = Path(fn).stem
                # Keep the first occurrence; dataset should not contain duplicates.
                if stem not in mapping:
                    mapping[stem] = str(Path(dirpath) / fn)
        return mapping

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple[Image.Image, int]:
        rec = self.records[idx]
        img = Image.open(rec.path).convert("RGB")
        return img, rec.label


def build_ham10000_datasets(cfg) -> Dict[str, object]:
    """Build HAM10000 datasets using deterministic aux/train/val splitting."""
    root = cfg.data.root
    metadata_csv = getattr(cfg.data, "metadata_csv", None)

    train_tfm, val_tfm = build_transforms(cfg)
    base = HAM10000Dataset(root=root, metadata_csv=metadata_csv)

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


def build_ham10000_dataloaders(cfg) -> Tuple[DataLoader, DataLoader]:
    batch_size = cfg.data.batch_size
    num_workers = getattr(cfg.data, "num_workers", 4)

    ds = build_ham10000_datasets(cfg)
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


def build_ham10000_aux_dataloader(cfg) -> DataLoader:
    aux_cfg = getattr(getattr(cfg, "privacy", object()), "aux_dataset", object())
    batch_size = getattr(aux_cfg, "batch_size", 64)
    num_workers = getattr(aux_cfg, "num_workers", getattr(cfg.data, "num_workers", 4))

    ds = build_ham10000_datasets(cfg)["aux"]
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
