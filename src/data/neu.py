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
    """Build NEU-CLS surface defect classification datasets.

    Expected layout (NEU-CLS original download):
      root/
        Cr_1.bmp
        Cr_2.bmp
        ...
        In_1.bmp
        ...
      where filename format is: "{class_name}_{index}.bmp"

    Notes:
      - class_name is parsed as the substring before the first underscore.
      - Images are loaded with PIL and converted to RGB.
    """
    import os
    from dataclasses import dataclass
    from pathlib import Path
    from typing import Dict, List, Tuple

    from PIL import Image
    from torch.utils.data import Dataset

    from src.data.splits import (
        get_default_r_aux,
        get_default_train_val_split,
        get_split_seed,
        two_stage_split_indices,
    )
    from src.data.subset import TransformedSubset
    from src.data.transforms import build_transforms

    @dataclass(frozen=True)
    class _NEURec:
        path: str
        label: int
        cls_name: str

    class _NEUFlatDataset(Dataset):
        def __init__(self, root: str) -> None:
            self.root = Path(root)
            if not self.root.exists():
                raise FileNotFoundError(f"NEU root not found: {self.root}")

            # Collect image files (support bmp as primary; allow jpg/png for robustness)
            exts = {".bmp", ".png", ".jpg", ".jpeg"}
            files: List[Path] = []
            for p in self.root.iterdir():
                if p.is_file() and p.suffix.lower() in exts:
                    files.append(p)

            if len(files) == 0:
                raise RuntimeError(
                    f"No image files found under {self.root}. "
                    "Expected flat folder with files like 'Cr_1.bmp'."
                )

            # Parse class name from filename: token before the first underscore
            cls_names: List[str] = []
            parsed: List[Tuple[Path, str]] = []
            for fp in files:
                stem = fp.stem  # e.g., "Cr_123"
                if "_" not in stem:
                    # Skip or raise; here we raise to avoid silent label mistakes
                    raise ValueError(
                        f"Invalid NEU filename (missing underscore): {fp.name}. "
                        "Expected format '{class_name}_{index}.bmp'."
                    )
                cls = stem.split("_", 1)[0]
                cls_names.append(cls)
                parsed.append((fp, cls))

            # Deterministic class ordering
            classes = sorted(set(cls_names))
            class_to_idx = {c: i for i, c in enumerate(classes)}

            records: List[_NEURec] = []
            for fp, cls in parsed:
                records.append(_NEURec(path=str(fp), label=class_to_idx[cls], cls_name=cls))

            self.records = records
            self.classes = classes
            self.class_to_idx = class_to_idx

        def __len__(self) -> int:
            return len(self.records)

        def __getitem__(self, idx: int):
            rec = self.records[idx]
            img = Image.open(rec.path).convert("RGB")
            return img, rec.label

    # -------- build base dataset + deterministic aux/train/val split --------
    root = cfg.data.root
    train_tfm, val_tfm = build_transforms(cfg)

    base = _NEUFlatDataset(root=root)

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
            "class_to_idx": base.class_to_idx,
            "layout": "flat",
            "filename_format": "{class_name}_{index}.bmp",
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
