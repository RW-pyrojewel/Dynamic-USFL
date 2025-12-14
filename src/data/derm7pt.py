# src/data/derm7pt.py
# -*- coding: utf-8 -*-
import os
import csv
from collections import Counter
from typing import Optional, Tuple, List, Set

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

__all__ = [
    "HAM_CLASSES",
    "HAM_CLASS_TO_IDX",
    "Derm7ptDataset",
    "build_derm7pt_dataloader",
]

# -------------------------
# 1. HAM10000 标签空间
# -------------------------

HAM_CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
HAM_CLASS_TO_IDX = {c: i for i, c in enumerate(HAM_CLASSES)}


def map_derm7pt_to_ham(raw_label: str) -> Optional[str]:
    """
    把 Derm7pt 的 diagnosis 映射到 HAM10000 的 7 类标签。
    未能映射则返回 None（例如 miscellaneous）。
    """
    if raw_label is None:
        return None
    s = str(raw_label).strip().lower()

    # 1) melanoma 系（避免把 melanocytic nevus 误判为 melanoma）
    if "melanoma" in s and "nevus" not in s and "naevus" not in s:
        # e.g., "melanoma", "melanoma in situ", ...
        return "mel"

    # 2) 各种 nevus / naevus -> nv
    if "nevus" in s or "naevus" in s:
        return "nv"

    # 3) basal cell carcinoma -> bcc
    if "basal cell carcinoma" in s or "bcc" in s:
        return "bcc"

    # 4) actinic / squamous / bowen -> akiec
    if (
        "actinic keratosis" in s
        or "bowen" in s
        or "intraepithelial carcinoma" in s
        or "squamous cell carcinoma" in s
    ):
        return "akiec"

    # 5) benign keratosis-like -> bkl
    if (
        "seborrheic keratosis" in s
        or "seborrhoeic keratosis" in s
        or "lichenoid keratosis" in s
        or "solar lentigo" in s
        or "lentigo simplex" in s
        or (s.startswith("lentigo") and "maligna" not in s)
    ):
        return "bkl"

    # 6) dermatofibroma -> df
    if "dermatofibroma" in s:
        return "df"

    # 7) vascular -> vasc
    if (
        "hemangioma" in s
        or "angioma" in s
        or "angiokeratoma" in s
        or "pyogenic granuloma" in s
        or "vascular" in s
    ):
        return "vasc"

    # 8) miscellaneous 等其它 -> 不映射，丢弃
    return None


# -------------------------
# 2. 工具函数
# -------------------------

def _build_default_transform(img_size: int):
    """与主任务大致一致的预处理。"""
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def _load_exclude_ids(path_txt: Optional[str]) -> Set[str]:
    """从 txt 读取需要排除的 image_id（例如 HAM10000 的 id 做去重）。"""
    if not path_txt or not os.path.exists(path_txt):
        return set()
    ids = set()
    with open(path_txt, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                ids.add(s)
    return ids


def _load_split_ids(index_csv: str) -> Set[str]:
    """
    从 train_indexes.csv / valid_indexes.csv / test_indexes.csv
    读取当前 split 使用的 image_id 集合。

    尝试的列名优先级：
      - "image_id"
      - "isic_id"
      - 其它：默认第一列
    """
    if not os.path.isfile(index_csv):
        raise FileNotFoundError(f"Split index file not found: {index_csv}")

    split_ids: Set[str] = set()
    with open(index_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise RuntimeError(f"Empty index CSV: {index_csv}")

        # 猜测 id 列
        fieldnames = [c for c in reader.fieldnames if c is not None]
        id_col = None
        for cand in ("image_id", "isic_id", "indexes", "id"):
            if cand in fieldnames:
                id_col = cand
                break
        if id_col is None:
            id_col = fieldnames[0]  # 兜底：第一列

        for row in reader:
            v = (row.get(id_col, "") or "").strip()
            if v:
                split_ids.add(v)

    return split_ids


# -------------------------
# 3. Dataset 实现
# -------------------------

class Derm7ptDataset(Dataset):
    """
    Derm7pt 辅助数据集，用于映射到 HAM10000 的 7 类标签。

    默认目录结构（root 即 Derm7pt_release_v0）::

        root/
          images/
          meta/
            meta.csv
            train_indexes.csv
            valid_indexes.csv
            test_indexes.csv

    meta.csv 至少包含：
      - case_num            : 图片 ID（不带扩展名）
      - diagnosis           : 诊断名称
      - clinic / derm       : 图像相对路径（本模块默认使用 derm 列）
    """

    def __init__(
        self,
        root: str,
        meta_csv: Optional[str] = None,
        split: str = "all",          # "train"/"valid"/"val"/"test"/"all"
        img_size: int = 224,
        transform=None,
        exclude_ids_txt: Optional[str] = None,
        id_col: str = "case_num",
        label_col: str = "diagnosis",
        path_col: str = "derm",      # 关键：使用 derm 通道
        return_ids: bool = False,
    ) -> None:
        super().__init__()
        self.root = root
        self.return_ids = return_ids

        if meta_csv is None:
            meta_csv = os.path.join(root, "meta", "meta.csv")
        self.meta_csv = meta_csv

        # split -> 对应的 indexes 文件
        split = split.lower()
        if split in {"val"}:
            split = "valid"
        if split in {"train", "valid", "test"}:
            index_dir = os.path.dirname(self.meta_csv)
            index_csv = os.path.join(index_dir, f"{split}_indexes.csv")
            split_ids = _load_split_ids(index_csv)
        else:
            split_ids = None   # "all": 不做划分过滤

        if transform is None:
            transform = _build_default_transform(img_size)
        self.transform = transform

        exclude_ids = _load_exclude_ids(exclude_ids_txt)

        # ------------ 读取 meta.csv & 构造样本列表 ------------
        if not os.path.isfile(self.meta_csv):
            raise FileNotFoundError(f"meta.csv not found: {self.meta_csv}")

        samples: List[Tuple[str, str, str, int]] = []

        with open(self.meta_csv, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                image_id = (row.get(id_col, "") or "").strip()
                if not image_id:
                    continue

                # 先按 split 文件过滤
                if split_ids is not None and image_id not in split_ids:
                    continue

                # 再按 exclude_ids 过滤（例如和 HAM10000 重复的）
                if image_id in exclude_ids:
                    continue

                raw_label = (row.get(label_col, "") or "").strip()
                if not raw_label:
                    continue

                ham_label = map_derm7pt_to_ham(raw_label)
                if ham_label is None:
                    # miscellaneous 等，直接跳过
                    continue

                # derm 路径（相对 root）
                rel_path = (row.get(path_col, "") or "").strip()
                if not rel_path:
                    # 有些行可能 derm 为空，可以选择跳过或 fallback 到 clinic，这里先跳过
                    continue
                img_path = os.path.join(root, "images", rel_path)
                if not os.path.isfile(img_path):
                    continue

                label_idx = HAM_CLASS_TO_IDX[ham_label]
                samples.append((image_id, img_path, ham_label, label_idx))

        if len(samples) == 0:
            raise RuntimeError(
                "No valid Derm7pt samples after mapping / split / filtering. "
                "Check meta.csv / *_indexes.csv / 列名 / 路径配置。"
            )

        # 简单打印一下分布，方便 sanity check
        cls_counts = Counter([s[2] for s in samples])
        print(f"[Derm7ptDataset] split={split}  class counts:", dict(cls_counts))
        print(f"[Derm7ptDataset] total samples: {len(samples)}")

        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        image_id, img_path, _, label_idx = self.samples[idx]

        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)

        if self.return_ids:
            return img, label_idx, image_id
        return img, label_idx


# -------------------------
# 4. DataLoader 构造函数
# -------------------------

def build_derm7pt_dataloader(
    root: str,
    split: str = "train",         # train / valid / test / all
    img_size: int = 224,
    batch_size: int = 128,
    num_workers: int = 4,
    shuffle: bool = True,
    exclude_ids_txt: Optional[str] = None,
    return_ids: bool = False,
) -> DataLoader:
    """
    简单封装，方便直接构造 DataLoader。

    示例：
        aux_loader = build_derm7pt_dataloader(
            root="./data/Derm7pt_release_v0",
            split="train",   # 或 "valid" / "test" / "all"
            batch_size=256,
            img_size=224,
        )
    """
    dataset = Derm7ptDataset(
        root=root,
        meta_csv=None,
        split=split,
        img_size=img_size,
        transform=None,
        exclude_ids_txt=exclude_ids_txt,
        id_col="case_num",
        label_col="diagnosis",
        path_col="derm",          # 关键：使用 derm 通道
        return_ids=return_ids,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )
    return loader
