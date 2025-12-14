# src/data/cinic10.py

from __future__ import annotations

import os
from typing import Optional, List

from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms
from torchvision.datasets import ImageFolder


# CINIC-10 published channel stats (authors' repo)
# mean: [0.47889522, 0.47227842, 0.43047404]
# std : [0.24205776, 0.23828046, 0.25874835]
# See: https://github.com/BayesWatch/cinic-10 
CINIC10_MEAN = (0.47889522, 0.47227842, 0.43047404)
CINIC10_STD = (0.24205776, 0.23828046, 0.25874835)

# Common split dirnames seen in the wild
_SPLIT_ALIASES = {
    "train": ["train"],
    "valid": ["valid", "validate", "validation", "val"],
    "test": ["test"],
}


def _resolve_split_dir(root: str, split: str) -> List[str]:
    """
    Return a list of candidate directories for a split.
    Supports:
      - 'train', 'valid'/'validate'/'val'/'validation', 'test'
      - 'train+valid' / 'train+validate' / 'train+val' / 'train+validation'
    """
    s = split.strip().lower().replace(" ", "")

    # Combined split
    if "+" in s:
        parts = s.split("+")
        if len(parts) != 2:
            raise ValueError(f"Unsupported CINIC-10 split: {split}")
        left, right = parts[0], parts[1]
        left_dirs = _resolve_split_dir(root, left)
        right_dirs = _resolve_split_dir(root, right)
        # pick the first existing dir for each side
        chosen = []
        for cand_list in (left_dirs, right_dirs):
            found = None
            for d in cand_list:
                if os.path.isdir(d):
                    found = d
                    break
            if found is None:
                raise FileNotFoundError(
                    f"CINIC-10 split '{split}': none of these dirs exist: {cand_list}"
                )
            chosen.append(found)
        return chosen

    # Single split
    if s in _SPLIT_ALIASES:
        return [os.path.join(root, name) for name in _SPLIT_ALIASES[s]]

    # Accept explicit directory name
    return [os.path.join(root, s)]


def _build_transform(img_size: int, train: bool, use_augmentation: bool) -> transforms.Compose:
    tfms = []
    if train and use_augmentation:
        # CINIC-10 images are 32x32 like CIFAR-10
        if img_size != 32:
            # keep behavior predictable; you can always override upstream
            tfms.append(transforms.Resize((img_size, img_size)))
        else:
            tfms.append(transforms.RandomCrop(32, padding=4))
        tfms.append(transforms.RandomHorizontalFlip())
    else:
        if img_size != 32:
            tfms.append(transforms.Resize((img_size, img_size)))

    tfms.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=CINIC10_MEAN, std=CINIC10_STD),
        ]
    )
    return transforms.Compose(tfms)


def build_cinic10_dataloader(
    root: str,
    split: str = "train",
    batch_size: int = 128,
    num_workers: int = 4,
    img_size: int = 32,
    use_augmentation: bool = True,
    shuffle: Optional[bool] = None,
    pin_memory: bool = True,
    drop_last: bool = False,
) -> DataLoader:
    """
    Build CINIC-10 DataLoader.

    Expected directory structure (authors' repo):
        root/
          train/<class>/*.png
          valid(or validate)/<class>/*.png
          test/<class>/*.png
    

    Parameters
    ----------
    root : str
        Path to CINIC-10 root directory.
    split : str
        'train', 'valid' (or 'validate'/'val'/'validation'), 'test',
        or 'train+valid' (aliases supported).
    """
    split_dirs = _resolve_split_dir(root, split)

    # Combined split returns two dirs; otherwise we may get alias candidates.
    # For single split: pick the first existing directory among candidates.
    datasets = []

    if len(split_dirs) == 2 and "+" in split.replace(" ", "").lower():
        chosen_dirs = split_dirs
    else:
        chosen = None
        for d in split_dirs:
            if os.path.isdir(d):
                chosen = d
                break
        if chosen is None:
            raise FileNotFoundError(
                f"CINIC-10 split '{split}': none of these dirs exist: {split_dirs}"
            )
        chosen_dirs = [chosen]

    is_train = ("train" in split.lower()) and ("test" not in split.lower())
    tfm = _build_transform(img_size=img_size, train=is_train, use_augmentation=use_augmentation)

    for d in chosen_dirs:
        datasets.append(ImageFolder(root=d, transform=tfm))

    ds = datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)

    if shuffle is None:
        shuffle = is_train

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )
