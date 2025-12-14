# src/data/transforms.py
from typing import Tuple
from torchvision import transforms


def build_transforms(cfg) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Build train/val transforms from cfg.data.
    Supports simple augmentation switches.
    """
    img_size = getattr(cfg.data, "img_size", 224)
    aug = getattr(cfg.data, "augmentation", None)

    train_tfms = []
    val_tfms = []

    # Common resize / crop
    train_tfms += [
        transforms.Resize((img_size, img_size)),
    ]
    val_tfms += [
        transforms.Resize((img_size, img_size)),
    ]

    # Simple augmentations
    if aug is not None:
        if getattr(aug, "random_crop", False):
            padding = getattr(aug, "random_crop_padding", 4)
            train_tfms.append(transforms.RandomCrop(img_size, padding=padding))
        if getattr(aug, "random_horizontal_flip", False):
            train_tfms.append(transforms.RandomHorizontalFlip())
        if getattr(aug, "random_vertical_flip", False):
            train_tfms.append(transforms.RandomVerticalFlip())
        if getattr(aug, "color_jitter", False):
            train_tfms.append(
                transforms.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05
                )
            )

    # ToTensor + Normalize
    mean = getattr(cfg.data, "mean", [0.485, 0.456, 0.406])
    std = getattr(cfg.data, "std", [0.229, 0.224, 0.225])

    train_tfms += [
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]
    val_tfms += [
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]

    return transforms.Compose(train_tfms), transforms.Compose(val_tfms)
