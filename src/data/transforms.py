from __future__ import annotations

from typing import Tuple

from torchvision import transforms


def build_transforms(cfg) -> Tuple[transforms.Compose, transforms.Compose]:
    """Build train/val transforms from cfg.data.

    This is intentionally lightweight and controlled by cfg.data.augmentation switches.
    The default mean/std are ImageNet's (common for ResNet/MobileNet backbones).
    """
    img_size = getattr(cfg.data, "img_size", 224)
    aug = getattr(cfg.data, "augmentation", None)

    train_tfms = []
    val_tfms = []

    # --- Spatial transforms ---
    # Train:
    if aug is not None and getattr(aug, "random_resized_crop", False):
        # Typical ImageNet-style augmentation.
        scale = getattr(aug, "random_resized_crop_scale", (0.8, 1.0))
        ratio = getattr(aug, "random_resized_crop_ratio", (3.0 / 4.0, 4.0 / 3.0))
        train_tfms.append(transforms.RandomResizedCrop(img_size, scale=scale, ratio=ratio))
    else:
        train_tfms.append(transforms.Resize((img_size, img_size)))
        if aug is not None and getattr(aug, "random_crop", False):
            padding = getattr(aug, "random_crop_padding", 4)
            train_tfms.append(transforms.RandomCrop(img_size, padding=padding))

    # Val:
    val_tfms.append(transforms.Resize((img_size, img_size)))

    # --- Simple augmentations ---
    if aug is not None:
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

    # --- ToTensor + Normalize ---
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
