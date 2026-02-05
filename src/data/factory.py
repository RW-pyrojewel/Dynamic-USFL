from __future__ import annotations

from typing import Callable, Dict, Tuple

from torch.utils.data import DataLoader


def _get_dataset_name(cfg) -> str:
    name = getattr(cfg.data, "dataset", None)
    if name is None:
        raise ValueError("cfg.data.dataset is required (e.g., 'cifar10', 'ham10000', 'eurosat', 'neu').")
    return str(name).lower()


def build_dataloaders(cfg) -> Tuple[DataLoader, DataLoader]:
    """Factory: build (train_loader, val_loader) for cfg.data.dataset."""
    name = _get_dataset_name(cfg)

    if name == "cifar10":
        from src.data.cifar10 import build_cifar10_dataloaders

        return build_cifar10_dataloaders(cfg)
    if name == "ham10000":
        from src.data.ham10000 import build_ham10000_dataloaders

        return build_ham10000_dataloaders(cfg)
    if name == "eurosat":
        from src.data.eurosat import build_eurosat_dataloaders

        return build_eurosat_dataloaders(cfg)
    if name in ("neu", "neu_surface", "neu_defect"):
        from src.data.neu import build_neu_dataloaders

        return build_neu_dataloaders(cfg)

    raise ValueError(f"Unknown dataset: {name}")


def build_aux_dataloader(cfg) -> DataLoader:
    """Factory: build auxiliary labeled dataloader for SAE-USL attacks."""
    name = _get_dataset_name(cfg)

    if name == "cifar10":
        from src.data.cifar10 import build_cifar10_aux_dataloader

        return build_cifar10_aux_dataloader(cfg)
    if name == "ham10000":
        from src.data.ham10000 import build_ham10000_aux_dataloader

        return build_ham10000_aux_dataloader(cfg)
    if name == "eurosat":
        from src.data.eurosat import build_eurosat_aux_dataloader

        return build_eurosat_aux_dataloader(cfg)
    if name in ("neu", "neu_surface", "neu_defect"):
        from src.data.neu import build_neu_aux_dataloader

        return build_neu_aux_dataloader(cfg)

    raise ValueError(f"Unknown dataset: {name}")


def build_datasets(cfg) -> Dict[str, object]:
    """Factory: build dataset dict for cfg.data.dataset.

    Returns a dict with keys: 'train', 'val', 'aux', 'meta'.
    """
    name = _get_dataset_name(cfg)

    if name == "cifar10":
        from src.data.cifar10 import build_cifar10_datasets

        return build_cifar10_datasets(cfg)
    if name == "ham10000":
        from src.data.ham10000 import build_ham10000_datasets

        return build_ham10000_datasets(cfg)
    if name == "eurosat":
        from src.data.eurosat import build_eurosat_datasets

        return build_eurosat_datasets(cfg)
    if name in ("neu", "neu_surface", "neu_defect"):
        from src.data.neu import build_neu_datasets

        return build_neu_datasets(cfg)

    raise ValueError(f"Unknown dataset: {name}")
