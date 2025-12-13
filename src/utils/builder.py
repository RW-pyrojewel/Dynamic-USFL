# src/utils/builder.py
"""
Utility functions to build dataloaders and models based on config.
"""

from src.data.cifar10 import build_cifar10_dataloaders
from src.data.ham10000 import build_ham10000_dataloaders
from src.models.mobilenet_usfl import MobileNetV2USFLBackbone
from src.models.resnet_usfl import ResNet18USFLBackbone


def build_dataloaders(cfg):
    """根据 cfg.data.dataset 选择对应的 dataloader 构建函数。"""
    name = cfg.data.dataset.lower()
    if name == "ham10000":
        return build_ham10000_dataloaders(cfg)
    elif name == "cifar10":
        return build_cifar10_dataloaders(cfg)
    else:
        raise ValueError(f"Unsupported dataset: {cfg.data.dataset}")


def build_backbone(cfg):
    """根据 cfg.model.backbone 选择对应的 USFLBackbone 实现。"""
    backbone_name = cfg.model.backbone.lower()
    num_classes = cfg.data.num_classes
    pretrained = getattr(cfg.model, "pretrained", False)

    if backbone_name == "resnet18":
        return ResNet18USFLBackbone(num_classes=num_classes, pretrained=pretrained)
    elif backbone_name == "mobilenetv2":
        return MobileNetV2USFLBackbone(num_classes=num_classes, pretrained=pretrained)
    else:
        raise ValueError(f"Unsupported backbone: {cfg.model.backbone}")
    