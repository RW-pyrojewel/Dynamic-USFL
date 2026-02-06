from src.models.model_usfl import USFLBackbone
from src.models.resnet_usfl import ResNet18USFLBackbone
from src.models.mobilenet_usfl import MobileNetV2USFLBackbone


def build_backbone(cfg) -> USFLBackbone:
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
    