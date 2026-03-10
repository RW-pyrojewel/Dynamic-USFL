from src.models.model_usfl import USFLBackbone
from src.models.resnet_usfl import ResNet18USFLBackbone
from src.models.mobilenet_usfl import MobileNetV2USFLBackbone


def build_backbone(cfg) -> USFLBackbone:
    """根据 cfg.model.backbone 选择对应的 USFLBackbone 实现。"""
    backbone_name = cfg.model.backbone.lower()
    num_classes = cfg.data.num_classes
    img_size = cfg.data.img_size
    pretrained = getattr(cfg.model, "pretrained", False)
    
    small_input = img_size is not None and img_size <= 64

    if backbone_name == "resnet18":
        return ResNet18USFLBackbone(num_classes=num_classes, pretrained=pretrained, small_input=small_input)
    elif backbone_name == "mobilenetv2":
        return MobileNetV2USFLBackbone(num_classes=num_classes, pretrained=pretrained, small_input=small_input)
    else:
        raise ValueError(f"Unsupported backbone: {cfg.model.backbone}")
    