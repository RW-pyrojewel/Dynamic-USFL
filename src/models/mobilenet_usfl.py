from typing import Tuple
import torch
from torch import nn
from torchvision import models

from src.models.model_usfl import USFLBackbone
from src.profiling.hooks import SegmentTimer


class MobileNetV2USFLBackbone(USFLBackbone):
    """
    MobileNetV2 backbone that can be logically split into macro blocks.
    """
    def __init__(self, num_classes: int = 1000, pretrained: bool = False) -> None:
        super().__init__()

        base = models.mobilenet_v2(pretrained=pretrained)

        features = list(base.features.children())
        # 原始 MobileNet 在 classifier 前有全局平均池化与展平操作，
        # 这里补回 AdaptiveAvgPool2d + Flatten，以确保 Linear 收到形状 (N, 1280)
        classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(1280, num_classes),
        )

        # 按照 MobileNetV2 的结构划分为若干块
        self.layers = nn.ModuleList()
        self.layers.append(nn.Sequential(*features[0:2]))   # 初始卷积层
        self.layers.append(nn.Sequential(*features[2:4]))   # 第一组 inverted residuals
        self.layers.append(nn.Sequential(*features[4:7]))   # 第二组 inverted residuals
        self.layers.append(nn.Sequential(*features[7:14]))  # 第三组 inverted residuals
        self.layers.append(nn.Sequential(*features[14:18])) # 第四组 inverted residuals
        self.layers.append(nn.Sequential(*features[18:19])) # 最后一个卷积层
        self.layers.append(classifier)                       # 分类器

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.layers:
            x = block(x)
        return x

    def forward_segments(
        self,
        x: torch.Tensor,
        cut1: int,
        cut2: int,
        timer: SegmentTimer = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        根据 (cut1, cut2) 把前向划成三段：
          - front  : [0, cut1]
          - middle : (cut1, cut2]
          - back   : (cut2, L)
        """
        cut1 %= len(self.layers)
        cut2 %= len(self.layers)
        
        assert 0 <= cut1 <= cut2 < len(self.layers) - 1, f"Invalid cuts: {(cut1, cut2)}. Layer count: {len(self.layers)}"

        z = x
        z_front, z_middle, z_back = None, None, None
        
        if timer is not None:
            with timer.time("front"):
                for i in range(cut1 + 1):
                    z = self.layers[i](z)
                z_front = z
            with timer.time("middle"):
                for i in range(cut1 + 1, cut2 + 1):
                    z = self.layers[i](z)
                z_middle = z
            with timer.time("back"):
                for i in range(cut2 + 1, len(self.layers)):
                    z = self.layers[i](z)
                z_back = z
        else:
            for i in range(cut1 + 1):
                z = self.layers[i](z)
            z_front = z
            for i in range(cut1 + 1, cut2 + 1):
                z = self.layers[i](z)
            z_middle = z         
            for i in range(cut2 + 1, len(self.layers)):
                z = self.layers[i](z)
            z_back = z
        
        return z_front, z_middle, z_back
    