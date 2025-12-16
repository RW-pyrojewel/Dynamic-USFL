from typing import Tuple
import torch
from torch import nn
from torchvision import models

from src.models.model_usfl import USFLBackbone
from src.profiling.hooks import SegmentTimer


class ResNet18USFLBackbone(USFLBackbone):
    """
    ResNet-18 backbone that can be logically split into macro blocks.
    """
    def __init__(self, num_classes: int = 1000, pretrained: bool = False) -> None:
        super().__init__()

        base = models.resnet18(pretrained=pretrained)

        conv1_block = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)
        layer1 = base.layer1
        layer2 = base.layer2
        layer3 = base.layer3
        layer4 = base.layer4
        avgpool = base.avgpool
        fc = nn.Sequential(nn.Flatten(), nn.Linear(512, num_classes))

        self.layers = nn.ModuleList(
            [conv1_block, layer1, layer2, layer3, layer4, avgpool, fc]
        )

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
    
