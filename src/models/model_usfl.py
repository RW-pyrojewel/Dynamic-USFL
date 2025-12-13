from typing import Tuple
import torch
from torch import nn
from abc import ABCMeta, abstractmethod

from src.profiling.hooks import SegmentTimer

class USFLBackbone(nn.Module, metaclass=ABCMeta):
    """
    USFL Backbone interface.
    """
    @abstractmethod
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
    
    @abstractmethod
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
        raise NotImplementedError
    