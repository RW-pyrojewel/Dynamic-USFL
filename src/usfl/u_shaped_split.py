# src/usfl/u_shaped_split.py
from typing import Tuple, Dict, Any

from torch import Tensor

from src.models.model_usfl import USFLBackbone
from src.profiling.hooks import SegmentTimer
from src.profiling.analyzer import RoundAnalyzer


class USFLOrchestrator:
    """
    USFL orchestrator.
    - 持有一个 USFLBackbone（nn.Module）
    - 管理 (cut1, cut2)
    - 在两处切分点收集 smashed data 和 profiling
    """
    def __init__(
        self,
        backbone: USFLBackbone,
        cut1: int,
        cut2: int,
        enable_profiling: bool = True,
        bytes_per_elem: int = 4,
    ) -> None:
        self.backbone = backbone
        self.cut1 = cut1
        self.cut2 = cut2

        self.enable_profiling = enable_profiling
        self.timer = SegmentTimer() if enable_profiling else None
        self.round_analyzer = RoundAnalyzer(bytes_per_elem=bytes_per_elem)
        
        self.last_front_acts = None
        self.last_back_acts = None

    def set_cuts(self, cut1: int, cut2: int) -> None:
        self.cut1, self.cut2 = cut1, cut2

    def forward_three_segments(
        self,
        x: Tensor,
    ) -> Tuple[Tensor, Dict[str, Any]]:
        """
        执行 front -> middle -> back 的前向。

        返回:
          - logits: 最终输出
          - profiling: dict，包含 RoundAnalyzer 总结后的
                       "comm_bytes", "comp_time", "t_front", ... 等字段
        """
        raw_prof: Dict[str, Any] = {}
        self.timer.reset() if self.timer is not None else None

        # 利用 backbone.forward_segments 一次性跑完三段，
        # 同时用 SegmentTimer 给三段 forward 计时。
        z_front, z_middle, z_back = self.backbone.forward_segments(x, self.cut1, self.cut2, self.timer)

        # 在 cut1 / cut2 处截取 smashed data
        smashed1 = z_front.detach()
        smashed2 = z_middle.detach()
        raw_prof["smashed1_numel"] = smashed1.numel() if self.cut1 != self.cut2 else 0
        raw_prof["smashed2_numel"] = smashed2.numel() if self.cut1 != self.cut2 else 0
        raw_prof["cut_points"] = (self.cut1, self.cut2)
        
        # 供收集 smashed data 以及 smashed data's gradient 之用
        self.last_front_acts = smashed1
        self.last_back_acts = z_middle
        self.last_back_acts.retain_grad()

        if self.timer is not None:
            times = self.timer.get_times()
            raw_prof["t_front"] = times.get("front", 0.0)
            raw_prof["t_middle"] = times.get("middle", 0.0) if self.cut1 != self.cut2 else 0.0
            raw_prof["t_back"] = times.get("back", 0.0)
        else:
            raw_prof["t_front"] = 0.0
            raw_prof["t_middle"] = 0.0
            raw_prof["t_back"] = 0.0

        # 交给 RoundAnalyzer 计算 round 级指标
        summary = self.round_analyzer.summarize(raw_prof)

        # profiling 返回综合信息：raw + summary
        profiling = {**raw_prof, **summary}

        logits = z_back
        return logits, profiling
