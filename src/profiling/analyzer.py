# src/profiling/analyzer.py
from typing import Dict, Any


class RoundAnalyzer:
    """
    将 StaticSplitUSFL 收集的 raw profiling 字典转成统一的 round 指标。
    假设 raw 里至少包含：
      - "smashed1_numel"
      - "smashed2_numel"
      - "t_front", "t_middle", "t_back"
    """
    def __init__(self, bytes_per_elem: int = 4) -> None:
        self.bytes_per_elem = bytes_per_elem

    def summarize(self, raw: Dict[str, Any]) -> Dict[str, float]:
        smashed1 = int(raw.get("smashed1_numel", 0))
        smashed2 = int(raw.get("smashed2_numel", 0))
        bytes_up = bytes_down = (smashed1 + smashed2) * self.bytes_per_elem

        t_front = float(raw.get("t_front", 0.0))
        t_middle = float(raw.get("t_middle", 0.0))
        t_back = float(raw.get("t_back", 0.0))
        comp_time_client = t_front + t_back
        comp_time_server = t_middle

        # 返回你在 train_loop 里会用到的字段
        return {
            "bytes_up": bytes_up,
            "bytes_down": bytes_down,
            "comp_time_client": comp_time_client,
            "comp_time_server": comp_time_server,
            "t_front": t_front,
            "t_middle": t_middle,
            "t_back": t_back,
            "smashed1_numel": float(smashed1),
            "smashed2_numel": float(smashed2),
        }
