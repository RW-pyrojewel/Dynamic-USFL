# src/profiling/hooks.py
from contextlib import contextmanager
from typing import Dict
import time

from torch import nn, Tensor


class SegmentTimer:
    """
    训练过程中给不同 segment 计时的简单工具。
    用法：
        timer = SegmentTimer()
        with timer.time("front"):
            ...
        with timer.time("middle"):
            ...
        times = timer.get_times()
    """
    def __init__(self) -> None:
        self._times: Dict[str, float] = {}

    @contextmanager
    def time(self, name: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            t1 = time.perf_counter()
            self._times[name] = self._times.get(name, 0.0) + (t1 - t0)

    def reset(self) -> None:
        self._times.clear()

    def get_times(self) -> Dict[str, float]:
        # 返回一个拷贝，避免外面随便改
        return dict(self._times)
    

class ActivationSizeHooks:
    """
    对模型里每个 module（或指定的一组模块）挂 forward_hook，
    记录输出 activation 的 numel，用于离线 per-layer profiling。
    """
    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self.handles = []
        # 记录格式：key -> {"count": int, "total_numel": int}
        self.records: Dict[str, Dict[str, float]] = {}

    def _hook_fn(self, module: nn.Module, inp, out: Tensor):
        name = module.__class__.__name__
        key = f"{name}_id={id(module)}"
        rec = self.records.setdefault(key, {"count": 0, "total_numel": 0})
        rec["count"] += 1
        # 对于 tensor list/tuple 也做一点适配
        if isinstance(out, (list, tuple)):
            numel = sum(o.numel() for o in out if isinstance(o, Tensor))
        else:
            numel = out.numel()
        rec["total_numel"] += numel

    def register_on_leaf_modules(self) -> None:
        """
        给所有叶子模块（没有子模块的）挂 hook。
        如果你以后只想给 backbone.layers 里的 macro blocks 挂，
        可以在外面传指定模块列表来循环注册。
        """
        self.remove()  # 先移除旧的
        for m in self.model.modules():
            if len(list(m.children())) == 0:
                h = m.register_forward_hook(self._hook_fn)
                self.handles.append(h)

    def remove(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles = []

    def reset_records(self) -> None:
        self.records.clear()

    def get_avg_numel(self) -> Dict[str, float]:
        """
        返回每个模块平均输出激活元素个数（总 numel / 次数）。
        用于 HiveMind 图里的 per-layer activation size 估计。
        """
        avg = {}
        for k, v in self.records.items():
            if v["count"] > 0:
                avg[k] = v["total_numel"] / v["count"]
        return avg
