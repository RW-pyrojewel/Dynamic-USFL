# src/network/simulator.py
from __future__ import annotations

import csv
from dataclasses import dataclass
from typing import List, Optional, Tuple, Any
import math
import random


@dataclass
class LinkProfile:
    """
    单个链路配置：上/下行带宽 + RTT。

    所有带宽单位：Mbps（Megabits per second）
    RTT 单位：毫秒
    """
    name: str
    bw_up_mbps: float
    bw_down_mbps: float
    rtt_ms: float


class NetworkSimulator:
    """
    一个简单的网络仿真器，用于估计通信时延。

    支持三种模式：
      - fixed: 所有 client 用同一组 (bw_up, bw_down, rtt)，无时间波动
      - per_client: 每个 client 有自己的 profile，时间上仍可加波动
      - heterogeneous: 从一个 profile 列表按概率为 client 抽取 profile，再加时间波动

    时间波动使用一个很简单的 AR(1) 过程：
      cap_t = cap_mean * (1 + noise_t)
      noise_t = rho * noise_{t-1} + sqrt(1 - rho^2) * eps_t, eps_t ~ N(0, jitter_std^2)
    """

    def __init__(
        self,
        profiles: List[LinkProfile],
        num_clients: int,
        mode: str = "fixed",
        profile_probs: Optional[List[float]] = None,
        temporal_corr: float = 0.0,
        jitter_std: float = 0.0,
        seed: Optional[int] = None,
        net_csv: Optional[str] = None,
    ) -> None:
        """
        Parameters
        ----------
        profiles:
            链路 profile 列表。长度至少为 1。
            - fixed 模式：只用 profiles[0]
            - per_client 模式：len(profiles) 必须 == num_clients
            - heterogeneous 模式：按 profile_probs 在 profiles 中采样
        num_clients:
            客户端数量。
        mode:
            "fixed", "per_client", "heterogeneous" 之一。
        profile_probs:
            heterogeneous 模式下为各 profile 的采样概率；若为 None，则均匀。
        temporal_corr:
            时域相关系数 rho ∈ [0,1)。0 表示各轮独立，接近 1 表示缓慢变化。
        jitter_std:
            capacity 归一化扰动项的标准差。例如 0.2 表示 20% 左右的波动。
        seed:
            随机种子，便于复现。
        net_csv:
            网络状况日志 CSV 文件路径。
        """
        assert len(profiles) >= 1, "profiles must not be empty."
        assert 0.0 <= temporal_corr < 1.0, "temporal_corr must be in [0,1)."
        assert jitter_std >= 0.0, "jitter_std must be non-negative."

        self.profiles = profiles
        self.num_clients = num_clients
        self.mode = mode
        self.temporal_corr = temporal_corr
        self.jitter_std = jitter_std
        self.net_csv = net_csv

        if seed is not None:
            random.seed(seed)
        
        if self.net_csv is not None:
            self.net_csv_fieldnames = [
                "client_idx", 
                "global_round", 
                "bw_up_mbps", 
                "bw_down_mbps", 
                "rtt_ms", 
                "comm_time",
            ]
            with open(self.net_csv, "w") as f:
                writer = csv.DictWriter(f, fieldnames=self.net_csv_fieldnames)
                writer.writeheader()

        # 为每个 client 分配一个 base profile
        if mode == "fixed":
            base_profile_indices = [0 for _ in range(num_clients)]
        elif mode == "per_client":
            assert len(profiles) == num_clients, \
                "per_client mode requires len(profiles) == num_clients."
            base_profile_indices = list(range(num_clients))
        elif mode == "heterogeneous":
            if profile_probs is None:
                profile_probs = [1.0 / len(profiles)] * len(profiles)
            assert len(profile_probs) == len(profiles)
            # 归一化防止用户写得不严格
            s = sum(profile_probs)
            profile_probs = [p / s for p in profile_probs]
            base_profile_indices = [
                self._choice_with_probs(profile_probs) for _ in range(num_clients)
            ]
        else:
            raise ValueError(f"Unknown network mode: {mode}")

        self.base_profile_indices = base_profile_indices

        # 为每个 client 维护一个 (noise_up, noise_down, noise_rtt) 状态（AR(1)）
        self.noise_state = [
            [0.0, 0.0, 0.0] for _ in range(num_clients)
        ]

    # ------------------------------------------------------------------
    # 公共接口
    # ------------------------------------------------------------------

    def get_base_profile(self, client_idx: int) -> LinkProfile:
        """返回某个 client 的基准 profile（不含时间扰动）。"""
        idx = self.base_profile_indices[client_idx]
        return self.profiles[idx]

    def sample_link(
        self,
        client_idx: int,
    ) -> Tuple[float, float, float]:
        """
        为给定 client 和轮次返回 (bw_up_mbps, bw_down_mbps, rtt_ms)。

        如果 temporal_corr == 0 且 jitter_std == 0，则永远返回 base profile。
        """
        base = self.get_base_profile(client_idx)

        if self.jitter_std <= 1e-12:
            # 无时间抖动，直接返回 base
            return base.bw_up_mbps, base.bw_down_mbps, base.rtt_ms

        rho = self.temporal_corr
        bw_up_noise, bw_down_noise, rtt_noise = self.noise_state[client_idx]

        # 一步 AR(1) 更新: x_t = rho * x_{t-1} + sqrt(1-rho^2) * eps_t
        # eps_t ~ N(0, jitter_std^2) 用 Box-Muller 简单近似
        def ar1_step(prev_noise: float) -> float:
            u1, u2 = random.random(), random.random()
            eps = math.sqrt(-2.0 * math.log(u1 + 1e-12)) * math.cos(2 * math.pi * u2)
            eps *= self.jitter_std
            return rho * prev_noise + math.sqrt(max(1.0 - rho * rho, 0.0)) * eps

        bw_up_noise = ar1_step(bw_up_noise)
        bw_down_noise = ar1_step(bw_down_noise)
        rtt_noise = ar1_step(rtt_noise)

        self.noise_state[client_idx] = [bw_up_noise, bw_down_noise, rtt_noise]

        # 把 noise 看成相对涨跌比例：cap_t = cap_mean * (1 + noise)
        # 为防止带宽为负，加一个下限
        bw_up = max(base.bw_up_mbps * (1.0 + bw_up_noise), 1e-3)
        bw_down = max(base.bw_down_mbps * (1.0 + bw_down_noise), 1e-3)

        # RTT 用 additive 模型：rtt_t = rtt_mean * (1 + noise)
        rtt_ms = max(base.rtt_ms * (1.0 + rtt_noise), 0.0)

        return bw_up, bw_down, rtt_ms

    def estimate_comm_time(
        self,
        client_idx: int,
        global_round: int,
        bytes_up: int,
        bytes_down: int,
    ) -> float:
        """
        根据本轮带宽和 RTT 估计通信时延（秒）。

        模型：一轮通信包含一次 RTT（往返） + 上/下行传输时间
          t_up   = RTT/2 + bits_up   / (bw_up * 1e6)
          t_down = RTT/2 + bits_down / (bw_down * 1e6)
          t_comm = t_up + t_down

        若你只模拟上行（split learning 常见），可以把 bytes_down 设成 0。
        """
        bw_up, bw_down, rtt_ms = self.sample_link(client_idx)

        bits_up = float(bytes_up) * 8.0
        bits_down = float(bytes_down) * 8.0

        rtt_s = rtt_ms / 1000.0
        t_up = rtt_s / 2.0 + (bits_up / max(bw_up * 1e6, 1e-6))
        t_down = rtt_s / 2.0 + (bits_down / max(bw_down * 1e6, 1e-6))

        if self.net_csv is not None:
            with open(self.net_csv, "a") as f:
                writer = csv.DictWriter(f, fieldnames=self.net_csv_fieldnames)
                writer.writerow({
                    "client_idx": client_idx,
                    "global_round": global_round,
                    "bw_up_mbps": bw_up,
                    "bw_down_mbps": bw_down,
                    "rtt_ms": rtt_ms,
                    "comm_time": t_up + t_down
                })

        return t_up + t_down

    # ------------------------------------------------------------------
    # 内部辅助
    # ------------------------------------------------------------------

    @staticmethod
    def _choice_with_probs(probs: List[float]) -> int:
        """
        按给定概率在 [0..len(probs)-1] 里采样一个索引。
        """
        r = random.random()
        cumulative = 0.0
        for i, p in enumerate(probs):
            cumulative += p
            if r <= cumulative:
                return i
        return len(probs) - 1


# ----------------------------------------------------------------------
# 工厂函数：从 cfg 构建 NetworkSimulator
# ----------------------------------------------------------------------

def build_network_simulator(cfg: Any, num_clients: int) -> NetworkSimulator:
    """
    从 cfg.simulation.comm 构建一个 NetworkSimulator。

    若 cfg 没有 simulation.comm 字段，则回退到“全局固定带宽+延迟”的简单模型。
    """
    # 没有 network 配置：退化为固定模型，参数从你旧 cfg 中拿或给默认
    if not hasattr(cfg, "simulation") or not hasattr(cfg.simulation, "comm"):
        # 兼容以前：比如 cfg.profiling.bandwidth_mbps / cfg.profiling.latency_ms
        profiling = getattr(cfg, "profiling", None)
        if profiling is not None:
            bw_up = float(getattr(profiling, "bandwidth_mbps", 80.0))
            bw_down = bw_up
            rtt_ms = float(getattr(profiling, "latency_ms", 10.0))
        else:
            bw_up = bw_down = 80.0
            rtt_ms = 10.0

        profiles = [LinkProfile(name="default", bw_up_mbps=bw_up,
                                bw_down_mbps=bw_down, rtt_ms=rtt_ms)]
        return NetworkSimulator(
            profiles=profiles,
            num_clients=num_clients,
            mode="fixed",
            temporal_corr=0.0,
            jitter_std=0.0,
        )

    net_cfg = cfg.simulation.comm

    mode = getattr(net_cfg, "mode", "fixed")

    seed = getattr(cfg, "seed", None)
    net_csv = getattr(cfg.logging, "net_csv", None) if hasattr(cfg, "logging") else None

    if mode == "fixed":
        bw_up = float(getattr(net_cfg, "bw_up_mbps", 80.0))
        bw_down = float(getattr(net_cfg, "bw_down_mbps", bw_up))
        rtt_ms = float(getattr(net_cfg, "rtt_ms", 10.0))

        profiles = [LinkProfile(name="fixed", bw_up_mbps=bw_up,
                                bw_down_mbps=bw_down, rtt_ms=rtt_ms)]
        temporal_corr = float(getattr(net_cfg, "temporal_corr", 0.0))
        jitter_std = float(getattr(net_cfg, "jitter_std", 0.0))

        return NetworkSimulator(
            profiles=profiles,
            num_clients=num_clients,
            mode="fixed",
            temporal_corr=temporal_corr,
            jitter_std=jitter_std,
            seed=seed,
            net_csv=net_csv,
        )

    elif mode == "per_client":
        # 期望 cfg.network.per_client 是一个列表，每个元素有 bw_up_mbps, bw_down_mbps, rtt_ms
        per_client_cfg = getattr(net_cfg, "per_client", None)
        if per_client_cfg is None:
            raise ValueError("network.mode='per_client' but network.per_client is missing.")

        if len(per_client_cfg) != num_clients:
            raise ValueError(
                f"network.per_client length ({len(per_client_cfg)}) "
                f"must equal num_clients ({num_clients})."
            )

        profiles: List[LinkProfile] = []
        for i, pc in enumerate(per_client_cfg):
            name = getattr(pc, "name", f"client_{i}")
            bw_up = float(getattr(pc, "bw_up_mbps", 80.0))
            bw_down = float(getattr(pc, "bw_down_mbps", bw_up))
            rtt_ms = float(getattr(pc, "rtt_ms", 10.0))
            profiles.append(
                LinkProfile(name=name, bw_up_mbps=bw_up, bw_down_mbps=bw_down, rtt_ms=rtt_ms)
            )

        temporal_corr = float(getattr(net_cfg, "temporal_corr", 0.0))
        jitter_std = float(getattr(net_cfg, "jitter_std", 0.0))

        return NetworkSimulator(
            profiles=profiles,
            num_clients=num_clients,
            mode="per_client",
            temporal_corr=temporal_corr,
            jitter_std=jitter_std,
            seed=seed,
        )

    elif mode == "heterogeneous":
        # 期望 cfg.network.profiles 是一个列表
        profiles_cfg = getattr(net_cfg, "profiles", None)
        if profiles_cfg is None or len(profiles_cfg) == 0:
            raise ValueError("network.mode='heterogeneous' but network.profiles is empty.")

        profiles: List[LinkProfile] = []
        for i, pc in enumerate(profiles_cfg):
            name = getattr(pc, "name", f"profile_{i}")
            bw_up = float(getattr(pc, "bw_up_mbps", 80.0))
            bw_down = float(getattr(pc, "bw_down_mbps", bw_up))
            rtt_ms = float(getattr(pc, "rtt_ms", 10.0))
            profiles.append(
                LinkProfile(name=name, bw_up_mbps=bw_up, bw_down_mbps=bw_down, rtt_ms=rtt_ms)
            )

        profile_probs = getattr(net_cfg, "profile_probs", None)
        if profile_probs is not None:
            profile_probs = list(profile_probs)

        temporal_corr = float(getattr(net_cfg, "temporal_corr", 0.0))
        jitter_std = float(getattr(net_cfg, "jitter_std", 0.0))

        return NetworkSimulator(
            profiles=profiles,
            num_clients=num_clients,
            mode="heterogeneous",
            profile_probs=profile_probs,
            temporal_corr=temporal_corr,
            jitter_std=jitter_std,
            seed=seed,
        )

    else:
        raise ValueError(f"Unknown network.mode: {mode}")
