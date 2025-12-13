# src/bandits/linucb_dualcut.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

# 一个切分动作 d = (s1, s2)，s1 < s2
CutPair = Tuple[int, int]


@dataclass
class BanditDecision:
    """
    LinUCB 每一轮的决策结果（便于上层记录 / 日志）。

    Attributes
    ----------
    action : CutPair
        被选中的切分点二元组 (s1, s2)。
    j_hat : float
        该动作在当前轮次的预测目标值 \hat{J}_t(d) = S(d) + \hat{D}_t(d) - alpha * sigma(d)。
    d_hat : float
        预测的动态代价 \hat{D}_t(d) = theta^T x_d（仅通信 + 边缘计算部分）。
    sigma : float
        不确定性项 sigma(d) = sqrt( x_d^T A^{-1} x_d )。
    """

    action: CutPair
    j_hat: float
    d_hat: float
    sigma: float


class LinUCBDualCut:
    r"""
    L2S 风格的 LinUCB-E 线性 bandit，用于 USFL 中的动态双切分点选择。

    目标：在若干候选切分对 d = (s1, s2) 之间，在线最小化

        J_t(d) = S(d) + D_t(d),

    其中
        - S(d): 离线预先计算好的静态代价（精度 + 端侧计算 + 隐私）。
        - D_t(d): 通信 + 边缘侧计算构成的动态代价，在未知环境下随时间变化，
                  用线性模型 D_t(d) ≈ theta^T x_d 进行估计。

    本类内部用的是“最小化版”的 UCB：
        选取使得
            \hat{J}_t(d) = S(d) + \hat{D}_t(d) - alpha * sigma_t(d)
        最小的 d。

    其中
        - \hat{D}_t(d) = theta_hat^T x_d,
        - sigma_t(d)   = sqrt( x_d^T A^{-1} x_d ),
        - theta_hat    = A^{-1} b.

    Notes
    -----
    1. 这里的“上下文” x_d 只依赖于切分 d（与 L2S 相同），
       环境变化被吸收到参数 theta 中。
       x_d 通常由“边缘端 MAC 数 + IR 大小 + 参数量”等特征组成。

    2. 纯本地训练（PDT）场景下，如果需要保持与 L2S 一致的行为，
       可以在上层传入 x_d = 0 向量，并在强制采样轮次通过 forbidden_actions
       将该动作从候选集中屏蔽掉（LinUCB-E 逻辑由上层控制，这里不做 schedule）。

    3. 该实现以“代价 cost”为一阶对象，没有把 cost 转成 reward，
       避免符号混乱；UCB 公式中直接使用 cost 的预测值。
    """

    def __init__(
        self,
        actions: Sequence[CutPair],
        features: Mapping[CutPair, np.ndarray],
        static_costs: Mapping[CutPair, float],
        alpha: float = 1.0,
        lambda_reg: float = 1.0,
    ) -> None:
        """
        Parameters
        ----------
        actions : Sequence[CutPair]
            所有候选切分点二元组的列表，例如 [(0, 3), (1, 4), ...]。
        features : Mapping[CutPair, np.ndarray]
            每个切分动作 d 对应的特征向量 x_d。
            要求:
              - features[d].shape = (d_feat,)
              - 对于 actions 中的每个 d 必须存在对应条目。
        static_costs : Mapping[CutPair, float]
            每个动作的静态代价 S(d)，例如:
              S(d) = λ_acc * C_acc(d)
                   + λ_comp * C_comp-dev(d)
                   + λ_priv * C_priv(d)
            要求:
              - static_costs[d] 为标量 float
              - 对于 actions 中的每个 d 必须存在对应条目。
        alpha : float, default 1.0
            UCB 中的不确定性系数，越大越偏向探索。
        lambda_reg : float, default 1.0
            A 矩阵的 L2 正则项：初始化为 A_0 = lambda_reg * I。
        """
        if not actions:
            raise ValueError("LinUCBDualCut: 'actions' must be a non-empty sequence.")

        self.actions: List[CutPair] = list(actions)
        self._features: Dict[CutPair, np.ndarray] = {}
        self._static_costs: Dict[CutPair, float] = {}

        # 检查并规范化特征维度
        first_action = self.actions[0]
        if first_action not in features:
            raise KeyError(f"Features for action {first_action} are missing.")

        first_feature = np.asarray(features[first_action], dtype=np.float64).ravel()
        self.d_feat: int = int(first_feature.shape[0])

        for a in self.actions:
            if a not in features:
                raise KeyError(f"Features for action {a} are missing.")
            x = np.asarray(features[a], dtype=np.float64).ravel()
            if x.shape[0] != self.d_feat:
                raise ValueError(
                    f"Feature dimension mismatch for action {a}: "
                    f"expected {self.d_feat}, got {x.shape[0]}"
                )
            self._features[a] = x

            if a not in static_costs:
                raise KeyError(f"Static cost for action {a} is missing.")
            self._static_costs[a] = float(static_costs[a])

        self.alpha: float = float(alpha)
        self.lambda_reg: float = float(lambda_reg)

        # LinUCB 状态：A, A_inv, b, theta
        self._A: np.ndarray = self.lambda_reg * np.eye(self.d_feat, dtype=np.float64)
        self._A_inv: np.ndarray = np.linalg.inv(self._A)
        self._b: np.ndarray = np.zeros(self.d_feat, dtype=np.float64)
        self._theta: np.ndarray = np.zeros(self.d_feat, dtype=np.float64)

        # 记录更新轮次
        self.num_updates: int = 0

    # ------------------------------------------------------------------
    # 公共接口
    # ------------------------------------------------------------------
    @property
    def theta(self) -> np.ndarray:
        """当前的参数估计向量 theta_hat。"""
        return self._theta

    @property
    def A(self) -> np.ndarray:
        """当前的协方差矩阵 A。"""
        return self._A

    @property
    def A_inv(self) -> np.ndarray:
        """当前的 A^{-1}。"""
        return self._A_inv

    @property
    def b(self) -> np.ndarray:
        """当前的一阶统计量向量 b。"""
        return self._b

    def reset(self) -> None:
        """重置 LinUCB 内部状态（A, b, theta），但不改变 actions / features。"""
        self._A = self.lambda_reg * np.eye(self.d_feat, dtype=np.float64)
        self._A_inv = np.linalg.inv(self._A)
        self._b = np.zeros(self.d_feat, dtype=np.float64)
        self._theta = np.zeros(self.d_feat, dtype=np.float64)
        self.num_updates = 0

    # ------------------------------------------------------------------
    # 核心算子：预测 / 选臂 / 更新
    # ------------------------------------------------------------------
    def feature(self, action: CutPair) -> np.ndarray:
        """返回某个动作 d 对应的特征向量 x_d（只读视图）。"""
        return self._features[action]

    def static_cost(self, action: CutPair) -> float:
        """返回某个动作 d 的静态代价 S(d)。"""
        return self._static_costs[action]

    def predict_dynamic_cost(self, action: CutPair) -> Tuple[float, float]:
        """
        预测某个动作 d 的动态代价 D_hat(d) 及其标准差 sigma(d)。

        Returns
        -------
        d_hat : float
            预测的动态代价 \hat{D}_t(d) = theta^T x_d。
        sigma : float
            不确定性项 sigma(d) = sqrt( x_d^T A^{-1} x_d )。
        """
        x = self._features[action]
        d_hat = float(self._theta @ x)
        sigma_sq = float(x @ self._A_inv @ x)
        sigma = float(np.sqrt(max(sigma_sq, 0.0)))
        return d_hat, sigma

    def select_action(
        self,
        forbidden_actions: Optional[Iterable[CutPair]] = None,
    ) -> BanditDecision:
        """
        根据当前 theta / A^{-1} 以及静态代价 S(d)，选取一轮的切分动作。

        对候选动作集 A = {d} \ forbidden_actions，计算

            \hat{J}_t(d) = S(d) + \hat{D}_t(d) - alpha * sigma(d),

        选择使 \hat{J}_t(d) 最小的动作。

        Parameters
        ----------
        forbidden_actions : Iterable[CutPair], optional
            在本轮中禁止选择的动作集合。
            用于实现 LinUCB-E 中“强制采样”时屏蔽 PDT 等动作。

        Returns
        -------
        decision : BanditDecision
            本轮决策结果，包含所选动作及其 \hat{J}_t, \hat{D}_t, sigma。
        """
        if forbidden_actions is not None:
            forbidden = set(forbidden_actions)
        else:
            forbidden = set()

        best_action: Optional[CutPair] = None
        best_j_hat: float = float("inf")
        best_d_hat: float = 0.0
        best_sigma: float = 0.0

        for a in self.actions:
            if a in forbidden:
                continue

            d_hat, sigma = self.predict_dynamic_cost(a)
            j_hat = self._static_costs[a] + d_hat - self.alpha * sigma

            if j_hat < best_j_hat:
                best_j_hat = j_hat
                best_action = a
                best_d_hat = d_hat
                best_sigma = sigma

        if best_action is None:
            raise RuntimeError(
                "LinUCBDualCut.select_action: no available actions after applying 'forbidden_actions'."
            )

        return BanditDecision(
            action=best_action,
            j_hat=best_j_hat,
            d_hat=best_d_hat,
            sigma=best_sigma,
        )

    def update(self, action: CutPair, dynamic_cost_observed: float) -> None:
        """
        用一条观测 (action, D_t(action)) 更新线性模型参数。

        Parameters
        ----------
        action : CutPair
            本轮实际执行的动作 d。
        dynamic_cost_observed : float
            观测到的动态代价 D_t(d)，即
                D_t(d) = J_t^{obs}(d) - S(d),
            或者你在上层调用时直接传入
                λ_comm * C_comm,t(d) + λ_comp * C_comp-edge,t(d)。
        """
        x = self._features[action]
        y = float(dynamic_cost_observed)

        # A ← A + x x^T, b ← b + x y
        self._A = self._A + np.outer(x, x)
        self._b = self._b + x * y

        # 更新 A^{-1} 和 theta
        # 这里直接用逆矩阵，d_feat 很小（≈8），代价可以接受。
        self._A_inv = np.linalg.inv(self._A)
        self._theta = self._A_inv @ self._b
        self.num_updates += 1


# ----------------------------------------------------------------------
# 构造函数：从 cfg 读取 bandit 的超参数
# ----------------------------------------------------------------------


def build_linucb_dualcut(
    cfg,
    actions: Sequence[CutPair],
    features: Mapping[CutPair, np.ndarray],
    static_costs: Mapping[CutPair, float],
) -> LinUCBDualCut:
    """
    使用配置文件构造 LinUCBDualCut 实例的辅助函数。

    期望 cfg.bandit 下有字段:
        - alpha: float, default 1.0
        - lambda_reg: float, default 1.0

    Parameters
    ----------
    cfg : SimpleNamespace or similar
        全局配置对象。
    actions : Sequence[CutPair]
        候选切分动作列表。
    features : Mapping[CutPair, np.ndarray]
        每个动作的特征向量 x_d。
    static_costs : Mapping[CutPair, float]
        每个动作的静态代价 S(d)。

    Returns
    -------
    bandit : LinUCBDualCut
        已初始化的 bandit 实例。
    """
    bandit_cfg = getattr(cfg, "bandit", None)
    if bandit_cfg is not None:
        alpha = float(getattr(bandit_cfg, "alpha", 1.0))
        lambda_reg = float(getattr(bandit_cfg, "lambda_reg", 1.0))
    else:
        alpha = 1.0
        lambda_reg = 1.0

    return LinUCBDualCut(
        actions=actions,
        features=features,
        static_costs=static_costs,
        alpha=alpha,
        lambda_reg=lambda_reg,
    )
