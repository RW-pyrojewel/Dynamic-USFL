# src/privacy/metrics.py
from __future__ import annotations

import math
from .config import PrivacyConfig


def lia_auc_to_privacy(auc: float, cfg: PrivacyConfig) -> float:
    """
    Default mapping: P_label = AUC (clipped to [0,1]).
    """
    auc = float(auc)
    return max(0.0, min(1.0, auc))


def mia_quality_to_privacy(quality: float, cfg: PrivacyConfig) -> float:
    """
    Default mapping for MIA:
      quality = MSE (smaller => better recon => worse privacy)
      P_sample = exp(-MSE / scale) clipped to [0,1]
    """
    quality = float(quality)
    scale = max(1e-8, float(cfg.metrics.mia_scale))
    p = math.exp(-quality / scale)
    return max(0.0, min(1.0, p))


def combine_privacy_scores(cfg: PrivacyConfig, P_label: float, P_sample: float) -> float:
    w_l = cfg.sae.weight_lia if cfg.sae.enable_lia else 0.0
    w_s = cfg.sae.weight_mia if cfg.sae.enable_mia else 0.0
    denom = max(1e-8, w_l + w_s)
    return (w_l * float(P_label) + w_s * float(P_sample)) / denom
