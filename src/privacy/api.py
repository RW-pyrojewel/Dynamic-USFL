# src/privacy/api.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from src.models.model_usfl import USFLBackbone

from .config import get_privacy_cfg, PrivacyConfig
from .data import build_aux_loader, load_victim_batch
from .metrics import combine_privacy_scores
from .sae import (
    train_or_load_sae_attacker,
    evaluate_lia_attack,
    evaluate_mia_attack,
)


@dataclass
class PrivacyResult:
    # privacy scores used by J
    P_label: float
    P_sample: float
    P_global: float

    # raw metrics (for analysis / logging)
    lia_auc: Optional[float] = None
    lia_acc: Optional[float] = None
    mia_mse: Optional[float] = None


def evaluate_privacy_for_cut(
    cfg,
    cut_key: str,
    cut_dir: str,
    cut1: int,
    backbone_template: USFLBackbone,
    device: Optional[str] = None,
) -> PrivacyResult:
    """
    End-to-end privacy evaluation for a given cut pair.

    Assumptions:
      - cut_dir/privacy_samples.pt exists and contains: A_front, y, and optionally x
      - attacker is trained with labeled aux raw samples (x_aux, y_aux)
      - victim A_front is used as unlabeled pool during attacker training (latent-space MixMatch)
      - victim A_front is also used at evaluation time (as latent Z)
    """
    priv_cfg: PrivacyConfig = get_privacy_cfg(cfg)
    if not priv_cfg.enabled:
        return PrivacyResult(P_label=0.0, P_sample=0.0, P_global=0.0)

    device = device or getattr(cfg.training, "device", "cuda")

    victim_batch = load_victim_batch(cut_dir=cut_dir, device=device)
    if victim_batch is None or victim_batch.num_samples == 0:
        return PrivacyResult(P_label=0.0, P_sample=0.0, P_global=0.0)

    aux_loader = build_aux_loader(cfg, priv_cfg)

    attacker = train_or_load_sae_attacker(
        cfg=cfg,
        priv_cfg=priv_cfg,
        cut_key=cut_key,
        cut1=cut1,
        backbone_template=backbone_template,
        victim_batch=victim_batch,
        aux_loader=aux_loader,
        device=device,
    )

    P_label = 0.0
    lia_auc = None
    lia_acc = None
    if priv_cfg.sae.enable_lia:
        lia = evaluate_lia_attack(cfg, priv_cfg, attacker, victim_batch, device)
        P_label = float(lia.P_label)
        lia_auc = float(lia.auc)
        lia_acc = float(lia.acc)

    P_sample = 0.0
    mia_mse = None
    if priv_cfg.sae.enable_mia:
        mia = evaluate_mia_attack(cfg, priv_cfg, attacker, victim_batch, device)
        P_sample = float(mia.P_sample)
        mia_mse = float(mia.mse) if mia.mse == mia.mse else None  # nan check

    P_global = combine_privacy_scores(priv_cfg, P_label=P_label, P_sample=P_sample)

    return PrivacyResult(
        P_label=P_label,
        P_sample=P_sample,
        P_global=P_global,
        lia_auc=lia_auc,
        lia_acc=lia_acc,
        mia_mse=mia_mse,
    )
