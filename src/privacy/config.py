# src/privacy/config.py
from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any


@dataclass
class AuxDatasetConfig:
    name: str
    root: str
    split: str
    batch_size: int
    num_workers: int


@dataclass
class MixMatchConfig:
    """
    MixMatch hyper-parameters (Fu Sec. 3.3; MixMatch paper referenced therein).
    """
    labeled_fraction: float  # fraction of aux data treated as labeled
    T: float                 # temperature for sharpening
    alpha: float             # Beta(alpha, alpha) for MixUp
    lambda_u: float          # weight for unsupervised loss
    use_augmentation: bool   # Fu also studies a "no augmentation" customized variant


@dataclass
class SAEConfig:
    # enable branches
    enable_lia: bool
    enable_mia: bool

    # how to aggregate into P_global
    weight_lia: float
    weight_mia: float

    # multi-task training loss weights
    lambda_lia: float
    lambda_rec: float

    # LIA training algorithm
    lia_algorithm: str             # "mix-match" for Fu Sec. 3.3
    mixmatch: MixMatchConfig

    # training control
    force_retrain: bool
    epochs: int
    lr: float
    weight_decay: float
    patience: int

    # model heads
    lia_hidden_dim: int
    lia_dropout: float

    # decoder (MIA) capacity
    decoder_channel_min: int
    decoder_out_act: str           # "sigmoid" or "identity"

    # checkpoint
    save_attacker_ckpt: bool


@dataclass
class MetricsConfig:
    """
    Mapping from attack metrics to privacy scores.
    """
    lia_to_priv: str   # "auc_identity": P_label = AUC
    mia_metric: str    # "mse" (default)
    mia_scale: float   # scale used in mapping quality -> privacy


@dataclass
class PrivacyConfig:
    enabled: bool
    aux: AuxDatasetConfig
    sae: SAEConfig
    metrics: MetricsConfig


def _get(ns: Any, name: str, default):
    return getattr(ns, name, default) if ns is not None else default


def get_privacy_cfg(cfg) -> PrivacyConfig:
    """
    Parse cfg.privacy.* into typed config objects.

    Compatibility notes:
      - Accept both privacy.enable and privacy.enabled
    """
    p = getattr(cfg, "privacy", SimpleNamespace())

    # accept both "enable" and "enabled"
    enabled = _get(p, "enable", None)
    if enabled is None:
        enabled = _get(p, "enabled", True)

    aux_ns = getattr(p, "aux_dataset", SimpleNamespace())
    sae_ns = getattr(p, "sae", SimpleNamespace())
    met_ns = getattr(p, "metrics", SimpleNamespace())
    mixmatch_ns = getattr(sae_ns, "mixmatch", SimpleNamespace())

    aux = AuxDatasetConfig(
        name=str(_get(aux_ns, "name", "derm7pt")),
        root=str(_get(aux_ns, "root", "./data/derm7pt")),
        split=str(_get(aux_ns, "split", "train")),
        batch_size=int(_get(aux_ns, "batch_size", 64)),
        num_workers=int(_get(aux_ns, "num_workers", 4)),
    )

    mixmatch = MixMatchConfig(
        labeled_fraction=float(_get(mixmatch_ns, "labeled_fraction", 0.1)),
        T=float(_get(mixmatch_ns, "T", 0.5)),
        alpha=float(_get(mixmatch_ns, "alpha", 0.75)),
        lambda_u=float(_get(mixmatch_ns, "lambda_u", 75.0)),
        use_augmentation=bool(_get(mixmatch_ns, "use_augmentation", False)),
    )

    sae = SAEConfig(
        enable_lia=bool(_get(sae_ns, "enable_lia", True)),
        enable_mia=bool(_get(sae_ns, "enable_mia", True)),
        weight_lia=float(_get(sae_ns, "weight_lia", 0.5)),
        weight_mia=float(_get(sae_ns, "weight_mia", 0.5)),
        lambda_lia=float(_get(sae_ns, "lambda_lia", 1.0)),
        lambda_rec=float(_get(sae_ns, "lambda_rec", 1.0)),
        lia_algorithm=str(_get(sae_ns, "lia_algorithm", "mix-match")),
        mixmatch=mixmatch,
        force_retrain=bool(_get(sae_ns, "force_retrain", False)),
        epochs=int(_get(sae_ns, "epochs", 20)),
        lr=float(_get(sae_ns, "lr", 1e-3)),
        weight_decay=float(_get(sae_ns, "weight_decay", 1e-4)),
        patience=int(_get(sae_ns, "patience", 5)),
        lia_hidden_dim=int(_get(sae_ns, "lia_hidden_dim", 256)),
        lia_dropout=float(_get(sae_ns, "lia_dropout", 0.0)),
        decoder_channel_min=int(_get(sae_ns, "decoder_channel_min", 32)),
        decoder_out_act=str(_get(sae_ns, "decoder_out_act", "sigmoid")),
        save_attacker_ckpt=str(_get(sae_ns, "save_attacker_ckpt", True)),
    )

    metrics = MetricsConfig(
        lia_to_priv=str(_get(met_ns, "lia_to_priv", "auc_identity")),
        mia_metric=str(_get(met_ns, "mia_metric", "mse")),
        mia_scale=float(_get(met_ns, "mia_scale", 1.0)),
    )

    return PrivacyConfig(
        enabled=bool(enabled),
        aux=aux,
        sae=sae,
        metrics=metrics,
    )
