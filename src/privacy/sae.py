# src/privacy/sae.py
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import trange

from src.models.model_usfl import USFLBackbone

from .config import PrivacyConfig
from .data import VictimBatch
from .metrics import lia_auc_to_privacy, mia_quality_to_privacy
from .models import (
    LatentSpec,
    ShadowEncoder,
    LabelHead,
    MirrorDecoder,
    SAESLAttacker,
    build_front_template_from_backbone,
)


@dataclass
class LIAEval:
    auc: float
    acc: float
    P_label: float


@dataclass
class MIAEval:
    mse: float
    P_sample: float


def _infer_img_shape_from_cfg(cfg) -> Tuple[int, int, int]:
    data_cfg = cfg.data
    c = int(getattr(data_cfg, "in_channels", 3))
    h = int(getattr(data_cfg, "img_size", 224))
    w = int(getattr(data_cfg, "img_size", 224))
    return c, h, w


def _split_aux_dataset(aux_loader: DataLoader, labeled_fraction: float, seed: int = 0) -> Tuple[DataLoader, DataLoader]:
    """
    Create two loaders (labeled, unlabeled) from the given aux_loader.dataset.
    We only need (x, y) from labeled; unlabeled uses x only.
    """
    ds = aux_loader.dataset
    n = len(ds)
    rng = np.random.RandomState(seed)
    idx = np.arange(n)
    rng.shuffle(idx)

    n_l = max(1, int(round(n * labeled_fraction)))
    labeled_idx = idx[:n_l].tolist()
    unlabeled_idx = idx[n_l:].tolist()

    # keep batch size same
    common_kwargs = dict(
        batch_size=aux_loader.batch_size,
        num_workers=aux_loader.num_workers,
        pin_memory=getattr(aux_loader, "pin_memory", False),
        drop_last=True,
    )

    labeled_loader = DataLoader(Subset(ds, labeled_idx), shuffle=True, **common_kwargs)
    unlabeled_loader = DataLoader(Subset(ds, unlabeled_idx), shuffle=True, **common_kwargs)
    return labeled_loader, unlabeled_loader


def _sharpen(p: torch.Tensor, T: float) -> torch.Tensor:
    """
    Sharpening as in MixMatch:
      p_sharp = p^{1/T} / sum(p^{1/T})
    """
    if T <= 0:
        return p
    p_pow = p ** (1.0 / T)
    return p_pow / p_pow.sum(dim=1, keepdim=True).clamp_min(1e-12)


def _onehot(y: torch.Tensor, num_classes: int) -> torch.Tensor:
    return F.one_hot(y, num_classes=num_classes).float()


def _mixup(x1: torch.Tensor, y1: torch.Tensor, x2: torch.Tensor, y2: torch.Tensor, alpha: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    MixUp with Beta(alpha, alpha).
    """
    if alpha <= 0:
        return x1, y1
    lam = np.random.beta(alpha, alpha)
    lam = max(lam, 1.0 - lam)
    lam_t = torch.tensor(lam, dtype=x1.dtype, device=x1.device)
    x = lam_t * x1 + (1.0 - lam_t) * x2
    y = lam_t * y1 + (1.0 - lam_t) * y2
    return x, y


def build_attacker_for_cut(
    cfg,
    priv_cfg: PrivacyConfig,
    backbone_template: USFLBackbone,
    cut1: int,
    victim_batch: VictimBatch,
    device: str,
) -> SAESLAttacker:
    """
    Build SAE-SL attacker aligned with victim f1 architecture (structure only).
    """
    img_shape = _infer_img_shape_from_cfg(cfg)
    z_spec = LatentSpec.from_tensor(victim_batch.A_front)

    # Shadow encoder template: f1 structure (layers[0..cut1])
    front_template = build_front_template_from_backbone(backbone_template, cut1)
    encoder = ShadowEncoder(front_template, reinit=True).to(device)

    label_head = LabelHead(
        z_spec=z_spec,
        num_classes=int(cfg.data.num_classes),
        hidden_dim=priv_cfg.sae.lia_hidden_dim,
        dropout=priv_cfg.sae.lia_dropout,
    ).to(device)

    decoder = MirrorDecoder(
        z_spec=z_spec,
        img_shape=img_shape,
        channel_min=priv_cfg.sae.decoder_channel_min,
        out_act=priv_cfg.sae.decoder_out_act,
    ).to(device)

    return SAESLAttacker(encoder=encoder, label_head=label_head, decoder=decoder).to(device)


def train_or_load_sae_attacker(
    cfg,
    priv_cfg: PrivacyConfig,
    cut_key: str,
    cut1: int,
    backbone_template: USFLBackbone,
    victim_batch: VictimBatch,
    aux_loader: DataLoader,
    device: str,
) -> SAESLAttacker:
    """
    Train SAE-SL attacker on auxiliary raw samples (x_aux, y_aux), using MixMatch for LIA
    (Fu Sec. 3.3) and reconstruction loss for MIA (AutoEncoderNN style).

    Checkpoint is per cut_key.
    """
    os.makedirs(priv_cfg.sae.attacker_ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(priv_cfg.sae.attacker_ckpt_dir, f"sae_{cut_key}.pt")

    attacker = build_attacker_for_cut(cfg, priv_cfg, backbone_template, cut1, victim_batch, device)

    if os.path.exists(ckpt_path) and not priv_cfg.sae.force_retrain:
        state = torch.load(ckpt_path, map_location=device)
        attacker.load_state_dict(state["model"])
        return attacker

    # Split aux into labeled/unlabeled pools for MixMatch
    labeled_loader, unlabeled_loader = _split_aux_dataset(
        aux_loader,
        labeled_fraction=priv_cfg.sae.mixmatch.labeled_fraction,
        seed=int(getattr(cfg, "seed", 0)),
    )
    unlabeled_iter = iter(unlabeled_loader)

    opt = torch.optim.Adam(
        attacker.parameters(),
        lr=priv_cfg.sae.lr,
        weight_decay=priv_cfg.sae.weight_decay,
    )

    best = float("inf")
    patience = priv_cfg.sae.patience
    best_state = None

    num_classes = int(cfg.data.num_classes)
    T = priv_cfg.sae.mixmatch.T
    alpha = priv_cfg.sae.mixmatch.alpha
    lambda_u = priv_cfg.sae.mixmatch.lambda_u

    for epoch in trange(priv_cfg.sae.epochs, desc=f"[privacy] Train attacker {cut_key}", leave=False):
        attacker.train()
        epoch_loss = 0.0
        n_seen = 0

        for x_l, y_l in labeled_loader:
            # fetch unlabeled batch
            try:
                x_u, _ = next(unlabeled_iter)
            except StopIteration:
                unlabeled_iter = iter(unlabeled_loader)
                x_u, _ = next(unlabeled_iter)

            x_l = x_l.to(device, non_blocking=True)
            y_l = y_l.to(device, non_blocking=True)
            x_u = x_u.to(device, non_blocking=True)

            # ---- MixMatch pseudo-label for unlabeled ----
            with torch.no_grad():
                # customized MixMatch can be "no augmentation": use x_u directly
                # We still average two forward passes for stability (K=2).
                logits_u1, _, _ = attacker.forward_aux(x_u)
                logits_u2, _, _ = attacker.forward_aux(x_u)
                p_u = (F.softmax(logits_u1, dim=1) + F.softmax(logits_u2, dim=1)) / 2.0
                q_u = _sharpen(p_u, T)

            # labeled one-hot
            y_l_oh = _onehot(y_l, num_classes)

            # concat for mixup
            x_all = torch.cat([x_l, x_u], dim=0)
            y_all = torch.cat([y_l_oh, q_u], dim=0)

            # shuffle pair for mixup
            perm = torch.randperm(x_all.size(0), device=device)
            x2 = x_all[perm]
            y2 = y_all[perm]

            x_mix, y_mix = _mixup(x_all, y_all, x2, y2, alpha=alpha)

            # split back
            x_l_mix = x_mix[: x_l.size(0)]
            y_l_mix = y_mix[: x_l.size(0)]
            x_u_mix = x_mix[x_l.size(0) :]
            y_u_mix = y_mix[x_l.size(0) :]

            # ---- forward on mixed batches ----
            opt.zero_grad()

            # labeled branch
            logits_l, xhat_l, _ = attacker.forward_aux(x_l_mix)
            # unlabeled branch
            logits_u, xhat_u, _ = attacker.forward_aux(x_u_mix)

            # supervised loss Lx (cross-entropy with soft labels)
            logp_l = F.log_softmax(logits_l, dim=1)
            Lx = -(y_l_mix * logp_l).sum(dim=1).mean()

            # unsupervised loss Lu (MSE between probs and pseudo-labels)
            p_u_mix = F.softmax(logits_u, dim=1)
            Lu = F.mse_loss(p_u_mix, y_u_mix)

            loss_lia = Lx + lambda_u * Lu

            # reconstruction loss (both labeled/unlabeled mixed)
            loss_rec = 0.0
            if priv_cfg.sae.enable_mia:
                # use original images as targets (mixup already mixes images; target is mixed image)
                loss_rec = F.mse_loss(xhat_l, x_l_mix) + F.mse_loss(xhat_u, x_u_mix)

            loss = 0.0
            if priv_cfg.sae.enable_lia:
                loss = loss + priv_cfg.sae.lambda_lia * loss_lia
            if priv_cfg.sae.enable_mia:
                loss = loss + priv_cfg.sae.lambda_rec * loss_rec

            loss.backward()
            opt.step()

            bs = int(x_l.size(0))
            epoch_loss += float(loss.item()) * bs
            n_seen += bs

        epoch_loss = epoch_loss / max(1, n_seen)

        if epoch_loss < best:
            best = epoch_loss
            best_state = {"model": attacker.state_dict(), "epoch": epoch, "loss": best}
            patience = priv_cfg.sae.patience
        else:
            patience -= 1
            if patience <= 0:
                break

    if best_state is None:
        best_state = {"model": attacker.state_dict(), "epoch": priv_cfg.sae.epochs, "loss": float("nan")}
    torch.save(best_state, ckpt_path)
    attacker.load_state_dict(best_state["model"])
    return attacker


def _compute_multiclass_auc(y_true: np.ndarray, prob: np.ndarray, multi_class: str = "ovr") -> float:
    """
    Compute multiclass ROC-AUC.
    """
    try:
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(y_true, prob, multi_class=multi_class))
    except Exception:
        # fallback: return accuracy-like proxy
        y_pred = prob.argmax(axis=1)
        return float((y_pred == y_true).mean())


@torch.no_grad()
def evaluate_lia_attack(
    cfg,
    priv_cfg: PrivacyConfig,
    attacker: SAESLAttacker,
    victim_batch: VictimBatch,
    device: str,
) -> LIAEval:
    """
    Evaluate LIA on victim smashed activations A_front.
    Output P_label using cfg.metrics mapping (default: identity on AUC).
    """
    attacker.eval()

    # evaluate in chunks to avoid allocating huge intermediate tensors
    N = victim_batch.num_samples
    # prefer aux batch size as a hint, cap to a reasonable size
    try:
        hint_bs = int(priv_cfg.aux.batch_size)
    except Exception:
        hint_bs = 64
    eval_bs = max(1, min(hint_bs, 64))

    logits_list = []
    for i in range(0, N, eval_bs):
        chunk = victim_batch.A_front[i : i + eval_bs].to(device)
        logits_chunk, _ = attacker.forward_victim(chunk)
        logits_list.append(logits_chunk.detach().cpu())

    logits = torch.cat(logits_list, dim=0)
    prob = F.softmax(logits, dim=1).numpy()
    y_true = victim_batch.y.detach().cpu().numpy()
    y_pred = prob.argmax(axis=1)

    acc = float((y_pred == y_true).mean())
    auc = _compute_multiclass_auc(y_true, prob, multi_class=str(getattr(cfg.metrics, "multi_class", "ovr")))
    P_label = lia_auc_to_privacy(auc, priv_cfg)
    return LIAEval(auc=auc, acc=acc, P_label=P_label)


@torch.no_grad()
def evaluate_mia_attack(
    cfg,
    priv_cfg: PrivacyConfig,
    attacker: SAESLAttacker,
    victim_batch: VictimBatch,
    device: str,
) -> MIAEval:
    """
    Evaluate MIA (reconstruction) on victim smashed activations A_front.
    Requires victim_batch.x (original images) to compute reconstruction quality.
    """
    if victim_batch.x is None:
        return MIAEval(mse=float("nan"), P_sample=0.0)

    attacker.eval()

    # compute reconstruction in chunks to avoid OOM
    N = victim_batch.num_samples
    try:
        hint_bs = int(priv_cfg.aux.batch_size)
    except Exception:
        hint_bs = 64
    eval_bs = max(1, min(hint_bs, 64))

    total_se = 0.0
    total_elems = 0
    x_true = victim_batch.x.to(device)

    for i in range(0, N, eval_bs):
        chunk_A = victim_batch.A_front[i : i + eval_bs].to(device)
        _, x_hat_chunk = attacker.forward_victim(chunk_A)
        x_true_chunk = x_true[i : i + eval_bs]

        # sum of squared errors for this chunk
        se = F.mse_loss(x_hat_chunk, x_true_chunk, reduction="sum").item()
        total_se += float(se)
        total_elems += int(x_true_chunk.numel())

    # mean mse over all elements
    mse = float(total_se / max(1, total_elems))
    P_sample = mia_quality_to_privacy(mse, priv_cfg)
    return MIAEval(mse=mse, P_sample=P_sample)
