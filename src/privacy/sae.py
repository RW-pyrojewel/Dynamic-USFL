# src/privacy/sae.py
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Tuple, Iterator

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, TensorDataset
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


def _create_labeled_loader(
    cfg,
    priv_cfg: PrivacyConfig,
    aux_loader: DataLoader,
    victim_batch: VictimBatch,
    z_victim_cpu: torch.Tensor,
) -> DataLoader:
    """
    Labeled pool from aux, sized relative to victim pool:
      rho = |L| / (|L| + |U|)  =>  |L| = |U| * rho / (1 - rho)
    where U is victim pool size (A_front count), L is labeled aux pool size.

    NOTE:
      This redefines `mixmatch.labeled_fraction` as rho above (overall labeled fraction),
      not "fraction of aux treated as labeled".
    """
    rho = float(priv_cfg.sae.mixmatch.labeled_fraction)
    eps = 1e-6
    rho = min(max(rho, eps), 1.0 - eps)  # avoid division by zero

    U = int(getattr(victim_batch, "num_samples", int(z_victim_cpu.size(0))))
    desired_L = int(round(U * rho / (1.0 - rho)))

    aux_ds = aux_loader.dataset
    aux_n = len(aux_ds)
    desired_L = max(1, min(desired_L, aux_n))

    if desired_L >= aux_n:
        labeled_loader = aux_loader
    else:
        g = torch.Generator()
        g.manual_seed(int(getattr(cfg.seed, "master", 42)))
        idx = torch.randperm(aux_n, generator=g)[:desired_L].tolist()
        labeled_loader = DataLoader(
            Subset(aux_ds, idx),
            batch_size=aux_loader.batch_size,
            shuffle=True,
            num_workers=aux_loader.num_workers,
            pin_memory=getattr(aux_loader, "pin_memory", False),
            drop_last=False,
        )
    return labeled_loader


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


def _mixup(
    x1: torch.Tensor,
    y1: torch.Tensor,
    x2: torch.Tensor,
    y2: torch.Tensor,
    alpha: float
) -> Tuple[torch.Tensor, torch.Tensor]:
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


def _next_victim_z_cpu(victim_loader: DataLoader, victim_iter: Iterator) -> Tuple[torch.Tensor, Iterator]:
    try:
        (z_,) = next(victim_iter)
    except StopIteration:
        victim_iter = iter(victim_loader)
        (z_,) = next(victim_iter)
    return z_, victim_iter


def _covariance(feat: torch.Tensor) -> torch.Tensor:
    """
    feat: [B, C]
    returns: [C, C] covariance
    """
    if feat.dim() != 2:
        raise ValueError(f"Expected [B,C] features, got {tuple(feat.shape)}")
    b = feat.size(0)
    if b <= 1:
        # avoid NaNs; return zeros
        return torch.zeros(feat.size(1), feat.size(1), device=feat.device, dtype=feat.dtype)
    x = feat - feat.mean(dim=0, keepdim=True)
    return (x.t() @ x) / (b - 1)


def _latent_align_loss(z_l: torch.Tensor, z_u: torch.Tensor) -> torch.Tensor:
    """
    Very light latent alignment regularizer between:
      - z_l = ShadowEncoder(x_aux) (trainable)
      - z_u = victim A_front (constant)
    Uses:
      - mean alignment (MSE of channel means)
      - CORAL (covariance alignment) on GAP features

    Both are computed on f = GAP(z) -> [B, C] to keep it lightweight.
    """
    f_l = F.adaptive_avg_pool2d(z_l, (1, 1)).flatten(1)
    f_u = F.adaptive_avg_pool2d(z_u, (1, 1)).flatten(1)

    mean_loss = F.mse_loss(f_l.mean(dim=0), f_u.mean(dim=0))
    cov_loss = F.mse_loss(_covariance(f_l), _covariance(f_u))
    return mean_loss + cov_loss


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

    # STRICT mirror decoder: traced from encoder.front with dummy forward
    decoder = MirrorDecoder.from_encoder(
        encoder_front=encoder.front,
        z_spec=z_spec,
        img_shape=img_shape,
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
    Train SAE-SL attacker with labeled auxiliary raw samples (x_aux, y_aux) and
    unlabeled victim smashed activations A_front (latent-space MixMatch) for LIA.
    MIA reconstruction is trained on aux labeled only; victim latents NEVER participate in MIA.
    """
    output_dir = os.path.join(cfg.experiment.output_dir)
    cut_dir = os.path.join(output_dir, cut_key)
    if os.path.exists(cut_dir):
        output_dir = cut_dir
    ckpt_path = os.path.join(output_dir, "checkpoints", "sae_attacker.pth")

    attacker = build_attacker_for_cut(cfg, priv_cfg, backbone_template, cut1, victim_batch, device)

    if os.path.exists(ckpt_path) and not priv_cfg.sae.force_retrain:
        state = torch.load(ckpt_path, map_location=device)
        attacker.load_state_dict(state["model"])
        return attacker

    # Unlabeled pool from victim smashed activations A_front (latent Z)
    # NOTE: Do NOT use victim labels during training.
    z_victim_cpu = victim_batch.A_front.detach().cpu()
    victim_bs = int(getattr(aux_loader, "batch_size", 32))
    victim_bs = max(1, min(victim_bs, int(z_victim_cpu.size(0))))
    victim_loader = DataLoader(
        TensorDataset(z_victim_cpu),
        batch_size=victim_bs,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
    )
    victim_iter = iter(victim_loader)

    # Labeled pool from aux dataset (subsampled by rho rule)
    labeled_loader = _create_labeled_loader(cfg, priv_cfg, aux_loader, victim_batch, z_victim_cpu)

    # step-driven: ensure each epoch has enough optimization steps
    steps_per_epoch = max(1, max(len(labeled_loader), len(victim_loader)))

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
    lambda_align = float(getattr(priv_cfg.sae, "lambda_align", 0.0))

    for epoch in trange(priv_cfg.sae.epochs, desc=f"[privacy] Train attacker {cut_key}", leave=False):
        attacker.train()
        epoch_loss = 0.0
        n_seen = 0

        labeled_iter = iter(labeled_loader)
        for _ in range(steps_per_epoch):
            # cycle labeled
            try:
                x_l, y_l = next(labeled_iter)
            except StopIteration:
                labeled_iter = iter(labeled_loader)
                x_l, y_l = next(labeled_iter)

            # fetch victim unlabeled latents (latent Z) and align batch size to x_l
            target_bs = int(x_l.size(0))
            z_chunks = []
            got = 0
            while got < target_bs:
                z_part, victim_iter = _next_victim_z_cpu(victim_loader, victim_iter)
                z_chunks.append(z_part)
                got += int(z_part.size(0))
            z_u = torch.cat(z_chunks, dim=0)[:target_bs]

            x_l = x_l.to(device, non_blocking=True)
            y_l = y_l.to(device, non_blocking=True)
            z_u = z_u.to(device, non_blocking=True)

            # ---- encode labeled aux into latent ----
            z_l = attacker.encoder(x_l)

            # ---- very light latent alignment regularizer (encoder-only) ----
            loss_align = 0.0
            if lambda_align > 0:
                loss_align = _latent_align_loss(z_l, z_u)

            # ---- MixMatch pseudo-label for victim unlabeled latents ----
            with torch.no_grad():
                logits_u1, _ = attacker.forward_latent(z_u)
                logits_u2, _ = attacker.forward_latent(z_u)
                p_u = (F.softmax(logits_u1, dim=1) + F.softmax(logits_u2, dim=1)) / 2.0
                q_u = _sharpen(p_u, T)

            # labeled one-hot
            y_l_oh = _onehot(y_l, num_classes)

            # concat for mixup (latent-space)
            z_all = torch.cat([z_l, z_u], dim=0)
            y_all = torch.cat([y_l_oh, q_u], dim=0)

            # shuffle pair for mixup
            perm = torch.randperm(z_all.size(0), device=device)
            z2 = z_all[perm]
            y2 = y_all[perm]

            z_mix, y_mix = _mixup(z_all, y_all, z2, y2, alpha=alpha)

            # split back
            z_l_mix = z_mix[: z_l.size(0)]
            y_l_mix = y_mix[: z_l.size(0)]
            z_u_mix = z_mix[z_l.size(0):]
            y_u_mix = y_mix[z_l.size(0):]

            # ---- forward on mixed latents ----
            opt.zero_grad()

            # labeled branch
            logits_l, _ = attacker.forward_latent(z_l_mix)
            # unlabeled branch (victim latents)
            logits_u, _ = attacker.forward_latent(z_u_mix)

            # supervised loss Lx
            logp_l = F.log_softmax(logits_l, dim=1)
            Lx = -(y_l_mix * logp_l).sum(dim=1).mean()

            # unsupervised loss Lu (MSE between probs and pseudo-labels)
            p_u_mix = F.softmax(logits_u, dim=1)
            Lu = F.mse_loss(p_u_mix, y_u_mix)

            loss_lia = Lx + lambda_u * Lu

            # reconstruction loss (aux labeled only) -- victim latents NEVER used here
            loss_rec = 0.0
            if priv_cfg.sae.enable_mia:
                # decoder-only forward to avoid MIA gradients affecting label_head
                xhat_l = attacker.decoder(z_l)
                loss_rec = F.mse_loss(xhat_l, x_l)

            loss = 0.0
            if priv_cfg.sae.enable_lia:
                loss = loss + priv_cfg.sae.lambda_lia * loss_lia
            if lambda_align > 0:
                loss = loss + lambda_align * loss_align
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

    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
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

    N = victim_batch.num_samples
    try:
        hint_bs = int(priv_cfg.aux.batch_size)
    except Exception:
        hint_bs = 64
    eval_bs = max(1, min(hint_bs, 64))

    logits_list = []
    for i in range(0, N, eval_bs):
        chunk = victim_batch.A_front[i: i + eval_bs].to(device)
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
        chunk_A = victim_batch.A_front[i: i + eval_bs].to(device)
        _, x_hat_chunk = attacker.forward_victim(chunk_A)
        x_true_chunk = x_true[i: i + eval_bs]

        se = F.mse_loss(x_hat_chunk, x_true_chunk, reduction="sum").item()
        total_se += float(se)
        total_elems += int(x_true_chunk.numel())

    mse = float(total_se / max(1, total_elems))
    P_sample = mia_quality_to_privacy(mse, priv_cfg)
    return MIAEval(mse=mse, P_sample=P_sample)
