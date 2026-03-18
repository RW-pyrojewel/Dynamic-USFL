# src/privacy/sae.py
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Tuple, Iterator, Optional

import numpy as np
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
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
    SAEUSLAttacker,
    build_front_template_from_backbone,
    ShadowTail,                      # <--- 新增导入
    build_back_template_from_backbone # <--- 新增导入
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
    rho = float(priv_cfg.sae.mixmatch.labeled_fraction)
    eps = 1e-6
    rho = min(max(rho, eps), 1.0 - eps)

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
    if feat.dim() != 2:
        raise ValueError(f"Expected [B,C] features, got {tuple(feat.shape)}")
    b = feat.size(0)
    if b <= 1:
        return torch.zeros(feat.size(1), feat.size(1), device=feat.device, dtype=feat.dtype)
    x = feat - feat.mean(dim=0, keepdim=True)
    return (x.t() @ x) / (b - 1)


def _latent_align_loss(z_l: torch.Tensor, z_u: torch.Tensor) -> torch.Tensor:
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
    cut2: int,
    victim_batch: VictimBatch,
    device: str,
) -> SAEUSLAttacker:
    """
    Build SAE-USL attacker. Dynamically adjusts architecture if supervised-grad is enabled.
    """
    img_shape = _infer_img_shape_from_cfg(cfg)
    lia_algo = getattr(priv_cfg.sae, "lia_algorithm", "mix-match")

    if lia_algo == "supervised-grad":
        if victim_batch.grad_A_back is None:
            raise ValueError("supervised-grad LIA requires 'grad_A_back' in VictimBatch.")
        # LabelHead accepts gradients (which possess 4D shapes equivalent to A_back)
        z_spec_lia = LatentSpec.from_tensor(victim_batch.grad_A_back)
        tail_template = build_back_template_from_backbone(backbone_template, cut2)
        tail = ShadowTail(tail_template, reinit=True).to(device)
    elif lia_algo == "mix-match":
        # Original logic based on A_front
        z_spec_lia = LatentSpec.from_tensor(victim_batch.A_front)
        tail = None
    elif lia_algo == "zeroshot-grad":
        # Zero-Shot LIA does not require a trainable attacker model
        z_spec_lia = None
        tail = None
    else:
        raise NotImplementedError(f"Unsupported LIA algorithm: {lia_algo}")

    # Shadow encoder template: f1 structure (layers[0..cut1]) for MIA
    front_template = build_front_template_from_backbone(backbone_template, cut1)
    encoder = ShadowEncoder(front_template, reinit=True).to(device)

    if lia_algo in ("mix-match", "supervised-grad"):
        label_head = LabelHead(
            z_spec=z_spec_lia,
            num_classes=int(cfg.data.num_classes),
            hidden_dim=priv_cfg.sae.lia_hidden_dim,
            dropout=priv_cfg.sae.lia_dropout,
        ).to(device)
    else:
        label_head = None

    # STRICT mirror decoder
    decoder = MirrorDecoder.from_encoder(
        encoder_front=encoder.front,
        z_spec=LatentSpec.from_tensor(victim_batch.A_front),
        img_shape=img_shape,
        out_act=priv_cfg.sae.decoder_out_act,
    ).to(device)

    return SAEUSLAttacker(encoder=encoder, decoder=decoder, label_head=label_head, tail=tail).to(device)


def train_or_load_sae_attacker(
    cfg,
    priv_cfg: PrivacyConfig,
    cut_key: str,
    cut1: int,
    cut2: int,
    backbone_template: USFLBackbone,
    victim_batch: VictimBatch,
    aux_loader: DataLoader,
    device: str,
) -> SAEUSLAttacker:
    """
    Train SAE-USL attacker. Branching logic natively supports 'mix-match' and 'supervised-grad'.
    """
    output_dir = os.path.join(cfg.experiment.output_dir)
    cut_dir = os.path.join(output_dir, cut_key)
    if os.path.exists(cut_dir):
        output_dir = cut_dir
    ckpt_path = os.path.join(output_dir, "checkpoints", "sae_attacker.pth")

    attacker = build_attacker_for_cut(cfg, priv_cfg, backbone_template, cut1, cut2, victim_batch, device)

    if os.path.exists(ckpt_path) and not priv_cfg.sae.force_retrain:
        state = torch.load(ckpt_path, map_location=device)
        attacker.load_state_dict(state["model"])
        return attacker

    lia_algo = getattr(priv_cfg.sae, "lia_algorithm", "mix-match")

    # Optimizer parameters
    opt = torch.optim.Adam(
        params=attacker.parameters(),
        lr=priv_cfg.sae.lr,
        weight_decay=priv_cfg.sae.weight_decay,
    )

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

    labeled_loader = _create_labeled_loader(cfg, priv_cfg, aux_loader, victim_batch, z_victim_cpu)
    steps_per_epoch = max(1, max(len(labeled_loader), len(victim_loader)))

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
            try:
                x_l, y_l = next(labeled_iter)
            except StopIteration:
                labeled_iter = iter(labeled_loader)
                x_l, y_l = next(labeled_iter)

            # Advance victim_iter to keep timing identical (used actively in mixmatch)
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

            opt.zero_grad()
            loss = 0.0

            if lia_algo == "supervised-grad":
                # ==========================================================
                # Option 1: Supervised Gradient LIA (No MixMatch)
                # ==========================================================
                loss_lia = 0.0
                if priv_cfg.sae.enable_lia:
                    # 1. Simulate forward pass to cut2
                    a_front_aux = attacker.encoder(x_l)
                    z = a_front_aux
                    for i in range(cut1 + 1, cut2 + 1):
                        backbone_template.layers[i].to(device).eval()  # Ensure backbone is in eval mode for stable gradients
                        for p in backbone_template.layers[i].parameters():
                            p.requires_grad = False
                        z = backbone_template.layers[i](z)
                    a_back_aux = z
                    a_back_aux.retain_grad()
                    
                    # 2. Forward through ShadowTail and backward to calculate gradients
                    logits_proxy = attacker.tail(a_back_aux)
                    loss_proxy = F.cross_entropy(logits_proxy, y_l)
                    loss_proxy.backward(retain_graph=True) # Populates gradients for attacker.encoder and attacker.tail
                    
                    # 3. Use the extracted gradients to train LabelHead
                    grad_aux = a_back_aux.grad.detach()
                    logits_head = attacker.label_head(grad_aux)
                    loss_lia = F.cross_entropy(logits_head, y_l)
                    
                    loss = loss + priv_cfg.sae.lambda_lia * loss_lia

                loss_rec = 0.0
                if priv_cfg.sae.enable_mia:
                    # MIA uses original encoder and decoder architecture (up to cut1)
                    z_l = attacker.encoder(x_l)
                    xhat_l = attacker.decoder(z_l)
                    loss_rec = F.mse_loss(xhat_l, x_l)
                    loss = loss + priv_cfg.sae.lambda_rec * loss_rec

                if type(loss) is torch.Tensor:
                    loss.backward() # Computes gradients for encoder, decoder, label_head and tail (if exists)
                opt.step()

            elif lia_algo == "zeroshot-grad":
                # ==========================================================
                # Option 2: Zero-Shot Gradient LIA (Train MIA Only)
                # ==========================================================
                # 在这个分支里，不训练 LabelHead 和 ShadowTail，只训练 AutoEncoder
                z_l = attacker.encoder(x_l)

                # 1. 潜空间对齐（如果开启）：拉近辅助特征与截获的受害者特征分布
                loss_align = 0.0
                if lambda_align > 0:
                    loss_align = _latent_align_loss(z_l, z_u)
                    loss = loss + lambda_align * loss_align

                # 2. MIA 重构损失
                loss_rec = 0.0
                if priv_cfg.sae.enable_mia:
                    xhat_l = attacker.decoder(z_l)
                    loss_rec = F.mse_loss(xhat_l, x_l)
                    loss = loss + priv_cfg.sae.lambda_rec * loss_rec

                # 反向传播，更新 ShadowEncoder 和 MirrorDecoder
                if type(loss) is torch.Tensor:
                    loss.backward()
                    opt.step()
            
            elif lia_algo == "mix-match":
                # ==========================================================
                # Original: Smashed Data LIA with MixMatch
                # ==========================================================
                z_l = attacker.encoder(x_l)

                loss_align = 0.0
                if lambda_align > 0:
                    loss_align = _latent_align_loss(z_l, z_u)

                with torch.no_grad():
                    logits_u1, _ = attacker.forward_latent(z_u)
                    logits_u2, _ = attacker.forward_latent(z_u)
                    p_u = (F.softmax(logits_u1, dim=1) + F.softmax(logits_u2, dim=1)) / 2.0
                    q_u = _sharpen(p_u, T)

                y_l_oh = _onehot(y_l, num_classes)
                z_all = torch.cat([z_l, z_u], dim=0)
                y_all = torch.cat([y_l_oh, q_u], dim=0)

                perm = torch.randperm(z_all.size(0), device=device)
                z2 = z_all[perm]
                y2 = y_all[perm]

                z_mix, y_mix = _mixup(z_all, y_all, z2, y2, alpha=alpha)

                z_l_mix = z_mix[: z_l.size(0)]
                y_l_mix = y_mix[: z_l.size(0)]
                z_u_mix = z_mix[z_l.size(0):]
                y_u_mix = y_mix[z_l.size(0):]

                logits_l, _ = attacker.forward_latent(z_l_mix)
                logits_u, _ = attacker.forward_latent(z_u_mix)

                logp_l = F.log_softmax(logits_l, dim=1)
                Lx = -(y_l_mix * logp_l).sum(dim=1).mean()

                p_u_mix = F.softmax(logits_u, dim=1)
                Lu = F.mse_loss(p_u_mix, y_u_mix)

                loss_lia = Lx + lambda_u * Lu

                loss_rec = 0.0
                if priv_cfg.sae.enable_mia:
                    xhat_l = attacker.decoder(z_l)
                    loss_rec = F.mse_loss(xhat_l, x_l)

                if priv_cfg.sae.enable_lia:
                    loss = loss + priv_cfg.sae.lambda_lia * loss_lia
                if lambda_align > 0:
                    loss = loss + lambda_align * loss_align
                if priv_cfg.sae.enable_mia:
                    loss = loss + priv_cfg.sae.lambda_rec * loss_rec

                if type(loss) is torch.Tensor:
                    loss.backward()
                opt.step()
            
            else:
                raise NotImplementedError(f"Unsupported LIA algorithm: {lia_algo}")

            bs = int(x_l.size(0))
            epoch_loss += float(loss.item() if type(loss) is torch.Tensor else 0) * bs
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
    attacker: SAEUSLAttacker,
    victim_batch: VictimBatch,
    device: str,
) -> LIAEval:
    """
    Evaluate LIA smoothly handles both A_front (MixMatch) and grad_A_back (Supervised-Grad).
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
        chunk_A = victim_batch.A_front[i: i + eval_bs].to(device)
        chunk_grad = None
        if victim_batch.grad_A_back is not None:
            chunk_grad = victim_batch.grad_A_back[i: i + eval_bs].to(device)
            
        logits_chunk = attacker.label_head(chunk_grad) if chunk_grad is not None else attacker.label_head(chunk_A)
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
    priv_cfg: PrivacyConfig,
    attacker: SAEUSLAttacker,
    victim_batch: VictimBatch,
    device: str,
) -> MIAEval:
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
        x_hat_chunk = attacker.decoder(chunk_A)
        x_true_chunk = x_true[i: i + eval_bs]

        se = F.mse_loss(x_hat_chunk, x_true_chunk, reduction="sum").item()
        total_se += float(se)
        total_elems += int(x_true_chunk.numel())

    mse = float(total_se / max(1, total_elems))
    P_sample = mia_quality_to_privacy(mse, priv_cfg)
    return MIAEval(mse=mse, P_sample=P_sample)


def evaluate_zeroshot_grad_lia(
    priv_cfg: PrivacyConfig,
    victim_batch: VictimBatch,
    num_classes: int
) -> LIAEval:
    if victim_batch.grad_A_back is None:
        raise ValueError("Zero-Shot LIA requires 'grad_A_back' in VictimBatch.")

    grads = victim_batch.grad_A_back.cpu().numpy()
    N = grads.shape[0]
    grads_flat = grads.reshape(N, -1)

    norms = np.linalg.norm(grads_flat, axis=1, keepdims=True)
    norms[norms == 0] = 1e-12 
    grads_norm = grads_flat / norms

    kmeans = KMeans(n_clusters=num_classes, n_init=10, random_state=42)
    cluster_preds = kmeans.fit_predict(grads_norm)

    y_true = victim_batch.y.cpu().numpy()
    cost_matrix = np.zeros((num_classes, num_classes))
    for i in range(num_classes):
        for j in range(num_classes):
            cost_matrix[i, j] = -np.sum((cluster_preds == i) & (y_true == j))
            
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    mapping = {row: col for row, col in zip(row_ind, col_ind)}
    
    y_pred_mapped = np.array([mapping[c] for c in cluster_preds])

    acc = float((y_pred_mapped == y_true).mean())
    auc = acc 
    P_label = lia_auc_to_privacy(auc, priv_cfg)
    
    return LIAEval(auc=auc, acc=acc, P_label=P_label)
