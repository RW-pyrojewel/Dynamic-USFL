# src/usfl/train_loop.py
import os
from typing import Iterable, Optional, Dict
import torch
from torch import nn
from torch.utils.data import DataLoader
from copy import deepcopy
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
import numpy as np

from src.metrics.logger import MetricsLogger
from src.models.model_usfl import USFLBackbone
from src.data.data_distribution import create_client_loaders
from src.network.simulator import build_network_simulator
from src.usfl.aggregation import aggregate
from src.usfl.static_split import StaticSplitUSFL


def build_optimizer(cfg, params):
    opt_type = str(cfg.optimizer.type).lower()
    # 配置可能来自 YAML/命令行，类型可能为字符串——强制转为数值以避免类型错误
    lr = float(getattr(cfg.optimizer, "lr", 0.0))
    momentum = float(getattr(cfg.optimizer, "momentum", 0.0))
    weight_decay = float(getattr(cfg.optimizer, "weight_decay", 0.0))

    if opt_type == "sgd":
        return torch.optim.SGD(
            params,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
        )
    elif opt_type == "adam":
        return torch.optim.Adam(
            params,
            lr=lr,
            weight_decay=weight_decay,
        )
    else:
        raise ValueError(f"Unknown optimizer type: {cfg.optimizer.type}")


# def build_scheduler(cfg, optimizer):
#     if getattr(cfg.scheduler, "type", "none") == "step_lr":
#         return torch.optim.lr_scheduler.StepLR(
#             optimizer,
#             step_size=cfg.scheduler.step_size,
#             gamma=cfg.scheduler.gamma,
#         )
#     elif cfg.scheduler.type == "none":
#         return None
#     else:
#         raise ValueError(f"Unknown scheduler type: {cfg.scheduler.type}")


def train_static_usfl(
    cfg,
    backbone: USFLBackbone,
    cuts: Iterable[int],
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    logger: MetricsLogger,
) -> None:
    """
    静态 USFL 训练循环（单机版）。
    """

    device = cfg.training.device
    cut1, cut2 = cuts

    # ------------------------------
    # 0. checkpoint 目录 & best_acc
    # ------------------------------
    # 根据 logger 推断本次 run 的目录：output_dir / cut_x_y
    if hasattr(logger, "output_dir") and hasattr(logger, "cut_dir"):
        run_dir = os.path.join(logger.output_dir, logger.cut_dir)
    else:
        # 兜底，至少保证写在 experiment.output_dir 下
        run_dir = getattr(logger, "output_dir", getattr(cfg.experiment, "output_dir", "."))

    ckpt_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    # ------------------------------
    # 1. 隐私采样配置 & 缓冲区
    # ------------------------------
    privacy_cfg = getattr(cfg, "privacy", None)
    privacy_enabled = bool(getattr(privacy_cfg, "enable", False)) if privacy_cfg is not None else False

    if privacy_enabled:
        max_priv_samples = int(getattr(privacy_cfg, "max_samples", 5000))
        store_x = bool(getattr(privacy_cfg, "store_x", True))
        priv_A_front_buf: list[torch.Tensor] = []
        priv_y_buf: list[torch.Tensor] = []
        priv_x_buf: Optional[list[torch.Tensor]] = [] if store_x else None
        priv_num_collected: int = 0
        print(f"[privacy] Sampling enabled: max_samples={max_priv_samples}, store_x={store_x}")
    else:
        max_priv_samples = 0
        store_x = False
        priv_A_front_buf = None
        priv_y_buf = None
        priv_x_buf = None
        priv_num_collected = 0
        print("[privacy] Sampling disabled.")

    # ------------------------------
    # 2. 多客户端数据 & 网络模拟器
    # ------------------------------
    num_clients = cfg.usfl.num_clients
    client_loaders = create_client_loaders(
        num_clients,
        train_loader,
        iid=cfg.data.iid,
    )
    num_clients = len(client_loaders)  # 可能因为样本不足而减少客户端数量
    net_sim = build_network_simulator(cfg, num_clients)

    # 计算每个客户端的样本数（优先使用 DataLoader.dataset 的长度）
    sample_counts = []
    for cl in client_loaders:
        if hasattr(cl, "dataset"):
            try:
                sample_counts.append(len(cl.dataset))
            except Exception:
                sample_counts.append(0)
        else:
            sample_counts.append(0)

    # 将 local_steps 作为本地遍历次数（local epochs）
    local_epochs = max(1, int(cfg.usfl.local_steps))

    # 每个客户端维护自己的全局轮计数
    global_round_clients = [0 for _ in range(num_clients)]

    # 精确度/召回率/F1/AUC 计算方式
    average = getattr(cfg.metrics, "average", "macro")
    multi_class = getattr(cfg.metrics, "multi_class", "ovr")

    # 数据元素字节数
    bytes_per_elem = getattr(cfg.data, "bytes_per_elem", 4)

    # 将模型放到 CPU 上，按需移动到 GPU
    backbone.to("cpu")

    # ------------------------------
    # 3. 主训练循环
    # ------------------------------

    print(
        f"[USFL Train] Starting training for {cfg.training.epochs} epochs, "
        f"{num_clients} clients, local_epochs={local_epochs}."
    )

    for epoch in range(1, cfg.training.epochs + 1):
        # 本轮（epoch）: 每个客户端在其本地数据上训练 local_epochs 个 epoch
        client_state_dicts = []

        for c_idx, client_loader in enumerate(client_loaders):
            # 拷贝全局模型到客户端
            client_model = deepcopy(backbone)
            client_model.to(device)
            client_model.train()

            orchestrator = StaticSplitUSFL(
                backbone=client_model,
                cut1=cut1,
                cut2=cut2,
                enable_profiling=True,
                bytes_per_elem=bytes_per_elem,
            )

            optimizer_client = build_optimizer(cfg, client_model.parameters())

            # 在客户端上做 local_epochs 次完整遍历
            for le in range(1, local_epochs + 1):
                for batch_idx, (x, y) in enumerate(client_loader):
                    global_round_clients[c_idx] += 1  # 每处理一个 batch，客户端的全局轮计数加 1

                    x = x.to(device, non_blocking=True)
                    y = y.to(device, non_blocking=True)

                    optimizer_client.zero_grad()

                    logits, profiling = orchestrator.forward_three_segments(x)

                    loss = criterion(logits, y)
                    loss.backward()
                    optimizer_client.step()

                    # ------------------------------
                    # 2.1 记录训练 metrics
                    # ------------------------------
                    metrics = {
                        "acc": accuracy_score(
                            y.cpu(), logits.argmax(dim=1).cpu()
                        ),
                        "loss": loss.item(),
                    }

                    # ------------------------------
                    # 2.2 采样 A_front / x / y
                    # ------------------------------
                    if (
                        privacy_enabled
                        and priv_num_collected < max_priv_samples
                    ):
                        # StaticSplitUSFL 在 forward_three_segments 内部会更新 last_front_acts
                        A_front = getattr(orchestrator, "last_front_acts", None)
                        if A_front is not None:
                            with torch.no_grad():
                                A_front_cpu = A_front.detach().cpu()
                                y_cpu = y.detach().cpu()
                                x_cpu = x.detach().cpu() if store_x else None

                                batch_size = A_front_cpu.size(0)
                                remaining = max_priv_samples - priv_num_collected
                                if remaining > 0:
                                    if batch_size > remaining:
                                        # 从当前 batch 随机抽样 remaining 个
                                        idx = torch.randperm(batch_size)[:remaining]
                                        A_front_cpu = A_front_cpu[idx]
                                        y_cpu = y_cpu[idx]
                                        if store_x:
                                            x_cpu = x_cpu[idx]
                                        taken = remaining
                                    else:
                                        taken = batch_size

                                    priv_A_front_buf.append(A_front_cpu)
                                    priv_y_buf.append(y_cpu)
                                    if store_x and priv_x_buf is not None:
                                        priv_x_buf.append(x_cpu)

                                    priv_num_collected += taken

                    # ---- logging ----
                    if (global_round_clients[c_idx] % cfg.logging.log_interval) == 0:
                        logger.log_train_round(
                            client_idx=c_idx,
                            global_round=global_round_clients[c_idx],
                            epoch=epoch,
                            batch_idx=batch_idx,
                            metrics=metrics,
                            profiling=profiling,
                            net_sim=net_sim,
                        )

            # 本客户端本轮训练结束，收集 state_dict
            client_state_dicts.append(
                {k: v.cpu() for k, v in client_model.state_dict().items()}
            )

            # 清理显存
            del client_model
            del optimizer_client
            torch.cuda.empty_cache()

        # 本 epoch 所有客户端本地训练完成 -> 联邦服务器聚合（一次）
        aggregated_state = aggregate(
            client_state_dicts,
            sample_counts=sample_counts,
            method=cfg.usfl.aggregation,
        )
        backbone.load_state_dict(aggregated_state)
        backbone.to(device)

        # 聚合后在 epoch 级别额外评估一次全局模型
        if val_loader is not None and (epoch % cfg.training.eval_every == 0):
            val_stats = evaluate_backbone_on_val(
                backbone,
                val_loader,
                device,
                criterion,
                average=average,
                multi_class=multi_class,
            )
            logger.log_val_epoch(epoch=epoch, val_stats=val_stats)

            print(
                f"[USFL Train] Epoch {epoch}: Val Acc={val_stats['val_acc']:.4f}, "
                f"Val Loss={val_stats['val_loss']:.4f}"
            )

            # ------ 保存 checkpoint ------
            if epoch % cfg.training.save_every == 0:
                # 确保保存到 CPU，防止之后 device 切换
                state_dict_cpu = {k: v.cpu() for k, v in backbone.state_dict().items()}
                torch.save(
                    {
                        "epoch": epoch,
                        "state_dict": state_dict_cpu,
                        "cuts": (cut1, cut2),
                    },
                    os.path.join(ckpt_dir, f"ckpt_epoch_{epoch}.pth"),
                )
                print(
                    f"[ckpt] New best backbone saved at epoch {epoch}"
                )

        backbone.to("cpu")

    # ------------------------------
    # 4. 训练结束：保存最后一版 backbone
    # ------------------------------
    state_dict_cpu = {k: v.cpu() for k, v in backbone.state_dict().items()}
    torch.save(
        {
            "epoch": cfg.training.epochs,
            "state_dict": state_dict_cpu,
            "cuts": (cut1, cut2),
            "best_val_acc": None
        },
        os.path.join(ckpt_dir, f"ckpt_last.pth"),
    )
    print(f"[ckpt] Last backbone saved")

    # ------------------------------
    # 5. 保存隐私评估所需的样本
    # ------------------------------
    if privacy_enabled and priv_num_collected > 0:
        A_front_all = torch.cat(priv_A_front_buf, dim=0)
        y_all = torch.cat(priv_y_buf, dim=0)
        save_dict = {"A_front": A_front_all, "y": y_all}

        if store_x and priv_x_buf is not None and len(priv_x_buf) > 0:
            x_all = torch.cat(priv_x_buf, dim=0)
            save_dict["x"] = x_all

        priv_path = os.path.join(run_dir, "privacy_samples.pt")
        torch.save(save_dict, priv_path)
        print(
            f"[privacy] Saved {A_front_all.size(0)} samples "
            f"(store_x={store_x}) to {priv_path}"
        )
    elif privacy_enabled:
        print("[privacy] Enabled but no samples were collected; nothing saved.")


def evaluate_backbone_on_val(
    backbone: USFLBackbone,
    val_loader: DataLoader,
    device: str,
    criterion: nn.Module,
    average: str = "macro",
    multi_class: str = "ovr",
) -> Dict[str, float]:
    """
    简单 centralized 验证：直接用 backbone.forward。
    """
    backbone.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    # 用于计算额外指标
    y_trues = []
    y_preds = []
    y_scores = []

    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = backbone(x)
            loss = criterion(logits, y)

            batch_size = y.size(0)
            total_loss += loss.item() * batch_size

            # 预测与概率处理：支持二分类（logit dim==1 或 channels==1）与多分类
            if logits.dim() == 1 or (logits.dim() == 2 and logits.size(1) == 1):
                # 二分类 logits -> sigmoid prob for positive class
                probs = torch.sigmoid(logits.view(-1)).cpu().numpy()
                preds = (probs >= 0.5).astype(int)
            else:
                # 多分类 -> softmax 概率
                probs = nn.functional.softmax(logits, dim=1).cpu().numpy()
                preds = probs.argmax(axis=1)

            y_np = y.cpu().numpy()
            total_correct += int((preds == y_np).sum())
            total_samples += batch_size

            y_trues.append(y_np)
            y_preds.append(preds)
            y_scores.append(probs)

    # 聚合所有批次
    if len(y_trues) > 0:
        y_true = np.concatenate(y_trues, axis=0)
        y_pred = np.concatenate(y_preds, axis=0)
        # y_scores 可能是一维（binary）或二维（multiclass）
        if isinstance(y_scores[0], np.ndarray) and y_scores[0].ndim == 1 and y_scores[0].shape == ():
            # 保险处理，强转为一维数组
            y_score = np.concatenate([np.atleast_1d(s) for s in y_scores], axis=0)
        else:
            y_score = np.concatenate(y_scores, axis=0)
    else:
        y_true = np.array([], dtype=int)
        y_pred = np.array([], dtype=int)
        y_score = np.array([])

    avg_loss = total_loss / max(total_samples, 1)
    avg_acc = total_correct / max(total_samples, 1)

    # 计算 precision/recall/f1（默认 macro）
    try:
        val_precision = float(precision_score(y_true, y_pred, average=average, zero_division=0))
        val_recall = float(recall_score(y_true, y_pred, average=average, zero_division=0))
        val_f1 = float(f1_score(y_true, y_pred, average=average, zero_division=0))
    except Exception:
        val_precision = val_recall = val_f1 = 0.0

    # 计算 AUC：二分类使用正类概率，多分类使用 ovr
    val_auc = 0.0
    try:
        if y_score.size == 0:
            val_auc = 0.0
        else:
            # 二分类：y_score 可能为一维概率
            if y_score.ndim == 1 or (y_score.ndim == 2 and y_score.shape[1] == 1):
                val_auc = float(roc_auc_score(y_true, y_score.ravel()))
            else:
                # 多分类：传入概率矩阵
                val_auc = float(roc_auc_score(y_true, y_score, multi_class=multi_class))
    except Exception:
        val_auc = 0.0

    return {
        "val_loss": float(avg_loss),
        "val_acc": float(avg_acc),
        "val_precision": float(val_precision),
        "val_recall": float(val_recall),
        "val_f1": float(val_f1),
        "val_auc": float(val_auc),
    }
