# src/metrics/logger.py
import csv
import json
import os
from typing import Dict, Any, Optional

from datetime import datetime

from torch.utils.tensorboard import SummaryWriter

from src.network.simulator import NetworkSimulator


class MetricsLogger:
    """
    负责把训练/验证过程中的指标写到 CSV（和可选的 TensorBoard）。

    训练日志：每个 client、每个 batch 一条记录：
      log_train_round(
        client_idx, global_round, epoch, batch_idx,
        metrics, profiling, net_sim
      )

    - metrics: acc 和 loss 等指标
    - profiling: 来自 StaticSplitUSFL 的 profiling dict
                  （比如 bytes_up, bytes_down, comp_time, t_front, t_middle, t_back, smashed1_numel, smashed2_numel, cut_points 等）
    - net_sim: NetworkSimulator 实例，用于计算通信时间
    """

    def __init__(self, exp_name: str, output_dir: str, cfg, cut_dir: str = None) -> None:
        self.exp_name = exp_name
        self.output_dir = os.path.abspath(output_dir)
        os.makedirs(self.output_dir, exist_ok=True)
        
        if cut_dir is not None:
            self.output_dir = os.path.join(self.output_dir, cut_dir)
            os.makedirs(self.output_dir, exist_ok=True)

        # train / val CSV 路径
        # 优先使用 cfg.logging 设置的路径，否则默认 train_metrics.csv / val_metrics.csv
        log_cfg = getattr(cfg, "logging", None)
        if log_cfg is not None and getattr(log_cfg, "train_csv", None) is not None:
            self.train_csv = os.path.join(self.output_dir, log_cfg.train_csv)
        else:
            self.train_csv = os.path.join(self.output_dir, "train_metrics.csv")
        if log_cfg is not None and getattr(log_cfg, "val_csv", None) is not None:
            self.val_csv = os.path.join(self.output_dir, log_cfg.val_csv)
        else:
            self.val_csv = os.path.join(self.output_dir, "val_metrics.csv")

        # TensorBoard
        use_tb = bool(getattr(log_cfg, "tensorboard", False)) if log_cfg is not None else False
        self.tb_writer: Optional[SummaryWriter] = None
        if use_tb and SummaryWriter is not None:
            tb_dir = os.path.join(self.output_dir, "tb_logs")
            os.makedirs(tb_dir, exist_ok=True)
            self.tb_writer = SummaryWriter(log_dir=tb_dir)

        # 写一份 config snapshot，方便复现
        self._dump_config(cfg)

        # 初始化 CSV 头
        self._train_header_written = os.path.isfile(self.train_csv) and os.path.getsize(self.train_csv) > 0
        self._val_header_written = os.path.isfile(self.val_csv) and os.path.getsize(self.val_csv) > 0
        
        # 读取仿真配置
        sim_cfg = getattr(cfg, "simulation", None)
        comp_cfg = getattr(sim_cfg, "comp", None) if sim_cfg is not None else None
        self.kappa_client = int(getattr(comp_cfg, "kappa_client", 1)) if comp_cfg is not None else 1
        self.kappa_server = int(getattr(comp_cfg, "kappa_server", 1)) if comp_cfg is not None else 1

    # ---- public API ----

    def log_train_round(
        self,
        client_idx: int,
        global_round: int,
        epoch: int,
        batch_idx: int,
        metrics: Dict[str, float],
        profiling: Dict[str, Any],
        net_sim: NetworkSimulator,
    ) -> None:
        """
        每个 client 的每个 mini-batch 调用一次。
        """
        row = self._build_train_row(
            client_idx=client_idx,
            global_round=global_round,
            epoch=epoch,
            batch_idx=batch_idx,
            metrics=metrics,
            profiling=profiling,
            net_sim=net_sim,
        )

        # 写 CSV（追加）
        self._write_train_row(row)

        # 写 TensorBoard（可选）
        if self.tb_writer is not None:
            step = global_round
            self.tb_writer.add_scalar("train/acc", row["acc"], step)
            self.tb_writer.add_scalar("train/comm_time", row["comm_time"], step)
            self.tb_writer.add_scalar("train/comp_time", row["comp_time"], step)

    def log_val_epoch(self, epoch: int, val_stats: Dict[str, float]) -> None:
        """
        epoch 级的验证日志。
        val_stats 示例：
            {"val_loss": float, "val_acc": float}
        """
        row = self._build_val_row(epoch, val_stats)
        
        # 写 CSV（追加）
        self._write_val_row(row)

        if self.tb_writer is not None:
            self.tb_writer.add_scalar("val/loss", row["val_loss"], epoch)
            self.tb_writer.add_scalar("val/acc", row["val_acc"], epoch)
            self.tb_writer.add_scalar("val/precision", row["val_precision"], epoch)
            self.tb_writer.add_scalar("val/recall", row["val_recall"], epoch)
            self.tb_writer.add_scalar("val/f1", row["val_f1"], epoch)
            self.tb_writer.add_scalar("val/auc", row["val_auc"], epoch)

    def close(self):
        if self.tb_writer is not None:
            self.tb_writer.close()

    # ---- internal helpers ----

    def _dump_config(self, cfg) -> None:
        """
        把 cfg 转成一个尽量可 JSON 序列化的 dict 存下来。
        """
        cfg_path = os.path.join(self.output_dir, "config_snapshot.json")

        def to_plain(obj: Any) -> Any:
            if isinstance(obj, (int, float, str, bool)) or obj is None:
                return obj
            if isinstance(obj, (list, tuple)):
                return [to_plain(x) for x in obj]
            if isinstance(obj, dict):
                return {str(k): to_plain(v) for k, v in obj.items()}
            # SimpleNamespace 等
            if hasattr(obj, "__dict__"):
                return {k: to_plain(v) for k, v in obj.__dict__.items()}
            return str(obj)

        try:
            cfg_dict = to_plain(cfg)
            with open(cfg_path, "w", encoding="utf-8") as f:
                json.dump(cfg_dict, f, ensure_ascii=False, indent=2)
        except Exception:
            # 静默失败，不影响训练
            pass

    def _build_train_row(
        self,
        client_idx: int,
        global_round: int,
        epoch: int,
        batch_idx: int,
        metrics: Dict[str, float],
        profiling: Dict[str, Any],
        net_sim: NetworkSimulator,
    ) -> Dict[str, Any]:
        """
        把三类信息拼成一行扁平 dict，方便写 CSV。
        你之后要画图时就是读这个 CSV。
        """
        row: Dict[str, Any] = {
            "exp_name": self.exp_name,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "client_idx": client_idx,
            "global_round": global_round,
            "epoch": epoch,
            "batch_idx": batch_idx,
        }

        # acc / loss
        row["acc"] = float(metrics.get("acc", 0.0))
        row["loss"] = float(metrics.get("loss", 0.0))
        
        # simulated comm / comp time
        bytes_up = float(profiling.get("bytes_up", 0.0))
        bytes_down = float(profiling.get("bytes_down", 0.0))
        comm_time = net_sim.estimate_comm_time(client_idx, bytes_up, bytes_down)
        comp_time_client = float(profiling.get("comp_time_client", 0.0))
        comp_time_server = float(profiling.get("comp_time_server", 0.0))
        comp_time = self.kappa_client * comp_time_client + self.kappa_server * comp_time_server
        row["comm_time"] = comm_time
        row["comp_time"] = comp_time
        row["comp_time_client"] = comp_time_client
        row["comp_time_server"] = comp_time_server

        # profiling info（常用字段单独拉出来）
        cut_points = profiling.get("cut_points", None)
        if isinstance(cut_points, (list, tuple)) and len(cut_points) == 2:
            row["cut1"] = int(cut_points[0])
            row["cut2"] = int(cut_points[1])
        else:
            row["cut1"] = -1
            row["cut2"] = -1

        row["t_front"] = float(profiling.get("t_front", 0.0))
        row["t_middle"] = float(profiling.get("t_middle", 0.0))
        row["t_back"] = float(profiling.get("t_back", 0.0))
        row["smashed1_numel"] = float(profiling.get("smashed1_numel", 0.0))
        row["smashed2_numel"] = float(profiling.get("smashed2_numel", 0.0))

        return row

    def _write_train_row(self, row: Dict[str, Any]) -> None:
        # 保证目录存在
        os.makedirs(os.path.dirname(self.train_csv), exist_ok=True)
        fieldnames = [
            "exp_name",
            "timestamp",
            "client_idx",
            "global_round",
            "epoch",
            "batch_idx",
            "acc",
            "loss",
            "comm_time",
            "comp_time",
            "comp_time_client",
            "comp_time_server",
            "cut1",
            "cut2",
            "t_front",
            "t_middle",
            "t_back",
            "smashed1_numel",
            "smashed2_numel",
        ]

        write_header = not self._train_header_written
        with open(self.train_csv, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
                self._train_header_written = True
            writer.writerow(row)

    def _build_val_row(self, epoch: int, val_stats: Dict[str, float]) -> Dict[str, Any]:
        """
        构建验证 epoch 行。同时统计轮级通信和计算时间。
        """
        row = {
            "epoch": epoch,
            "val_loss": float(val_stats.get("val_loss", 0.0)),
            "val_acc": float(val_stats.get("val_acc", 0.0)),
            "val_precision": float(val_stats.get("val_precision", 0.0)),
            "val_recall": float(val_stats.get("val_recall", 0.0)),
            "val_f1": float(val_stats.get("val_f1", 0.0)),
            "val_auc": float(val_stats.get("val_auc", 0.0)),
        }
        
        # 读 training CSV 计算 comm_time 和 comp_time
        train_csv = self.train_csv
        # 默认值
        row["comm_time"] = 0.0
        row["comp_time"] = 0.0
        row["comp_time_client"] = 0.0
        row["comp_time_server"] = 0.0

        if not os.path.isfile(train_csv):
            return row

        per_client: Dict[int, Dict[str, float]] = {}
        try:
            with open(train_csv, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    # 解析 epoch
                    try:
                        r_epoch = int(r.get("epoch", "-1"))
                    except Exception:
                        try:
                            r_epoch = int(float(r.get("epoch", "-1")))
                        except Exception:
                            continue
                    if r_epoch != epoch:
                        continue

                    # 解析 client_idx
                    client_raw = r.get("client_idx", None)
                    if client_raw is None or client_raw == "":
                        continue
                    try:
                        client_idx = int(client_raw)
                    except Exception:
                        try:
                            client_idx = int(float(client_raw))
                        except Exception:
                            continue

                    # 读取数值字段，容错处理空串
                    def _f(key: str) -> float:
                        try:
                            return float(r.get(key, 0.0) or 0.0)
                        except Exception:
                            return 0.0

                    comm_time = _f("comm_time")
                    comp_time = _f("comp_time")
                    comp_time_client = _f("comp_time_client")
                    comp_time_server = _f("comp_time_server")

                    if client_idx not in per_client:
                        per_client[client_idx] = {
                            "comm_time": 0.0,
                            "comp_time": 0.0,
                            "comp_time_client": 0.0,
                            "comp_time_server": 0.0,
                        }

                    per_client[client_idx]["comm_time"] += comm_time
                    per_client[client_idx]["comp_time"] += comp_time
                    per_client[client_idx]["comp_time_client"] += comp_time_client
                    per_client[client_idx]["comp_time_server"] += comp_time_server
        except Exception:
            return row

        if not per_client:
            return row

        # 各 client 的累计 comm_time/comp_time 的最大值
        max_comm = max(v["comm_time"] for v in per_client.values())
        max_comp = max(v["comp_time"] for v in per_client.values())

        # 找到累计 comp_time 最大的 client，取其 comp_time_client / comp_time_server
        max_comp_client = max(per_client.items(), key=lambda kv: kv[1]["comp_time"])[0]
        comp_client_val = per_client[max_comp_client]["comp_time_client"]
        comp_server_val = per_client[max_comp_client]["comp_time_server"]

        row["comm_time"] = float(max_comm)
        row["comp_time"] = float(max_comp)
        row["comp_time_client"] = float(comp_client_val)
        row["comp_time_server"] = float(comp_server_val)

        return row
    
    def _write_val_row(self, row: Dict[str, Any]) -> None:
        os.makedirs(os.path.dirname(self.val_csv), exist_ok=True)
        fieldnames = [
            "epoch", 
            "val_loss", 
            "val_acc", 
            "val_precision", 
            "val_recall", 
            "val_f1", 
            "val_auc", 
            "comm_time", 
            "comp_time", 
            "comp_time_client", 
            "comp_time_server"
        ]

        write_header = not self._val_header_written
        with open(self.val_csv, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
                self._val_header_written = True
            writer.writerow(row)
