# src/metrics/objectives.py
import csv
import json
import os
from typing import Dict, List, Optional

import numpy as np


def _resolve_log_paths(cfg, cut_dir: str = None) -> Dict[str, str]:
    """
    根据 cfg 推断 train/val 日志文件路径，
    逻辑要和 MetricsLogger 里的保持一致。
    """
    output_dir = os.path.abspath(cfg.experiment.output_dir)
    if cut_dir is not None:
        output_dir = os.path.join(output_dir, cut_dir)

    log_cfg = getattr(cfg, "logging", None)
    if log_cfg is not None and getattr(log_cfg, "train_csv", None) is not None:
        train_csv = os.path.join(output_dir, log_cfg.train_csv)
    else:
        train_csv = os.path.join(output_dir, "train_metrics.csv")
    if log_cfg is not None and getattr(log_cfg, "val_csv", None) is not None:
        val_csv = os.path.join(output_dir, log_cfg.val_csv)
    else:
        val_csv = os.path.join(output_dir, "val_metrics.csv")
    
    obj_cfg = getattr(cfg, "objective", None)
    if obj_cfg is not None and getattr(obj_cfg, "scale_csv", None) is not None:
        scale_csv = os.path.abspath(obj_cfg.scale_csv)
    else:
        scale_csv = os.path.join(
            output_dir.replace(output_dir[output_dir.find('_'):], "_offline_profiling"), 
            "scales.csv"
        )

    return {"train_csv": train_csv, "val_csv": val_csv, "scale_csv": scale_csv}


def _load_best_val_acc(val_csv: str, mode: str = "best") -> float:
    """
    从 val_metrics.csv 中取最终 / 最优 val_acc。
    mode: "best" 或 "final"
    """
    if not os.path.isfile(val_csv):
        raise FileNotFoundError(f"Validation metrics file not found: {val_csv}")

    best_acc = 0.0
    final_acc = 0.0

    with open(val_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        has_row = False
        for row in reader:
            has_row = True
            acc = float(row.get("val_acc", 0.0))
            final_acc = acc
            if acc > best_acc:
                best_acc = acc

    if not has_row:
        return 0.0

    if mode == "final":
        return final_acc
    return best_acc


def _load_train_totals(train_csv: str) -> Dict[str, float]:
    """
    从 train_metrics.csv 中累加总通信量 / 总计算时间，并返回各自的最小/最大值。
    这里假定 logger 至少写了：
      - comm_time
      - comp_time
    """
    if not os.path.isfile(train_csv):
        raise FileNotFoundError(f"Train metrics file not found: {train_csv}")

    comm_total = 0.0
    comp_total = 0.0
    comp_client_total = 0.0
    comp_server_total = 0.0

    with open(train_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # 安全解析数值，解析失败则当作 0.0
            try:
                comm = float(row.get("comm_time", 0.0))
            except (TypeError, ValueError):
                comm = 0.0
            try:
                comp = float(row.get("comp_time", 0.0))
            except (TypeError, ValueError):
                comp = 0.0
            try:
                comp_client = float(row.get("comp_time_client", 0.0))
            except (TypeError, ValueError):
                comp_client = 0.0
            try:
                comp_server = float(row.get("comp_time_server", 0.0))
            except (TypeError, ValueError):
                comp_server = 0.0

            comm_total += comm
            comp_total += comp
            comp_client_total += comp_client
            comp_server_total += comp_server

    return {
        "comm_total": comm_total,
        "comp_total": comp_total,
        "comp_client_total": comp_client_total,
        "comp_server_total": comp_server_total,
    }


def _load_scales(scale_csv: str) -> Dict[str, float]:
    """
    从 scales.csv 中加载通信量和计算时间的 min/max。
    """
    if not os.path.isfile(scale_csv):
        raise FileNotFoundError(f"Scale metrics file not found: {scale_csv}")

    scales = {}
    with open(scale_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = row.get("metric", "")
            try:
                value = float(row.get("value", 0.0))
            except (TypeError, ValueError):
                value = 0.0
            scales[key] = {"value": value}
    return scales


def count_total_latency(cfg, cut_keys: List[str]) -> Dict[str, List[float]]:
    """
    计算所有 cut 下的总通信时延和总计算时延。
    """
    comm_totals: List[float] = []
    comp_totals: List[float] = []

    for cut_key in cut_keys:
        paths = _resolve_log_paths(cfg, cut_dir=cut_key)
        train_csv = paths["train_csv"]
        totals = _load_train_totals(train_csv)
        comm = totals["comm_total"]
        comp = totals["comp_total"]
        comm_totals.append(comm)
        comp_totals.append(comp)

    if not comm_totals or not comp_totals:
        raise RuntimeError("No comm/comp totals collected. Check logging and train loop.")

    return {"comm_totals": comm_totals, "comp_totals": comp_totals}


def compute_final_objective(
    cfg,
    privacy_score: float,
    save_json: bool = True,
    cut_dir: str = None,
    comm_min: Optional[float] = None,
    comm_max: Optional[float] = None,
    comp_min: Optional[float] = None,
    comp_max: Optional[float] = None,
) -> Dict[str, float]:
    """
    根据最终模型精度、总通信量、总计算时间和 LIA AUC 计算全局 J。

    Parameters
    ----------
    cfg : SimpleNamespace
        训练使用的配置（包含 experiment/output_dir, logging, objective 等）。
    privacy_score : float
        隐私分数。
    save_json : bool
        是否把结果写到 output_dir/global_objective.json
    cut_dir : str, optional
        切分子目录名，如果有的话。
    min_comm : float, optional
        最小通信量（用于 min-max 归一化）。
    max_comm : float, optional
        最大通信量（用于 min-max 归一化）。
    min_comp : float, optional
        最小计算时间（用于 min-max 归一化）。
    max_comp : float, optional
        最大计算时间（用于 min-max 归一化）。

    Returns
    -------
    result : dict
        {
          "acc_final",
          "comm_total",
          "comp_total",
          "comp_client_total",
          "comp_server_total",
          "privacy_score",
          "acc_cost",
          "comm_cost",
          "comp_cost",
          "comp_cost_client",
          "comp_cost_server",
          "privacy_cost",
          "J"
        }
    """
    paths = _resolve_log_paths(cfg, cut_dir=cut_dir)
    train_csv = paths["train_csv"]
    val_csv = paths["val_csv"]
    scale_csv = paths["scale_csv"]

    # 1) 取最终/最优精度
    obj_cfg = getattr(cfg, "objective", None)
    acc_mode = "best"
    if obj_cfg is not None and getattr(obj_cfg, "acc_mode", None) in ("best", "final"):
        acc_mode = obj_cfg.acc_mode
    acc_final = _load_best_val_acc(val_csv, mode=acc_mode)
    acc_cost = 1.0 - acc_final

    # 2) 累加总通信量 / 总计算时间
    totals = _load_train_totals(train_csv)
    comm_total = totals["comm_total"]
    comp_total = totals["comp_total"]
    comp_client_total = totals["comp_client_total"]
    comp_server_total = totals["comp_server_total"]

    # 3) 隐私项
    privacy_cost = privacy_score

    # 4) 归一化
    normalize_method = getattr(obj_cfg, "normalize_method", "scale") if obj_cfg is not None else "scale"
    if normalize_method == "scale":
        scales = _load_scales(scale_csv)
        s_comm = scales.get("comm_time_scale", {}).get("value", 1.0)
        s_comp = scales.get("comp_time_scale", {}).get("value", 1.0)
        comm_cost = comm_total / (1e-6 + s_comm)
        comp_cost = comp_total / (1e-6 + s_comp)
        comp_cost_client = comp_client_total / (1e-6 + s_comp)
        comp_cost_server = comp_server_total / (1e-6 + s_comp)
    elif normalize_method == "minmax":
        if comm_min is None or comm_max is None or comp_min is None or comp_max is None:
            raise ValueError("min_comm, max_comm, min_comp, max_comp must be provided for minmax normalization.")
        comm_cost = (comm_total - comm_min) / max(1e-6, (comm_max - comm_min))
        comp_cost = (comp_total - comp_min) / max(1e-6, (comp_max - comp_min))
        comp_cost_client = comp_cost * (comp_client_total / comp_total) if comp_total > 0 else 0.0
        comp_cost_server = comp_cost * (comp_server_total / comp_total) if comp_total > 0 else 0.0
    else:
        comm_cost = comm_total
        comp_cost = comp_total
        comp_cost_client = comp_client_total
        comp_cost_server = comp_server_total

    # 5) 加权求 J
    if obj_cfg is not None and getattr(obj_cfg, "weights", None) is not None:
        w = obj_cfg.weights
        w_acc = float(getattr(w, "acc_cost", 1))
        w_comm = float(getattr(w, "comm", 1))
        w_comp = float(getattr(w, "comp", 1))
        w_priv = float(getattr(w, "privacy", 1))
    else:
        w_acc = w_comm = w_comp = w_priv = 1

    J = (
        w_acc * acc_cost
        + w_comm * comm_cost
        + w_comp * comp_cost
        + w_priv * privacy_cost
    ) / max(1e-6, w_acc + w_comm + w_comp + w_priv)

    result = {
        "acc_final": acc_final,
        "comm_total": comm_total,
        "comp_total": comp_total,
        "comp_client_total": comp_client_total,
        "comp_server_total": comp_server_total,
        "privacy_score": privacy_score,
        "acc_cost": acc_cost,
        "comm_cost": comm_cost,
        "comp_cost": comp_cost,
        "comp_cost_client": comp_cost_client,
        "comp_cost_server": comp_cost_server,
        "privacy_cost": privacy_cost,
        "J": J,
    }

    if save_json:
        out_dir = os.path.abspath(cfg.experiment.output_dir)
        os.makedirs(out_dir, exist_ok=True)
        if cut_dir is not None:
            out_dir = os.path.join(out_dir, cut_dir)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "global_objective.json")
        try:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    return result
