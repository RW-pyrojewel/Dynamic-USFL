import argparse
import csv
import os
from typing import Dict, List, Tuple

import torch

from src.config.parser import load_config, parse_overrides
from src.privacy.api import PrivacyResult, evaluate_privacy_for_cut
from src.usfl.train_loop import train_static_usfl

from src.metrics.logger import MetricsLogger
from src.metrics.objectives import compute_final_objective, count_per_epoch_latency_scale, count_total_latency
from src.utils.builder import build_backbone, build_dataloaders
from src.utils.seed import set_seed


def main():
    parser = argparse.ArgumentParser(description="Offline profiling over all USFL cuts.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--override",
        nargs="*",
        default=[],
        help="Optional config overrides in key=value form, e.g. training.epochs=50",
    )

    args = parser.parse_args()

    # 0. load config
    overrides = parse_overrides(args.override) if args.override else None
    cfg = load_config(args.config, overrides=overrides)
    exp_name = cfg.experiment.name
    output_dir = cfg.experiment.output_dir
    os.makedirs(output_dir, exist_ok=True)

    frontend_cut_range = cfg.usfl.split.frontend_cut_range
    backend_cut_range = cfg.usfl.split.backend_cut_range

    seed = cfg.seed
    set_seed(seed.master, seed.torch)

    # ---------- 全局容器 ----------
    cut_pairs: List[Tuple[int, int]] = []
    privacy_scores: Dict[str, PrivacyResult] = {}
    results: Dict[str, Dict] = {}
    j_rows: List[Dict] = []
    static_rows: List[Dict] = []

    acc_mode = (
        getattr(cfg.objective, "acc_mode", "final")
        if hasattr(cfg, "objective")
        else "final"
    ).capitalize()

    # -------------------------------
    # 阶段一：逐 cut 训练（只跑 train，不算 J）
    # -------------------------------            
    for cut1 in range(frontend_cut_range[0], frontend_cut_range[1] + 1):
        for cut2 in range(backend_cut_range[0], backend_cut_range[1] + 1):
            if cut2 < cut1:
                continue
            
            cut_pairs.append((cut1, cut2))
            cut_key = f"cut_{cut1}_{cut2}"
            cut_dir_name = cut_key

            # 1.2 build dataloaders
            train_loader, val_loader = build_dataloaders(cfg)

            # 1.3 build backbone model
            backbone = build_backbone(cfg)

            # 1.4 loss & logger
            criterion = torch.nn.CrossEntropyLoss()
            logger = MetricsLogger(
                exp_name=exp_name,
                output_dir=output_dir,
                cfg=cfg,
                cut_dir=cut_dir_name,
            )

            # 1.5 run static USFL training for this cut
            train_static_usfl(
                cfg=cfg,
                backbone=backbone,
                cuts=(cut1, cut2),
                train_loader=train_loader,
                val_loader=val_loader,
                criterion=criterion,
                logger=logger,
            )

            # 1.6 隐私分数
            privacy_res = evaluate_privacy_for_cut(
                cfg=cfg,
                cut_key=f"cut_{cut1}_{cut2}",
                cut_dir=os.path.join(output_dir, f"cut_{cut1}_{cut2}"),
                cut1=cut1,
                backbone_template=backbone,   # 仅用于“切结构”，内部会重置权重避免泄漏
                device=cfg.training.device,
            ) if cut1 != cut2 else PrivacyResult(P_label=0.0, P_sample=0.0, P_global=0.0)
            privacy_scores[cut_key] = privacy_res

            # 清理
            del train_loader, val_loader, backbone, criterion, logger
            torch.cuda.empty_cache()

    # ---------------------------------
    # 阶段二：根据所有 cut 的 raw totals 算 min/max
    # ---------------------------------
    totals = count_total_latency(cfg, cut_keys=[f"cut_{c1}_{c2}" for c1, c2 in cut_pairs])
    comm_totals = totals["comm_totals"]
    comp_totals = totals["comp_totals"]
    comm_min, comm_max = min(comm_totals), max(comm_totals)
    comp_min, comp_max = min(comp_totals), max(comp_totals)

    # ---------------------------------
    # 阶段三：用 min/max 归一化，算每个 cut 的 J，并写 CSV
    # ---------------------------------
    for cut_pair in cut_pairs:
        cut1, cut2 = cut_pair
        cut_key = f"cut_{cut1}_{cut2}"

        result = compute_final_objective(
            cfg,
            privacy_score=privacy_scores[cut_key].P_global,
            save_json=False,
            cut_dir=cut_key,
            comm_min=comm_min,
            comm_max=comm_max,
            comp_min=comp_min,
            comp_max=comp_max,
        )
        results[cut_key] = result

        # ---------- 1) J 值记录 ----------
        j_rows.append(
            {
                "cut1": cut1,
                "cut2": cut2,
                "acc_final": result["acc_final"],
                "acc_cost": result["acc_cost"],
                "comm_total": result["comm_total"],
                "comm_cost": result["comm_cost"],
                "comp_total": result["comp_total"],
                "comp_cost": result["comp_cost"],
                "comp_client_total": result["comp_client_total"],
                "comp_cost_client": result["comp_cost_client"],
                "comp_server_total": result["comp_server_total"],
                "comp_cost_server": result["comp_cost_server"],
                "privacy_score_global": result["privacy_score"],
                "privacy_score_label": privacy_scores[cut_key].P_label,
                "privacy_score_sample": privacy_scores[cut_key].P_sample,
                "privacy_cost": result["privacy_cost"],
                "J": result["J"],
            }
        )

        # ---------- 2) 静态成本记录 ----------
        w_acc = getattr(cfg.objective.weights, "acc_cost", 1.0)
        w_priv = getattr(cfg.objective.weights, "privacy", 1.0)
        w_comp = getattr(cfg.objective.weights, "comp", 1.0)
        w_comm = getattr(cfg.objective.weights, "comm", 1.0)

        static_cost = (
            w_acc * result["acc_cost"]
            + w_priv * result["privacy_cost"]
            + w_comp * result.get("comp_cost_client", 0.0)
        ) / (w_acc + w_priv + w_comp + w_comm)

        static_rows.append(
            {
                "cut1": cut1,
                "cut2": cut2,
                "acc_cost": result["acc_cost"],
                "privacy_cost": result["privacy_cost"],
                "comp_cost_client": result["comp_cost_client"],
                "static_cost": static_cost,
            }
        )

        # ---------- 3) 控制台打印 ----------
        print(f"\n=== Objective Summary for Cut {cut1} & {cut2} ===")
        print(f"{acc_mode} accuracy            : {result['acc_final']:.4f}")
        print(f"Total communication time    : {result['comm_total']:.3e}")
        print(f"Total computation time      : {result['comp_total']:.3f} s")
        print(f"Privacy score (raw)         : {result['privacy_score']:.4f}")
        print(f"J_global                    : {result['J']:.6f}")

    # 4. 全部 cut 遍历完：选出最优 cut 并打印汇总
    best_cut_key = min(results.keys(), key=lambda k: results[k]["J"])
    best_J = results[best_cut_key]["J"]

    print("\n=== All Cut Points Summary ===")
    for cut_key, result in results.items():
        print(f"\n--- {cut_key} ---")
        print(f"{acc_mode} accuracy            : {result['acc_final']:.4f}")
        print(f"Total communication time    : {result['comm_total']:.3e}")
        print(f"Total computation time      : {result['comp_total']:.3f} s")
        print(f"Privacy score (raw)         : {result['privacy_score']:.4f}")
        print(f"J_global                    : {result['J']:.6f}")
    print("\n=============================\n")
    print(f"The optimal cut points are {best_cut_key} with J = {best_J:.4f}")

    # 5. 计算通信时延和计算时延的尺度
    scales = count_per_epoch_latency_scale(
        cfg,
        cut_keys=[f"cut_{c1}_{c2}" for c1, c2 in cut_pairs],
    )
    scale_rows = [
        {"metric": "comm_time_scale", "value": scales["comm_scale"]},
        {"metric": "comp_time_scale", "value": scales["comp_scale"]},
    ]
    
    # 6. 写三个 CSV 文件（J / static / scales）
    _write_profiling_csvs(
        cfg,
        output_dir,
        j_rows,
        static_rows,
        scale_rows,
    )


def _write_profiling_csvs(
    cfg,
    output_dir: str,
    j_rows: List[Dict],
    static_rows: List[Dict],
    scale_rows: List[Dict],
) -> None:
    logging_cfg = cfg.logging

    # 1) J 值 CSV
    j_csv_path = os.path.join(output_dir, logging_cfg.j_score_csv)
    if j_rows:
        fieldnames = list(j_rows[0].keys())
        with open(j_csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(j_rows)
        print(f"[offline_profiling] J scores saved to: {j_csv_path}")

    # 2) 静态成本 CSV
    static_csv_path = os.path.join(output_dir, logging_cfg.static_cost_csv)
    if static_rows:
        fieldnames = list(static_rows[0].keys())
        with open(static_csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(static_rows)
        print(f"[offline_profiling] Static costs saved to: {static_csv_path}")

    # 3) scales.csv：给在线阶段的 scale 归一化用
    scale_csv_path = os.path.join(output_dir, logging_cfg.scale_csv)
    if scale_rows:
        fieldnames = list(scale_rows[0].keys())
        with open(scale_csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(scale_rows)
        print(f"[offline_profiling] Scales saved to: {scale_csv_path}")


if __name__ == "__main__":
    main()
