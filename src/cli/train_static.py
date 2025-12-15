# src/cli/train_static.py
"""
Entry point for running static dual-cut USFL baselines.

Example:
    python -m src.cli.train_static \
        --config configs/ham18_static_usfl.yaml

With overrides:
    python -m src.cli.train_static \
        --config configs/ham18_static_usfl.yaml \
        --override training.epochs=50 optimizer.lr=0.001
"""

import argparse
import csv
import os

import torch

from src.config.parser import load_config, parse_overrides
from src.privacy.api import evaluate_privacy_for_cut
from src.usfl.train_loop import train_static_usfl

from src.metrics.logger import MetricsLogger
from src.metrics.objectives import compute_final_objective
from src.utils.builder import build_backbone, build_dataloaders
from src.utils.seed import set_seed


def main():
    parser = argparse.ArgumentParser(description="Train static USFL baseline.")
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

    # 1. load config
    overrides = parse_overrides(args.override) if args.override else None
    cfg = load_config(args.config, overrides=overrides)
    exp_name = cfg.experiment.name
    output_dir = cfg.experiment.output_dir
    
    # 2. select cut points
    # cut1, cut2 = select_static_cut(cfg)
    cut1, cut2 = 0, 1   # 固定 cut 点
    
    # 3. set seed
    seed = getattr(cfg, "seed", 52)
    set_seed(seed)
    
    # 4. build dataloaders
    train_loader, val_loader = build_dataloaders(cfg)

    # 5. build backbone model
    backbone = build_backbone(cfg)

    # 6. build loss, objective function, and logger
    criterion = torch.nn.CrossEntropyLoss()

    # logger: 负责写 csv / tensorboard 等
    os.makedirs(output_dir, exist_ok=True)
    logger = MetricsLogger(exp_name=exp_name, output_dir=output_dir, cfg=cfg)

    # 7. run training loop (static USFL)
    train_static_usfl(
        cfg=cfg,
        backbone=backbone,
        cuts=(cut1, cut2),
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        logger=logger,
    )
    
    # 8. run attacks to evaluate privacy
    privacy_res = evaluate_privacy_for_cut(
        cfg=cfg,
        cut_key=f"cut_{cut1}_{cut2}",
        cut_dir=os.path.abspath(output_dir),
        cut1=cut1,
        backbone_template=backbone,   # 仅用于“切结构”，内部会重置权重避免泄漏
        device=cfg.training.device,
    )        
    result = compute_final_objective(
        cfg, 
        privacy_score=privacy_res.P_global, 
        save_json=True,
    )
    
    acc_mode = (getattr(cfg.objective, "acc_mode", "final") if hasattr(cfg, "objective") else "final").capitalize()

    print(f"\n=== Objective Summary for Cut {cut1} & {cut2} ===")
    print(f"{acc_mode} accuracy      : {result['acc_final']:.4f}")
    print(f"Total communication time   : {result['comm_total']:.3e}")
    print(f"Total computation time     : {result['comp_total']:.3f} s")
    print(f"Privacy score              : {result['privacy_score']:.4f}")
    print(f"J_global                   : {result['J']:.6f}")
    print(f"(detail saved to {os.path.join(output_dir, 'global_objective.json')})")


def select_static_cut(cfg) -> tuple[int, int]:
    usfl_cfg = getattr(cfg, "usfl", None)
    if usfl_cfg is None:
        raise ValueError("Config does not contain 'usfl' section.")
    
    j_score_csv = getattr(usfl_cfg.split, "j_score_csv", None) if hasattr(usfl_cfg, "split") else None
    if j_score_csv is None:
        raise ValueError("Config 'usfl.split' does not contain 'j_score_csv' path.")
    if not os.path.isfile(j_score_csv):
        raise FileNotFoundError(f"J score CSV file not found: {j_score_csv}")
    
    best_cut = (0, 1)
    best_j_score = float("inf")
    
    with open(j_score_csv, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                cut1 = int(row.get("cut1", 0))
            except (TypeError, ValueError):
                cut1 = 0
            try:
                cut2 = int(row.get("cut2", 1))
            except (TypeError, ValueError):
                cut2 = 1
            try:
                j_score = float(row.get("J", float("inf")))
            except (TypeError, ValueError):
                j_score = float("inf")
            if j_score < best_j_score:
                best_j_score = j_score
                best_cut = (cut1, cut2)
    
    return best_cut


if __name__ == "__main__":
    main()
