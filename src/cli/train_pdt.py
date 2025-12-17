# src/cli/train_pdt.py
"""
Entry point for running PDT dual-cut USFL baselines.

Example:
    python -m src.cli.train_pdt \
        --config configs/ham18_pdt_usfl.yaml

With overrides:
    python -m src.cli.train_pdt \
        --config configs/ham18_pdt_usfl.yaml \
        --override training.epochs=50 optimizer.lr=0.001
"""

import argparse
import os

import torch

from src.config.parser import load_config, parse_overrides
from src.usfl.train_loop import train_static_usfl

from src.metrics.logger import MetricsLogger
from src.metrics.objectives import compute_final_objective
from src.utils.builder import build_backbone, build_dataloaders
from src.utils.seed import set_seed


def main():
    parser = argparse.ArgumentParser(description="Train PDT USFL baseline.")
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
    
    # 2. select PDT cut points
    cut1, cut2 = 0, 0
    
    # 3. set seed
    seed = getattr(cfg, "seed", {"master": 42, "torch": 42})
    set_seed(seed.master, seed.torch)

    # 4. build dataloaders
    train_loader, val_loader = build_dataloaders(cfg)

    # 5. build backbone model
    backbone = build_backbone(cfg)

    # 6. build loss, objective function, and logger
    criterion = torch.nn.CrossEntropyLoss()

    # logger: 负责写 csv / tensorboard 等
    os.makedirs(output_dir, exist_ok=True)
    logger = MetricsLogger(exp_name=exp_name, output_dir=output_dir, cfg=cfg)

    # 7. run training loop (pdt USFL)
    train_static_usfl(
        cfg=cfg,
        backbone=backbone,
        cuts=(cut1, cut2),
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        logger=logger,
    )

    result = compute_final_objective(
        cfg, 
        privacy_score=0.0,  # PDT 无任何隐私暴露
        save_json=True,
    )
    
    acc_mode = (getattr(cfg.objective, "acc_mode", "final") if hasattr(cfg, "objective") else "final").capitalize()

    print(f"\n=== Objective Summary for PDT ===")
    print(f"{acc_mode} accuracy      : {result['acc_final']:.4f}")
    print(f"Total communication time   : {result['comm_total']:.3e}")
    print(f"Total computation time     : {result['comp_total']:.3f} s")
    print(f"Privacy score              : {result['privacy_score']:.4f}")
    print(f"J_global                   : {result['J']:.6f}")
    print(f"(detail saved to {os.path.join(output_dir, 'global_objective.json')})")


if __name__ == "__main__":
    main()
