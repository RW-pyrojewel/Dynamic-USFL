# src/cli/train_dynamic.py
"""
Entry point for running dynamic dual-cut USFL baselines.

Example:
    python -m src.cli.train_dynamic \
        --config configs/ham18_dynamic_usfl.yaml

With overrides:
    python -m src.cli.train_dynamic \
        --config configs/ham18_dynamic_usfl.yaml \
        --override training.epochs=50 optimizer.lr=0.001
"""

import argparse
import os
import csv
from collections import Counter

import torch

from src.config.parser import load_config, parse_overrides
from src.privacy.api import PrivacyResult, evaluate_privacy_for_cut
from src.usfl.train_loop import train_dynamic_usfl

from src.metrics.logger import MetricsLogger
from src.metrics.objectives import compute_final_objective
from src.data.factory import build_dataloaders
from src.models.factory import build_backbone
from src.utils.seed import set_seed


def main():
    parser = argparse.ArgumentParser(description="Train dynamic USFL baseline.")
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
    
    # 2. set seed
    seed = getattr(cfg, "seed", {"master": 42, "torch": 42})
    set_seed(seed.master, seed.torch)

    # 3. build dataloaders
    train_loader, val_loader = build_dataloaders(cfg)

    # 4. build backbone model
    backbone = build_backbone(cfg)

    # 5. build loss, objective function, and logger
    criterion = torch.nn.CrossEntropyLoss()

    # logger: 负责写 csv / tensorboard 等
    os.makedirs(output_dir, exist_ok=True)
    logger = MetricsLogger(exp_name=exp_name, output_dir=output_dir, cfg=cfg)

    # 6. run training loop (dynamic USFL)
    train_dynamic_usfl(
        cfg=cfg,
        backbone=backbone,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        logger=logger,
    )
    
    # 7. run attacks to evaluate privacy
    privacy_counts = {}
    for cut1 in range(len(backbone.layers) - 1):
        for cut2 in range(cut1, len(backbone.layers) - 1):
            cut_key = f"cut_{cut1}_{cut2}"
            cut_dir = os.path.join(output_dir, cut_key)
            if not os.path.exists(cut_dir):
                continue
            privacy_res = evaluate_privacy_for_cut(
                cfg=cfg,
                cut_key=cut_key,
                cut_dir=cut_dir,
                cut1=cut1,
                backbone_template=backbone,   # 仅用于“切结构”，内部会重置权重避免泄漏
                device=cfg.training.device,
            ) if cut1 != cut2 else PrivacyResult(P_label=0.0, P_sample=0.0, P_global=0.0)
            privacy_counts[cut_key] = privacy_res.P_global
            print(f"Privacy score for {cut_key}: {privacy_res.P_global:.4f}")

    # Compute weighted privacy score using bandit decisions counts
    # bandit CSV path is stored relative to output_dir at cfg.logging.bandit_csv
    bandit_counts = Counter()
    bandit_csv_rel = getattr(getattr(cfg, 'logging', None), 'bandit_csv', None)
    total_weight = 0
    if bandit_csv_rel:
        bandit_csv_path = os.path.join(output_dir, bandit_csv_rel)
        if os.path.exists(bandit_csv_path):
            try:
                with open(bandit_csv_path, newline='') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        try:
                            c1 = int(row.get('chosen_cut1', ''))
                            c2 = int(row.get('chosen_cut2', ''))
                        except Exception:
                            # skip rows with invalid ints
                            continue
                        key = f"cut_{c1}_{c2}"
                        bandit_counts[key] += 1
                        total_weight += 1
            except Exception:
                bandit_counts = Counter()

    # If we have counts, compute weighted average over privacy_counts keys
    if bandit_counts:
        weighted_sum = 0.0
        for k, v in privacy_counts.items():
            w = bandit_counts.get(k, 0)
            weighted_sum += v * w
        # If none of the evaluated cuts appear in bandit file, fall back to simple mean
        if total_weight > 0:
            privacy_score = weighted_sum / total_weight
        else:
            privacy_score = sum(privacy_counts.values()) / len(privacy_counts) if privacy_counts else 0.0
    else:
        # fallback to uniform average when bandit CSV missing or unreadable
        privacy_score = sum(privacy_counts.values()) / len(privacy_counts) if privacy_counts else 0.0
    
    result = compute_final_objective(
        cfg, 
        privacy_score=privacy_score, 
        save_json=True,
    )
    
    acc_mode = (getattr(cfg.objective, "acc_mode", "final") if hasattr(cfg, "objective") else "final").capitalize()

    print(f"\n=== Objective Summary for LinUCB-based Dynamic Dual Cut ===")
    print(f"{acc_mode} accuracy      : {result['acc_final']:.4f}")
    print(f"Total communication time   : {result['comm_total']:.3e}")
    print(f"Total computation time     : {result['comp_total']:.3f} s")
    print(f"Privacy score              : {result['privacy_score']:.4f}")
    print(f"J_global                   : {result['J']:.6f}")
    print(f"(detail saved to {os.path.join(output_dir, 'global_objective.json')})")


if __name__ == "__main__":
    main()
    