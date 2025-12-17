# src/utils/seed.py
"""
Utility function to set random seed for reproducibility.
"""

import random

import numpy as np
import torch


def set_seed(seed_master: int, seed_torch: int) -> None:
    """统一设置随机种子，保证可复现性。"""
    random.seed(seed_master)
    np.random.seed(seed_master)
    torch.manual_seed(seed_torch)
    torch.cuda.manual_seed_all(seed_torch)
    