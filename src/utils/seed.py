# src/utils/seed.py
"""
Utility function to set random seed for reproducibility.
"""

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """统一设置随机种子，保证可复现性。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    