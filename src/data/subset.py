from __future__ import annotations

from typing import Any, Callable, Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset


class TransformedSubset(Dataset):
    """A lightweight subset wrapper with an explicit transform.

    This is useful when we need multiple views (train/val/aux) over the same base
    dataset split, each with a different transform.
    """

    def __init__(
        self,
        dataset: Dataset,
        indices: Sequence[int] | torch.Tensor,
        transform: Optional[Callable] = None,
    ) -> None:
        self.dataset = dataset
        if isinstance(indices, torch.Tensor):
            self.indices = indices.to(dtype=torch.long)
        else:
            self.indices = torch.tensor(list(indices), dtype=torch.long)
        self.transform = transform

    def __len__(self) -> int:
        return int(self.indices.numel())

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        real_idx = int(self.indices[idx].item())
        x, y = self.dataset[real_idx]
        if self.transform is not None:
            x = self.transform(x)
        return x, y
