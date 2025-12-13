import torch
from torch.utils.data import DataLoader, Subset
import numpy as np
from typing import List, Optional

def _extract_targets(dataset) -> Optional[np.ndarray]:
    """
    尝试从 dataset 中提取 targets，支持 Subset、datasets with .targets / .labels，
    否则回退到遍历 dataset（可能较慢）。
    返回 numpy array 长度为 len(dataset)，或 None（无法提取）。
    """
    # Subset: 需要构建相对于该 dataset 的 targets（dataset may be Subset of original）
    if isinstance(dataset, Subset):
        underlying = dataset.dataset
        indices = list(dataset.indices)
        # 尝试从 underlying 直接取 targets/labels
        if hasattr(underlying, "targets"):
            all_targets = np.array(underlying.targets)
            return all_targets[np.array(indices)]
        if hasattr(underlying, "labels"):
            all_targets = np.array(underlying.labels)
            return all_targets[np.array(indices)]
        # 回退：逐项访问 Subset 来读取标签
        try:
            targets = [underlying[i][1] for i in indices]
            return np.array(targets)
        except Exception:
            return None

    # 非 Subset dataset
    if hasattr(dataset, "targets"):
        return np.array(dataset.targets)
    if hasattr(dataset, "labels"):
        return np.array(dataset.labels)

    # 回退：遍历 dataset（可能会很慢，但作为最后手段）
    try:
        targets = [y for _, y in dataset]
        return np.array(targets)
    except Exception:
        return None


def create_client_loaders(num_clients: int, train_loader: DataLoader, iid: bool = True, alpha: float = 0.5, seed: Optional[int] = None) -> List[DataLoader]:
    """
    创建客户端 DataLoader 列表（支持 IID / Non-IID 两种划分）。
    - iid=True: 按样本均匀随机划分（torch.utils.data.random_split）。
    - iid=False: 使用 Dirichlet(alpha) 在类别维度上分配样本（常见非IID 划分）。
    参数:
      num_clients: 客户端数量
      train_loader: 原始 train DataLoader（用于获取 dataset 与 loader 参数）
      iid: 是否做 IID 划分
      alpha: Dirichlet 参数（越小越不均匀），仅在 iid=False 时有效
      seed: 随机种子（可选）
    返回:
      list of DataLoader（长度可能小于传入的 num_clients，当样本不足时会自动减少分出的 client 数）
    """
    dataset = train_loader.dataset
    total = len(dataset)
    if total == 0:
        raise ValueError("train_loader.dataset is empty")

    rng = np.random.default_rng(seed)

    # reuse train_loader settings where possible
    dl_kwargs = dict(
        batch_size=getattr(train_loader, "batch_size", 1),
        shuffle=True,
        num_workers=getattr(train_loader, "num_workers", 0),
        pin_memory=getattr(train_loader, "pin_memory", False),
        collate_fn=getattr(train_loader, "collate_fn", None),
    )

    # IID 划分（简单随机切分）
    if iid:
        # 均匀划分
        base = total // num_clients
        remainder = total % num_clients
        lengths = [base + (1 if i < remainder else 0) for i in range(num_clients)]
        # 如果某些 client 长度为 0（数据量比 clients 少），降低 num_clients
        if any(l == 0 for l in lengths):
            nonzero_lengths = [l for l in lengths if l > 0]
            if len(nonzero_lengths) == 0:
                raise ValueError("No data available to split among clients")
            lengths = nonzero_lengths
        subsets = torch.utils.data.random_split(dataset, lengths, generator=torch.Generator().manual_seed(seed or 0))
        client_loaders = [DataLoader(sub, **dl_kwargs) for sub in subsets]
        return client_loaders

    # Non-IID 划分（Dirichlet 按类划分）
    targets = _extract_targets(dataset)
    if targets is None:
        # 无法获取标签，退回到 IID 划分
        base = total // num_clients
        remainder = total % num_clients
        lengths = [base + (1 if i < remainder else 0) for i in range(num_clients)]
        subsets = torch.utils.data.random_split(dataset, lengths, generator=torch.Generator().manual_seed(seed or 0))
        return [DataLoader(sub, **dl_kwargs) for sub in subsets]

    # targets长度应等于 len(dataset)
    if len(targets) != len(dataset):
        # 保守回退到 IID
        base = total // num_clients
        remainder = total % num_clients
        lengths = [base + (1 if i < remainder else 0) for i in range(num_clients)]
        subsets = torch.utils.data.random_split(dataset, lengths, generator=torch.Generator().manual_seed(seed or 0))
        return [DataLoader(sub, **dl_kwargs) for sub in subsets]

    num_classes = int(int(targets.max()) + 1)
    # 为每个 client 初始化索引列表（相对于 dataset 的位置 0..len(dataset)-1）
    client_indices = [[] for _ in range(num_clients)]

    # 对每个类别进行 Dirichlet 分配
    for k in range(num_classes):
        idx_k = np.where(targets == k)[0]
        if len(idx_k) == 0:
            continue
        rng.shuffle(idx_k)
        # sample proportions for this class
        proportions = rng.dirichlet(alpha * np.ones(num_clients))
        # convert proportions to counts
        counts = (proportions * len(idx_k)).astype(int)
        # adjust to make sum == len(idx_k)
        while counts.sum() < len(idx_k):
            counts[np.argmax(proportions)] += 1
        while counts.sum() > len(idx_k):
            counts[np.argmax(counts)] -= 1
        start = 0
        for i in range(num_clients):
            cnt = counts[i]
            if cnt > 0:
                client_indices[i].extend(idx_k[start : start + cnt].tolist())
                start += cnt

    # 若某些 client 为空，则把一些样本补齐（从最大 client 拆分）
    empty_clients = [i for i, idxs in enumerate(client_indices) if len(idxs) == 0]
    if len(empty_clients) > 0:
        # 按数量从大到小取样本，分配给空 client
        nonempty = sorted([(i, len(idxs)) for i, idxs in enumerate(client_indices) if len(idxs) > 1], key=lambda x: -x[1])
        for ec in empty_clients:
            if not nonempty:
                break
            donor_idx, _ = nonempty[0]
            # move one sample
            moved = client_indices[donor_idx].pop()
            client_indices[ec].append(moved)
            # update donor order
            nonempty = sorted([(i, len(idxs)) for i, idxs in enumerate(client_indices) if len(idxs) > 1], key=lambda x: -x[1])

    # 如果任然存在空 client，删除这些 client（返回数量可能小于 num_clients）
    final_client_indices = [idxs for idxs in client_indices if len(idxs) > 0]
    if len(final_client_indices) == 0:
        raise ValueError("Failed to create non-empty client splits")

    # 构造 Subset + DataLoader 列表
    client_loaders = []
    for idxs in final_client_indices:
        subset = Subset(dataset, idxs)
        client_loaders.append(DataLoader(subset, **dl_kwargs))

    return client_loaders
