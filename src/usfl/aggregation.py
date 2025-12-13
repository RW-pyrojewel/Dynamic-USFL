import copy
import torch

def aggregate(state_dicts, sample_counts=None, method='fedavg'):
    if method == 'fedavg':
        return FedAvg(state_dicts, sample_counts)
    else:
        raise ValueError(f"Aggregation method '{method}' not recognized.")

def FedAvg(state_dicts, sample_counts=None):
    """
    支持三种输入形式：
    1) state_dicts: [sd1, sd2, ...] 且 sample_counts=None -> 简单平均（向后兼容）
    2) state_dicts: [sd1, sd2, ...], sample_counts=[n1, n2, ...] -> 按样本数加权平均
    3) state_dicts: [(sd1, n1), (sd2, n2), ...] -> 自动解包为 state_dicts 和 sample_counts
    """
    # 自动解包 (state_dict, count) 形式
    if sample_counts is None and len(state_dicts) > 0 and all(isinstance(x, (tuple, list)) and len(x) == 2 for x in state_dicts):
        state_dicts, sample_counts = zip(*state_dicts)
        state_dicts = list(state_dicts)
        sample_counts = list(sample_counts)

    # 如果未提供 sample_counts，则退回到原先的简单平均
    if sample_counts is None:
        w_avg = copy.deepcopy(state_dicts[0])
        for k in w_avg.keys():
            for i in range(1, len(state_dicts)):
                w_avg[k] += state_dicts[i][k]
            w_avg[k] = w_avg[k] / float(len(state_dicts))
        return w_avg

    # 验证 sample_counts 长度
    if len(sample_counts) != len(state_dicts):
        raise ValueError("Length of sample_counts must match number of state_dicts.")

    # 计算按样本数加权的平均
    total_samples = float(sum(sample_counts))
    if total_samples == 0:
        raise ValueError("Total number of samples must be > 0.")

    w_avg = copy.deepcopy(state_dicts[0])
    for k in w_avg.keys():
        # 将初始值置零并使用浮点类型以避免整型与浮点数相加失败
        v0 = w_avg[k]
        if isinstance(v0, torch.Tensor):
            w_avg[k] = torch.zeros_like(v0, dtype=torch.float32)
            for i in range(len(state_dicts)):
                w_avg[k] += state_dicts[i][k].to(torch.float32) * float(sample_counts[i])
            w_avg[k] = w_avg[k] / total_samples
        else:
            # 处理 numpy 数组或标量
            w_avg[k] = v0 * 0.0
            for i in range(len(state_dicts)):
                w_avg[k] += state_dicts[i][k] * float(sample_counts[i])
            w_avg[k] = w_avg[k] / total_samples
    return w_avg
