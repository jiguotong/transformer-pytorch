import torch
import numpy as np

PAD_TOKEN_INDEX = 0


def pad_masking(x, target_len):
    """获得填充区域的mask,可以忽略掉某些位置的计算"""
    # x: (batch_size, seq_len)
    batch_size, seq_len = x.size()
    padded_positions = x == PAD_TOKEN_INDEX  # (batch_size, seq_len)
    pad_mask = padded_positions.unsqueeze(1).expand(batch_size, target_len, seq_len)
    return pad_mask


def subsequent_masking(x):
    """上三角mask矩阵，用于Masked Multi-head Attention, 值为1的位置（上三角）后续会被处理为0或极小值"""
    # x: (batch_size, seq_len - 1)
    batch_size, seq_len = x.size()
    subsequent_mask = np.triu(np.ones(shape=(seq_len, seq_len)), k=1).astype('uint8')
    subsequent_mask = torch.tensor(subsequent_mask).to(x.device)
    subsequent_mask = subsequent_mask.unsqueeze(0).expand(batch_size, seq_len, seq_len)
    return subsequent_mask