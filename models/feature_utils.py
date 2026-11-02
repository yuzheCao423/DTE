import torch
import torch.nn.functional as F
import numpy as np

def get_lags(context_length):
    if context_length <= 20:
        lags = [1, 2, 3]
    elif context_length <= 50:
        lags = [1, 2, 3, 5, 7]
    elif context_length <= 100:
        lags = [1, 2, 3, 5, 7, 10, 14]
    else:
        lags = [1, 2, 3, 5, 7, 14, 21, 30]
    return lags

def add_lag_features(x, lags):
    """
    为时间序列创建滞后特征

    参数:
        x: torch.Tensor, shape [Batch, Length, Dim]
        lags: list, 滞后步数列表

    返回:
        torch.Tensor, shape [Batch, Length, Dim + len(lags)]
    """
    features = [x]  # 原始特征
    for lag in lags:
        if lag > 0:
            lagged = F.pad(x[:, :-lag, :], (0, 0, lag, 0), value=0)
        else:
            raise ValueError("滞后步数必须为正整数")
        features.append(lagged)
    output = torch.cat(features, dim=-1)
    return output

def add_fourier_features(x, num_freqs=10):
    """
    x: shape (B, T, DIM)
    返回: shape (B, T, DIM + 2*num_freqs)
    """
    B, T, DIM = x.shape
    t = torch.arange(T, dtype=x.dtype, device=x.device)  # (T,)
    freqs = torch.arange(1, num_freqs + 1, dtype=x.dtype, device=x.device)  # (num_freqs,)
    angles = 2 * np.pi * t.unsqueeze(-1) * freqs.unsqueeze(0) / T  # (T, num_freqs)
    sin_features = torch.sin(angles)  # (T, num_freqs)
    cos_features = torch.cos(angles)  # (T, num_freqs)
    fourier_features = torch.cat([sin_features, cos_features], dim=-1)  # (T, 2*num_freqs)
    fourier_features = fourier_features.unsqueeze(0).expand(B, -1, -1)  # (B, T, 2*num_freqs)
    return torch.cat([x, fourier_features], dim=-1)  # (B, T, DIM + 2*num_freqs)
