import torch
import random
from torch import nn
from validators import Max
from MainClasses.HiPool import HiPool, HiPoolPlus, HiPoolFixed
def _ensure_mask(mask, x):
    if mask is None:
        return None
    if mask.dim() == 2:
        return mask.unsqueeze(-1)
    return mask


class MaxPool(nn.Module):
    def __init__(self):
        super().__init__()
        pass
    def forward(self, x, mask=None):
        mask = _ensure_mask(mask, x)
        if mask is not None:
            masked = x.masked_fill(mask == 0, torch.finfo(x.dtype).min)
            return torch.max(masked, dim=1)[0]
        return torch.max(x, dim=1)[0]

class AvgPool(nn.Module):
    def __init__(self):
        super().__init__()
        pass
    def forward(self, x, mask=None):
        mask = _ensure_mask(mask, x)
        if mask is not None:
            denom = mask.sum(dim=1).clamp_min(1.0)
            return (x * mask).sum(dim=1) / denom
        return torch.mean(x, dim=1)

class LinearSoftmaxPool(nn.Module):
    def __init__(self):
        super().__init__()
        pass
    def forward(self, x, mask=None):
        mask = _ensure_mask(mask, x)
        if mask is not None:
            numerator = torch.sum((x ** 2) * mask, dim=1)
            denominator = torch.sum(x * mask, dim=1).clamp_min(1e-8)
            return numerator / denominator
        return torch.sum(x ** 2, dim=1) / torch.sum(x, dim=1)

class ExpSoftmaxPool(nn.Module):
    def __init__(self):
        super().__init__()
        pass
    def forward(self, x, mask=None):
        mask = _ensure_mask(mask, x)
        exp_x = torch.exp(x)
        if mask is not None:
            numerator = torch.sum(x * exp_x * mask, dim=1)
            denominator = torch.sum(exp_x * mask, dim=1).clamp_min(1e-8)
            return numerator / denominator
        return (torch.sum(x * exp_x, dim=1) / torch.sum(exp_x, dim=1))

class AttentionPool(nn.Module):
    def __init__(self, seq_len):
        super(AttentionPool, self).__init__()
        self.linear = nn.Linear(seq_len, seq_len)
    def forward(self, inputs, mask=None):
        alphas = torch.sigmoid(self.linear(inputs.permute(0, 2, 1)))
        alphas = alphas.permute(0, 2, 1)
        mask = _ensure_mask(mask, inputs)
        if mask is not None:
            alphas = alphas * mask
            denom = alphas.sum(dim=1).clamp_min(1e-8)
            return torch.sum(inputs * alphas, dim=1) / denom
        return torch.sum(inputs * alphas, dim=1) / torch.sum(alphas, dim=1)

class AutoPool(nn.Module):
    """ Adaptive pooling operators for Multiple Instance Learning
    Adapted original code.
    This layer automatically adapts the pooling behavior to interpolate
    between min, mean and max-pooling for each class.
    Link : https://github.com/marl/autopool
    Args:
       input_size (int): Lenght of input_vector
       time_axis (int): Axis along which to perform the pooling.
          By default 1 (should be time) ie. (batch_size, time_sample_size, input_size)
    """
    def __init__(self, n_classes, time_axis=1):
        super(AutoPool, self).__init__()
        self.time_axis = time_axis
        self.alpha = nn.Parameter(torch.zeros(1, n_classes), requires_grad=True)

    def forward(self, x, mask=None):
        scaled = self.alpha * x
        mask = _ensure_mask(mask, x)
        if mask is not None:
            scaled = scaled.masked_fill(mask == 0, torch.finfo(x.dtype).min)
        weights = torch.softmax(scaled, dim=self.time_axis)
        return (x * weights).sum(dim=self.time_axis)

class PowerPool(nn.Module):
    """
    Power Pooling: An Adaptive Pooling Function for Weakly Labelled Sound Event Detection
    """
    def __init__(self, input_size, time_axis=1):
        super(PowerPool, self).__init__()
        self.time_axis = time_axis
        self.n = nn.Parameter(torch.zeros(1, input_size), requires_grad=True)
    def forward(self, x, mask=None):
        scaled = torch.pow(x, self.n)
        mask = _ensure_mask(mask, x)
        if mask is not None:
            numerator = torch.sum(x * scaled * mask, dim=1)
            denominator = torch.sum(scaled * mask, dim=1).clamp_min(1e-8)
            return numerator / denominator
        return torch.sum(x * scaled, dim=1) / torch.sum(scaled, dim=1)


class MILPooling(nn.Module):
    """
    Define the pooling functions for multi-instance learning, based on PyTorch.
    All the y_i is (B, T, C) dimensions.
    """
    def __init__(self, n_classes, seq_len=None):
        self.seq_len = seq_len
        self.n_classes = n_classes
    def get_pool(self, pool_name):
        if pool_name == 'max_pool':
            return MaxPool()
        elif pool_name == 'avg_pool':
            return AvgPool()
        elif pool_name == 'linear_pool':
            return LinearSoftmaxPool()
        elif pool_name == 'exp_pool':
            return ExpSoftmaxPool()
        elif pool_name == 'attention_pool':
            return AttentionPool(self.seq_len)
        elif pool_name == 'auto_pool':
            return AutoPool(self.n_classes)
        elif pool_name == 'power_pool':
            return PowerPool(self.n_classes)
        elif pool_name == 'hi_pool':
            return HiPool(self.seq_len, self.n_classes)
        elif pool_name == 'hi_pool_plus':
            return HiPoolPlus(self.seq_len, self.n_classes)
        elif pool_name == 'hi_pool_fixed':
            return HiPoolFixed(self.seq_len, self.n_classes)
        else:
            print('Error pooling style!')
            return
