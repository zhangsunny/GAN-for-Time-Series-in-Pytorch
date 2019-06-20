"""
自定义各种loss，方便理解损失函数
注意，所有的优化器都是最小化损失函数
所以，最大化时需要将损失值取负数
-E(Log(D(x)))
"""
import torch
from torch import nn


class LogLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, negation=True):
        log_val = torch.log(x)
        loss = torch.mean(log_val)
        if negation:
            loss = torch.neg(loss)
        return loss


class ItselfLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, negation=True):
        loss = torch.mean(x)
        if negation:
            loss = torch.neg(loss)
        return loss
