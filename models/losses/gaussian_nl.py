import torch
import torch.nn as nn
from torch import Tensor
from models.losses.base import Loss

class GNLLoss(nn.GaussianNLLLoss):

    def __init__(self, *, full: bool = False, eps: float = 1e-6, reduction: str = 'mean'):
        self.loss_func = nn.GaussianNLLLoss(full=full, eps=eps, reduction=reduction)


    def forward(self, y_hat, y, var) -> Tensor:
        losses = super().forward(y_hat, y)
        var = torch.exp(var)
        loss = self.loss_func(y_hat, y, var)
        losses['loss'] = loss
        return losses