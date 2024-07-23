import torch
import torch.nn as nn
from torch import Tensor

class GNLLoss(nn.GaussianNLLLoss):

    def __init__(self, *, full: bool = False, eps: float = 1e-6, reduction: str = 'mean'):
        super().__init__(full=full, eps=eps, reduction=reduction)


    def forward(self, y_hat, y, var) -> Tensor:
        var = torch.exp(var)
        rmse = torch.sqrt(torch.mean((y_hat-y)**2))
        loss = super().forward(y_hat, y, var)
        return {
            'loss': loss,
            'rmse': rmse,
        }