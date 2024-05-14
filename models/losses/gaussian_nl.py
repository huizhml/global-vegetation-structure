import torch
import torch.nn as nn
from torch import Tensor

class GNLLoss(nn.GaussianNLLLoss):

    def __init__(self, *, full: bool = False, eps: float = 1e-6, reduction: str = 'mean'):
        super().__init__(full=full, eps=eps, reduction=reduction)


    def forward(self, y_hat, y) -> Tensor:
        var = torch.exp(y_hat[..., 101:])
        input = y_hat[..., :101]
        rmse = torch.sqrt(torch.mean((input-y)**2))
        loss = super().forward(input, y, var)
        return {
            'loss': loss,
            'rmse': rmse,
        }