import torch
import torch.nn as nn
from torch import Tensor

class GNLLoss(nn.GaussianNLLLoss):

    def __init__(self, *, full: bool = False, eps: float = 1e-6, reduction: str = 'mean'):
        super().__init__(full=full, eps=eps, reduction=reduction)


    def forward(self, y_hat, y) -> Tensor:
        var = y_hat[..., 101:]
        input = y_hat[..., :101]
        return super().forward(input, y, var)