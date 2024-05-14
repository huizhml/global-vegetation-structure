import torch
import torch.nn as nn
from torch import Tensor

class MSELoss(nn.MSELoss):

    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super().__init__(size_average, reduce, reduction)


    def forward(self, y_hat, y) -> Tensor:
        mse = super().forward(y_hat, y)
        rmse = torch.sqrt(mse)
        return {
            'loss': mse,
            'rmse': rmse,
        }