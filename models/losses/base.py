import torch
import torch.nn as nn
from torch import Tensor

class Loss(nn.Module):

    def __init__(self, name:str='mae') -> None:
        self.name = name


    def forward(self, y_hat, y, *args) -> Tensor:
        residuals = y_hat - y
        mse = torch.mean(residuals**2)
        me = torch.mean(residuals)
        mae = torch.mean(torch.abs(residuals))
        rmse = torch.sqrt(mse)
        losses = {
            "mse": mse,
            "me": me,
            "mae": mae,
            "rmse": rmse,
        }
        losses['loss'] = losses[self.name]
        return losses


