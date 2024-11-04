import torch
import torch.nn as nn
from torch import Tensor

class Loss(nn.BCELoss):

    def __init__(self, name:str='mae') -> None:
        super().__init__()
        self.name = name


    def forward(self, y_hat, y, *args) -> Tensor:
        residuals = y_hat - y
        rmse_rh98 = torch.sqrt(torch.mean(residuals[:, 98]**2))
        me_rh98 = torch.mean(residuals[:, 98])
        mse = torch.mean(residuals**2)
        me = torch.mean(residuals)
        mae = torch.mean(torch.abs(residuals))
        rmse = torch.sqrt(mse)
        losses = {
            "mse": mse,
            "me": me,
            "mae": mae,
            "rmse": rmse,
            "rmse_rh98": rmse_rh98,
            "me_rh98": me_rh98,
            "pred": y_hat,
            "target": y
        }
        losses['loss'] = losses[self.name]
        return losses


