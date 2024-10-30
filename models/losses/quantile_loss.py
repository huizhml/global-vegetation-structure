from typing import List
import torch
import torch.nn as nn
from torch import Tensor
from models.losses.base import Loss


class QuantileLoss(Loss):

    def __init__(self, *, quantiles: List[float] = [0.05, 0.5, 0.95]) -> None:
        super().__init__()
        if isinstance(quantiles, list):
            assert all(0 < q < 1 for q in quantiles), "Quantiles should be in (0, 1) range"
        else:
            assert torch.all((0 < quantiles) & (quantiles < 1)), "Quantiles should be in (0, 1) range"
        self.quantiles = quantiles


    def forward(self, y_hat, y, *args) -> Tensor:
        if isinstance(self.quantiles, list):
            quantiles_tensor = torch.tensor(self.quantiles, device=y_hat.device).view(1, -1)
        else:
            quantiles_tensor = self.quantiles.view(1, -1)

        median_idx = self.quantiles.index(0.5)
        losses = super().forward(y_hat[:,:, median_idx:median_idx+1], y)
        
        residuals = y_hat - y
        # Calculate losses for each quantile
        quantile_losses = torch.max((quantiles_tensor - 1) * residuals, quantiles_tensor * residuals)
        # Sum the losses and take the mean
        quantile_loss = torch.mean(torch.sum(quantile_losses, dim=2))
        losses['loss'] = quantile_loss
        return losses