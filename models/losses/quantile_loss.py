from typing import List
import torch
import torch.nn as nn
from torch import Tensor

def multi_quantile_loss(preds, target, quantiles):

    # Convert quantiles to a tensor if it's a list
    if isinstance(quantiles, list):
        quantiles_tensor = torch.tensor(quantiles, device=preds.device).view(1, -1)
    else:
        quantiles_tensor = quantiles.view(1, -1)

    # Calculate errors
    errors = preds - target

    # Calculate losses for each quantile
    losses = torch.max((quantiles_tensor - 1) * errors, quantiles_tensor * errors)

    # Sum the losses and take the mean
    loss = torch.mean(torch.sum(losses, dim=1))

    return loss


class GNLLoss(nn.Module):

    def __init__(self, *, quantiles: List[float] = False):
        super().__init__()
        if isinstance(quantiles, list):
            assert all(0 < q < 1 for q in quantiles), "Quantiles should be in (0, 1) range"
        else:
            assert torch.all((0 < quantiles) & (quantiles < 1)), "Quantiles should be in (0, 1) range"
        self.quantiles = quantiles

    def forward(self, y_hat, y) -> Tensor:
        loss = multi_quantile_loss(y_hat, y, self.quantiles)
        var = torch.exp(var)
        rmse = torch.sqrt(torch.mean((y_hat-y)**2))
        loss = super().forward(y_hat, y, var)
        return {
            'loss': loss,
            'rmse': rmse,
        }