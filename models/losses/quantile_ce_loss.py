from typing import List
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from models.losses.base import Loss


class QuantileCELoss(Loss):
    
    def __init__(self, *, quantiles: List[float] = [0.05, 0.5, 0.95]) -> None:
        super().__init__()
        if isinstance(quantiles, list):
            assert all(0 < q < 1 for q in quantiles), "Quantiles should be in (0, 1) range"
        else:
            assert torch.all((0 < quantiles) & (quantiles < 1)), "Quantiles should be in (0, 1) range"
        self.quantiles = quantiles


    def forward(self, y_hat, y, mask, *args) -> Tensor:
        if isinstance(self.quantiles, list):
            quantiles_tensor = torch.tensor(self.quantiles, device=y_hat.device).view(1, -1)
        else:
            quantiles_tensor = self.quantiles.view(1, -1)

        median_idx = self.quantiles.index(0.5)
        lc = y[:,0].long() # the first output channel is the land cover prediction
        lc_hat = y_hat[:,:11] # we have 11 land cover classes
        ce_loss = F.cross_entropy(lc_hat, lc)
        
        acc = (lc_hat.argmax(dim=-1) == lc).float().mean()

        y_hat = y_hat[:,11:].unsqueeze(-1)
        y = y[..., 1:].unsqueeze(-1)
        batch_size = y.size(0)
        feature_size = y.size(1)
        y_hat = y_hat.reshape(batch_size, feature_size, -1)        
        losses = super().forward(y_hat[:,:, median_idx:median_idx+1], y)
        
        residuals = y_hat - y
        # Calculate losses for each quantile
        quantile_losses = torch.max((quantiles_tensor - 1) * residuals, quantiles_tensor * residuals)
        # Sum the losses and take the mean
        quantile_loss = torch.mean(torch.sum(quantile_losses, dim=2))
        losses['loss'] = quantile_loss + ce_loss
        losses['lc_ce'] = ce_loss
        losses['quantile_loss'] = quantile_loss
        losses['lc_acc'] = acc
        losses['pred'] = y_hat
        return losses
