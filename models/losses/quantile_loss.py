from typing import List, Any
import torch
import torch.nn as nn
from torch import Tensor
from models.losses.base import MaskedLoss


class QuantileLoss(MaskedLoss):

    def __init__(self, zero_out:bool=None, quantiles: List[float] = [0.05, 0.5, 0.95], *args: Any, **kwargs: Any) -> None:
        super().__init__(zero_out=zero_out, *args, **kwargs)
        if isinstance(quantiles, list):
            assert all(0 < q < 1 for q in quantiles), "Quantiles should be in (0, 1) range"
        else:
            assert torch.all((0 < quantiles) & (quantiles < 1)), "Quantiles should be in (0, 1) range"
        self.quantiles = quantiles


    def forward(self,  rhs_hat, rhs, lc, slope, latlon, sens, shot_number, training:bool=True) -> Tensor:
        if isinstance(self.quantiles, list):
            quantiles_tensor = torch.tensor(self.quantiles, device=rhs_hat.device).view(1, -1)
        else:
            quantiles_tensor = self.quantiles.view(1, -1)

        median_idx = self.quantiles.index(0.5)
        label_mask, loss_mask = self.get_mask(lc, slope)
        lc = lc//10 # high slope doesn't change the land cover, so not need to mask lc
        lc[lc==0.95] = 11
        lc = lc.long()

        if self.zero_out:
            rhs = rhs * label_mask.unsqueeze(-1)
        if training:
            rhs_hat = rhs_hat[loss_mask][..., 7, 7].unsqueeze(-1)
            rhs = rhs[loss_mask].float().unsqueeze(-1)
            center_lc = lc[loss_mask,..., 7, 7]
            slope = slope[loss_mask]
            latlon = latlon[loss_mask]
            sens = sens[loss_mask]
            shot_number = shot_number[loss_mask]
        else:
            rhs_hat = rhs_hat[..., 7, 7].unsqueeze(-1)
            rhs = rhs.float().unsqueeze(-1)
            center_lc = lc[..., 7, 7]
        veg_mask = torch.where((center_lc==5)|(center_lc==7) | (center_lc==8), 0, 1) # built-up, snow and ice, permanent water bodies
        veg_mask = veg_mask.bool()

        n, feature_size, _ = rhs.shape # NOTE: n <= batch_size
        rhs_hat = rhs_hat.reshape(n, feature_size, -1) # n, 101, 3
        residuals = rhs_hat - rhs
        error_metrics = self.error_metrics(residuals[..., median_idx])
        error_metrics_veg = self.error_metrics(residuals[veg_mask,..., median_idx])
        quantile_losses = torch.max((quantiles_tensor - 1) * residuals, quantiles_tensor * residuals)
        quantile_loss = torch.mean(torch.sum(quantile_losses, dim=2))
        error_metrics['loss'] = quantile_loss
        output = {
            'loss': quantile_loss,
            'mask': loss_mask,
            'veg_mask': veg_mask,
            'lc': lc,
            'rhs_hat': rhs_hat,
            'rhs': rhs,
            'slope': slope,
            'latlon': latlon,
            'sens': sens,
            'shot_number': shot_number # for indexing samples to visualize, no need to mask
        }
        return error_metrics, error_metrics_veg, output