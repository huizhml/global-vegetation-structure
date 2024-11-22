from typing import List
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from models.losses.base import MaskedLoss


class QuantileCELoss(MaskedLoss):
    
    def __init__(self, *, zero_out:bool=None, quantiles: List[float] = [0.05, 0.5, 0.95]) -> None:
        super().__init__(zero_out=zero_out)
        if isinstance(quantiles, list):
            assert all(0 < q < 1 for q in quantiles), "Quantiles should be in (0, 1) range"
        else:
            assert torch.all((0 < quantiles) & (quantiles < 1)), "Quantiles should be in (0, 1) range"
        self.quantiles = quantiles


    def forward(self,  y_hat, rhs, lc, slope, latlon, sens, shot_number, training:bool=True) -> Tensor:
        if isinstance(self.quantiles, list):
            quantiles_tensor = torch.tensor(self.quantiles, device=y_hat.device).view(1, -1)
        else:
            quantiles_tensor = self.quantiles.view(1, -1)

        median_idx = self.quantiles.index(0.5)
        label_mask, loss_mask = self.get_mask(lc, slope)
        if self.zero_out:
            rhs = rhs * label_mask.unsqueeze(-1)
        lc_hat = y_hat[:,:12] # we have 11 land cover classes + unknown
        lc = lc//10
        lc[lc==0.95] = 11
        lc = lc.long()
        if training:
            rhs_hat = y_hat[loss_mask, 12:, 7, 7] # (n, 303)
            rhs = rhs[loss_mask].unsqueeze(-1)
            center_lc = lc[loss_mask,..., 7, 7]
            slope = slope[loss_mask]
            latlon = latlon[loss_mask]
            sens = sens[loss_mask]
            shot_number = shot_number[loss_mask]
        else: # validation, calculate loss for high slope points as well
            rhs_hat = y_hat[:, 12:, 7, 7]
            rhs = rhs.unsqueeze(-1)
            center_lc = lc[..., 7, 7]

        veg_mask = torch.where((center_lc==5)|(center_lc==7) | (center_lc==8), 0, 1) # built-up, snow and ice, permanent water bodies
        veg_mask = veg_mask.bool()
        
        ce_loss = F.cross_entropy(lc_hat, lc)
        lc_pred = lc_hat.argmax(dim=1)
        acc = (lc_pred == lc).float().mean()

        n, feature_size, _ = rhs.shape
        rhs_hat = rhs_hat.reshape(n, feature_size, -1)
        residuals = rhs_hat - rhs
        error_metrics = self.error_metrics(residuals[..., median_idx])
        error_metrics_veg = self.error_metrics(residuals[veg_mask,..., median_idx])
        # Calculate losses for each quantile
        quantile_losses = torch.max((quantiles_tensor - 1) * residuals, quantiles_tensor * residuals)
        # Sum the losses and take the mean
        quantile_loss = torch.mean(torch.sum(quantile_losses, dim=2))
        error_metrics['loss'] = quantile_loss + ce_loss
        error_metrics['lc_ce'] = ce_loss
        error_metrics['quantile_loss'] = quantile_loss
        error_metrics['lc_acc'] = acc
        output = {
            'loss': error_metrics['loss'],
            'mask': loss_mask,
            'veg_mask': veg_mask,
            'lc_pred': lc_pred,
            'lc': lc,
            'rhs_hat': rhs_hat,
            'rhs': rhs,
            'slope': slope,
            'latlon': latlon,
            'sens': sens,
            'shot_number': shot_number
        }
        return error_metrics, error_metrics_veg, output

if __name__ == "__main__":
    # Test QuantileCELoss
    y_hat = torch.rand(10, 303, 15, 15)
    rhs = torch.rand(10, 101, 15, 15)
    lc = torch.randint(0, 12, (10, 15, 15))
    slope = torch.rand(10, 15, 15)
    latlon = torch.rand(10, 2)
    sens = torch.rand(10, 1)
    shot_number = torch.randint(0, 100, (10,))
    loss = QuantileCELoss()
    import ipdb; ipdb.set_trace()
    error_metrics, error_metrics_veg, output = loss(y_hat, rhs, lc, slope, latlon, sens, shot_number)
    print(error_metrics)
    print(error_metrics_veg)
    print(output)