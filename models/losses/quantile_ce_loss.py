from typing import List, Any
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

rh98_freq = [
    {'left': -torch.inf, 'right': 0, 'freq': 1},
    {'left': 0, 'right': 5, 'freq': 0.513}, 
    {'left': 5, 'right': 10, 'freq': 0.138}, 
    {'left': 10, 'right': 15, 'freq': 0.086}, 
    {'left': 15, 'right': 20, 'freq': 0.077}, 
    {'left': 20, 'right': 25, 'freq': 0.069}, 
    {'left': 25, 'right': 30, 'freq': 0.054}, 
    {'left': 30, 'right': 35, 'freq': 0.034}, 
    {'left': 35, 'right': 40, 'freq': 0.017}, 
    {'left': 40, 'right': 45, 'freq': 0.008}, 
    {'left': 45, 'right': 50, 'freq': 0.003},
    {'left': 50, 'right': torch.inf, 'freq': 1}
]
   

class QuantileCELoss(nn.Module):
    
    def __init__(self, quantiles: List[float] = [0.05, 0.5, 0.95], weight_factor:float=None, radius:int=None,  *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if isinstance(quantiles, list):
            assert all(0 < q < 1 for q in quantiles), "Quantiles should be in (0, 1) range"
        else:
            assert torch.all((0 < quantiles) & (quantiles < 1)), "Quantiles should be in (0, 1) range"
        self.quantiles = quantiles
        self.weight_factor = weight_factor

        # if radius is not None:
        #     self.slice = slice(7-radius, 8+radius)
        #     self.radius = radius*2+1
        # else:
        #     self.slice = slice(7,8)
        #     self.radius = 1


    def forward(self,  y_hat, slope_mask, rhs, lc, slope, lat, lon, predict_high_slope:bool=False) -> Tensor:
        if isinstance(self.quantiles, list):
            quantiles_tensor = torch.tensor(self.quantiles, device=y_hat.device).view(1, -1)
        else:
            quantiles_tensor = self.quantiles.view(1, -1)

        median_idx = self.quantiles.index(0.5)
        w = y_hat.shape[2]
        center_idx = w // 2
        rhs_hat = y_hat[:, :303, center_idx, center_idx] # (n, 303)

        lc_hat = y_hat[:,303:] # we have 11 land cover classes + unknown
        lc = lc//10
        lc[lc==0.95] = 11
        lc = lc.long()
        ce_loss = F.cross_entropy(lc_hat, lc)

        rhs = rhs.float().unsqueeze(-1)
        n, feature_size = rhs.shape[:2]
        rhs_hat = rhs_hat.reshape(n, feature_size, -1)
        residuals = (rhs_hat - rhs)
        # Calculate losses for each quantile
        quantile_loss = torch.max((quantiles_tensor - 1) * residuals, quantiles_tensor * residuals)
        quantile_loss = torch.mean(torch.sum(quantile_loss, dim=2), dim=(1))
        # rhs_hat = torch.gather(rhs_hat, -1, idx[:, None, None, None].expand(-1, feature_size, quantiles_tensor.shape[1], 1)).squeeze(-1)
        # quantile_loss = quantile_loss.values
        if self.weight_factor is not None:
            for val in rh98_freq:
                mask = (rhs[:, 98] >= val['left']) & (rhs[:, 98] < val['right'])
                mask = mask.squeeze()
                quantile_loss[mask] *= self.weight_factor / val['freq']
        quantile_loss = quantile_loss[slope_mask].mean()
        losses = {
            'loss': quantile_loss+ ce_loss,
            'ce_loss': ce_loss,
            'quantile_loss': quantile_loss
        }
        pred = {
            'rhs_hat': rhs_hat[..., median_idx],
            'lc_hat': lc_hat
        }
        target = {
            'rhs': rhs,
            'lc': lc
        }
        # output = {
        #     'loss': error_metrics['loss'],
        #     'mask': loss_mask,
        #     'veg_mask': veg_mask,
        #     'lc_pred': lc_pred,
        #     'lc': lc,
        #     'rhs_hat': rhs_hat,
        #     'rhs': rhs,
        #     'slope': slope,
        #     'latlon': latlon,
        #     'sens': sens,
        #     'shot_number': shot_number
        # }
        return losses, pred, target

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