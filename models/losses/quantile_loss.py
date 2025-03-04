from typing import List, Any
import torch
import torch.nn as nn
from torch import Tensor
import ipdb

track_digits = 5
# Total number of digits (fixed at 17)
total_digits = 17
# Calculate the divisor to zero out the last (total_digits - digits_to_keep) digits
divisor = 10 ** (total_digits - track_digits)
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
    
class QuantileLoss(nn.Module):

    def __init__(self, quantiles: List[float] = [0.05, 0.5, 0.95], weight_factor:float=None, radius:int=None, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if isinstance(quantiles, list):
            assert all(0 < q < 1 for q in quantiles), "Quantiles should be in (0, 1) range"
        else:
            assert torch.all((0 < quantiles) & (quantiles < 1)), "Quantiles should be in (0, 1) range"
        self.quantiles = quantiles
        self.weight_factor = weight_factor
        
        if radius is not None:
            self.slice = slice(7-radius, 8+radius)
            self.radius = radius*2+1
        else:
            self.slice = slice(7,8)
            self.radius = 1


    def forward(self,  rhs_hat, slope_mask, rhs, lc, slope, lat, lon, predict_high_slope:bool=False) -> Tensor:
        if isinstance(self.quantiles, list):
            quantiles_tensor = torch.tensor(self.quantiles, device=rhs_hat.device).view(1, -1)
        else:
            quantiles_tensor = self.quantiles.view(1, -1)

        median_idx = self.quantiles.index(0.5)
        w = rhs_hat.shape[2]
        center_idx = w // 2
        rhs_hat = rhs_hat[:, :, center_idx, center_idx] # (n, 303)
        rhs = rhs.float().unsqueeze(-1)

        n, feature_size, = rhs.shape[:2] # NOTE: n <= batch_size
        rhs_hat = rhs_hat.reshape(n, feature_size, -1)# n, 101, 3 # self.radius*self.radius
        residuals = (rhs_hat - rhs)
        quantile_loss = torch.max((quantiles_tensor - 1) * residuals, quantiles_tensor * residuals)
        quantile_loss = torch.mean(torch.sum(quantile_loss, dim=2), dim=1) # n
        # quantile_loss = quantile_loss.min(dim=1)
        # idx = quantile_loss.indices
        # tracks = (shot_number[slope_mask] // divisor)*divisor // 1e12
        # unique_tracks = torch.unique(tracks)
        # idx_masked = idx[slope_mask]
        # import ipdb; ipdb.set_trace()
        # shift_std = [idx_masked[tracks == track].std() for track in unique_tracks]
        # shift_std = torch.stack(shift_std).mean()
        # rhs_hat = torch.gather(rhs_hat, -1, idx[:, None, None, None].expand(-1, feature_size, quantiles_tensor.shape[1], 1)).squeeze(-1)
        # quantile_loss = quantile_loss.values
        if self.weight_factor is not None:
            for val in rh98_freq:
                mask = (rhs[:, 98] >= val['left']) & (rhs[:, 98] < val['right'])
                mask = mask.squeeze()
                quantile_loss[mask] *= self.weight_factor / val['freq']
        losses = {
            'loss': quantile_loss[slope_mask].mean()
        }
        pred = {
            'rhs_hat': rhs_hat[..., median_idx]
        }
        target = {
            'rhs': rhs
        }
        # output = {
        #     'loss': quantile_loss,
        #     'mask': loss_mask,
        #     'veg_mask': veg_mask,
        #     'lc': lc,
        #     'rhs_hat': rhs_hat,
        #     'rhs': rhs,
        #     'slope': slope,
        #     'latlon': latlon,
        #     'sens': sens,
        #     'shot_number': shot_number # for indexing samples to visualize, no need to mask
        # }
        return losses, pred, target