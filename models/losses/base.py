import torch
import torch.nn as nn
from torch import Tensor
from const import ESA_WC

class MaskedLoss(nn.BCELoss):

    def __init__(self, name:str='mae', zero_out:bool=False, slope_th: float = 20) -> None:
        super().__init__()
        self.name = name
        self.zero_out = zero_out
        self.slope_th = slope_th

    @torch.no_grad()
    def get_mask(self, lc, slope):
        zero_cls = torch.tensor([ESA_WC['Built-up'], ESA_WC['Snow and ice'], ESA_WC['Permanent water bodies']], device=lc.device)
        lc = lc[..., 7, 7]
        label_mask = torch.where(torch.isin(lc, zero_cls), 0, 1)
        
        # NOTE: if the central pixel is in the exclude class and the slope is greater than the threshold, the loss mask is 0
        exclude_cls = torch.tensor([ESA_WC['Grassland'], ESA_WC['Bare / sparse vegetation'], ESA_WC['Moss and lichen']], device=lc.device)
        loss_mask_wc = torch.where(torch.isin(lc, exclude_cls), 1, 0)
        slope = slope[..., 7, 7]
        loss_mask_slope = torch.where(slope > self.slope_th, 1, 0)
        loss_mask = loss_mask_wc * loss_mask_slope
        loss_mask = 1 - loss_mask
        loss_mask = loss_mask.type(torch.bool)
        return label_mask, loss_mask
    
    def error_metrics(self, residuals):
        res_square = residuals**2
        rmse_rh98 = torch.sqrt(torch.mean(res_square[:, 98]))
        me_rh98 = torch.mean(residuals[:, 98])
        rmse_rh100 = torch.sqrt(torch.mean(res_square[:, 100]))
        me_rh100 = torch.mean(residuals[:, 100])
        mse = torch.mean(res_square)
        me = torch.mean(residuals)
        mae = torch.mean(torch.abs(residuals))
        rmse = torch.sqrt(mse)
        return {
            "mse": mse,
            "me": me,
            "mae": mae,
            "rmse": rmse,
            "rmse_rh98": rmse_rh98,
            "me_rh98": me_rh98,
            "rmse_rh100": rmse_rh100,
            "me_rh100": me_rh100,
        }
    
    def forward(self,  rhs_hat, rhs, lc, slope, latlon, sens, shot_number) -> Tensor:
        label_mask, loss_mask = self.get_mask(lc, slope)
        if self.zero_out:
            rhs = rhs * label_mask.unsqueeze(-1)
        rhs_hat = rhs_hat[loss_mask][..., 7, 7].unsqueeze(-1)
        rhs = rhs[loss_mask].float().unsqueeze(-1)
        center_lc = lc[loss_mask,..., 7, 7]
        veg_mask = torch.where((center_lc==50)|(center_lc==70) | (center_lc==80), 0, 1) # built-up, snow and ice, permanent water bodies
        veg_mask = veg_mask.bool()
        residuals = rhs_hat - rhs
        error_metrics = self.error_metrics(residuals)
        error_metrics_veg = self.error_metrics(residuals[veg_mask])
        error_metrics['loss'] = error_metrics[self.name]
        output = {
            'loss': error_metrics['loss'],
            'mask': loss_mask, # used to mask land cover when predicting RH profiles
            'veg_mask': veg_mask,
            'lc': lc,
            'rhs_hat': rhs_hat,# high slope samples don't contribute to the evaluation metrics
            'rhs': rhs,
            'slope': slope[loss_mask],
            'latlon': latlon[loss_mask],
            'sens': sens[loss_mask],
            'shot_number': shot_number[loss_mask]
        }
        return error_metrics, error_metrics_veg, output