from typing import List
import torch
import torch.nn as nn
from torch import Tensor
from torchmetrics import Metric
from const import ESA_WC

class RegressionMetrics(Metric):
    def __init__(self, individual_rhs:List[int]=[98, 100], **kwargs):
        super().__init__(**kwargs)
        self.individual_rhs = individual_rhs
        self.add_state("n", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("sse", default=torch.zeros(101), dist_reduce_fx="sum")
        self.add_state("se", default=torch.zeros(101), dist_reduce_fx="sum")
        self.add_state("sae", default=torch.zeros(101), dist_reduce_fx="sum")


    def update(self, residuals: Tensor) -> None:
        self.n += residuals.shape[0]
        self.se += residuals.sum(dim=0) # ME
        self.sae += residuals.abs().sum(dim=0) # MAE
        self.sse += residuals.pow(2).sum(dim=0)  # MSE, RMSE

    def compute(self) -> Tensor:
        mse = self.sse.mean() / self.n
        metrics = dict(
            mse = mse,
            rmse = torch.sqrt(mse),
            me = self.se.mean() / self.n,
            mae = self.sae.mean() / self.n
        )
        for idx in self.individual_rhs:
            metrics.update({
                f'rmse_rh{idx}': torch.sqrt(self.sse[idx] / self.n),
                f'me_rh{idx}': self.se[idx] / self.n,
                f'mae_rh{idx}': self.sae[idx] / self.n
            })
        return metrics



class MaskedLoss(nn.BCELoss):

    def __init__(self, name:str='mae', zero_out:bool=False, slope_th: float = 20, individual_rhs:List[int]=[98, 100]) -> None:
        super().__init__()
        self.name = name
        self.zero_out = zero_out
        self.slope_th = slope_th
        self.individual_rhs = individual_rhs
        self.error_metrics = RegressionMetrics(individual_rhs=individual_rhs)
        self.error_metrics_veg = RegressionMetrics(individual_rhs=individual_rhs)

    def reset(self):
        self.error_metrics.reset()
        self.error_metrics_veg.reset()
    
    @torch.no_grad()
    def get_mask(self, lc, slope):
        zero_cls = torch.tensor([ESA_WC['Built-up'], ESA_WC['Snow and ice'], ESA_WC['Permanent water bodies']], device=lc.device)
        lc = lc[..., 7, 7]
        label_mask = torch.where(torch.isin(lc, zero_cls), 0, 1)
        
        # NOTE: if the central pixel is in the exclude class and the slope is greater than the threshold, the loss mask is 0
        # exclude_cls = torch.tensor([ESA_WC['Grassland'], ESA_WC['Bare / sparse vegetation'], ESA_WC['Moss and lichen']], device=lc.device)
        # loss_mask_wc = torch.where(torch.isin(lc, exclude_cls), 1, 0)
        slope = slope[..., 7, 7]
        loss_mask_slope = torch.where(slope < self.slope_th, 1, 0)
        # loss_mask = loss_mask_wc * loss_mask_slope
        # loss_mask = 1 - loss_mask
        loss_mask_slope = loss_mask_slope.type(torch.bool)
        return label_mask, loss_mask_slope
    
    def forward(self,  rhs_hat, rhs, lc, slope, latlon, sens, shot_number,predict_high_slope:bool=False) -> Tensor:
        label_mask, loss_mask = self.get_mask(lc, slope)
        if self.zero_out:
            rhs = rhs * label_mask.unsqueeze(-1)
        if not predict_high_slope:
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
            
        veg_mask = torch.where((center_lc==50)|(center_lc==70) | (center_lc==80), 0, 1) # built-up, snow and ice, permanent water bodies
        veg_mask = veg_mask.bool()
        residuals = rhs_hat - rhs
        error_metrics = self.error_metrics.update(residuals)
        error_metrics_veg = self.error_metrics_veg.update(residuals[veg_mask])
        error_metrics['loss'] = error_metrics[self.name]
        output = {
            'loss': error_metrics['loss'],
            'mask': loss_mask, # used to mask land cover when predicting RH profiles
            'veg_mask': veg_mask,
            'lc': lc,
            'rhs_hat': rhs_hat,# high slope samples don't contribute to the evaluation metrics
            'rhs': rhs,
            'slope': slope,
            'latlon': latlon,
            'sens': sens,
            'shot_number': shot_number
        }
        return error_metrics, error_metrics_veg, output