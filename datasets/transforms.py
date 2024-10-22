import numpy as np
import ipdb
import torch
import torch.nn as nn
from torch import Tensor
from kornia.enhance import normalize

from const import ESA_WC

MEAN = np.array([628.1879 , 
                 1475.1014 , 
                 1165.6053 , 
                 794.6629 , 
                 1873.0237 , 
                 2563.7012 , 
                 2852.5837 , 
                 2928.7439 , 
                 3045.0754 , 
                 3052.8425 , 
                 3006.8704 , 
                 2330.9114 ,
        ])
STD = np.array([531.9321,
                1304.8284 ,
                846.0083  ,
                636.7952 ,
                1286.7987 ,
                1097.4255 ,
                1140.2261 ,
                1151.3964 ,
                1139.3654 ,
                1128.2611 ,
                1634.0034 ,
                1698.3635
        # 20.39255142211914,
        # 0.385648638010025,
        # 0.3675101101398468
        ])

    
class SlopeWCMask(nn.Module):
    def __init__(self, slope_th: float = 20):
        super().__init__()
        self.slope_th = slope_th

    @torch.no_grad()
    def forward(self, x, y, wc, slope, latlon, *args) -> Tensor:
        # NOTE: zero out the central pixel if it is in classes: 'Built-up', 'Snow and ice', 'Permanent water bodies'
        zero_cls = torch.tensor([ESA_WC['Built-up'], ESA_WC['Snow and ice'], ESA_WC['Permanent water bodies']], device=x.device)
        # wc = wc[..., 7, 7]
        label_mask = torch.where(torch.isin(wc, zero_cls), 0, 1)
        y = y * label_mask
        # NOTE: if the central pixel is in the exclude class and the slope is greater than the threshold, the loss mask is 0
        exclude_cls = torch.tensor([ESA_WC['Grassland'], ESA_WC['Bare / sparse vegetation'], ESA_WC['Moss and lichen']], device=x.device)
        loss_mask_wc = torch.where(torch.isin(wc, exclude_cls), 1, 0)
        # slope = slope[..., 7, 7]
        loss_mask_slope = torch.where(slope > self.slope_th, 1, 0)
        loss_mask = loss_mask_wc * loss_mask_slope
        loss_mask = 1 - loss_mask
        loss_mask = loss_mask.type(torch.bool)
        return normalize(x.float(), MEAN, STD), y, loss_mask.squeeze()