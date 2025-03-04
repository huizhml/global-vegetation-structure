import numpy as np
import ipdb
import torch
import torch.nn as nn
from torch import Tensor
from kornia.enhance import normalize

from const import ESA_WC

# n = 1.1436e+08
MEAN = np.array([ 587.0255, 1361.4782, 1091.5225,  739.6226, 1772.4885, 2550.3611,
        2870.9454, 2952.2076, 3080.6981, 3086.7147, 2886.7224, 2175.1265 # train*_filtered_v1
    # 630.6893, # 628.1879 , # unfiltered
    #              1359.3749, # 1475.1014 , 
    #              1108.1755, # 1165.6053 , 
    #              747.1477, # 794.6629 , 
    #              1778.6479,  # 1873.0237 , 
    #              2570.6654, # 2563.7012 , 
    #              2872.9004, # 2852.5837 , 
    #              2942.0968, # 2928.7439 , 
    #              3083.4950, # 3045.0754 , 
    #              3091.6063, # 3052.8425 , 
    #              2837.6595, # 3006.8704 , 
    #              2107.9847, # 2330.9114 ,
        ])
STD = np.array([ 529.5716, 1314.4954,  846.1677,  635.8849, 1293.3423, 1078.8700,
        1119.9127, 1133.3539, 1116.3337, 1100.1368, 1651.3810, 1718.6127 # train*_filtered_v1
    # 608.2745, 1320.2338,  875.1261,  689.7431, 1298.4036, 1109.1152,
        # 1152.3742, 1167.4110, 1160.0978, 1145.2506, 1639.9369, 1669.9939 # 1698.3635
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
        wc = wc[..., 7, 7]
        label_mask = torch.where(torch.isin(wc, zero_cls), 0, 1)
        y = y * label_mask.unsqueeze(-1)
        # NOTE: if the central pixel is in the exclude class and the slope is greater than the threshold, the loss mask is 0
        exclude_cls = torch.tensor([ESA_WC['Grassland'], ESA_WC['Bare / sparse vegetation'], ESA_WC['Moss and lichen']], device=x.device)
        loss_mask_wc = torch.where(torch.isin(wc, exclude_cls), 1, 0)
        slope = slope[..., 7, 7]
        loss_mask_slope = torch.where(slope > self.slope_th, 1, 0)
        loss_mask = loss_mask_wc * loss_mask_slope
        loss_mask = 1 - loss_mask
        loss_mask = loss_mask.type(torch.bool)
        return normalize(x.float(), MEAN, STD), y, None, None, None, loss_mask.squeeze()

    

class Normalize(nn.Module):
    def __init__(self):
        super().__init__()

    @torch.no_grad()
    def forward(self, x) -> Tensor:
        return normalize(x.float(), MEAN, STD)