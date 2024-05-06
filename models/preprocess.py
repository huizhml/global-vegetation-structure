import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from kornia.enhance import normalize

from const import ESA_WC

MEAN = np.array([172.49044799804688,
        217.19879150390625,
        306.56103515625,
        384.7140808105469,
        491.4490051269531,
        662.2481079101562,
        867.2765502929688,
        927.40576171875,
        995.0731811523438,
        959.4855346679688,
        820.0773315429688,
        640.4260864257812,
        # 6.1001691818237305,
        # -0.01115760300308466,
        # 0.08859393745660782
        ])
STD = np.array([439.42572021484375,
        498.8157958984375,
        600.3677978515625,
        750.9793701171875,
        896.781494140625,
        1420.87890625,
        1624.6717529296875,
        1676.152099609375,
        1730.864990234375,
        1711.1083984375,
        1552.2928466796875,
        1207.4013671875,
        # 20.39255142211914,
        # 0.385648638010025,
        # 0.3675101101398468
        ])

class Standardize(nn.Module):
    """Module to perform pre-process using Kornia on torch tensors."""
    def __init__(self, slope_th: float = 20):
        super().__init__()
        self.slope_th =slope_th

    @torch.no_grad()  # disable gradients for effiency
    def forward(self, x, y, wc, slope, device) -> Tensor:
        # NOTE: zero out the central pixel if it is in classes: 'Built-up', 'Snow and ice', 'Permanent water bodies'
        zero_cls = torch.tensor([ESA_WC['Built-up'], ESA_WC['Snow and ice'], ESA_WC['Permanent water bodies']], device=device)
        label_mask = torch.where(torch.isin(wc[..., 7,7], zero_cls), 0, 1)
        y = y * label_mask
        # NOTE: if the central pixel is in the exclude class and the slope is greater than the threshold, the loss mask is 0
        exclude_cls = torch.tensor([ESA_WC['Grassland'], ESA_WC['Bare / sparse vegetation'], ESA_WC['Moss and lichen']], device=device)
        loss_mask_wc = torch.where(torch.isin(wc[..., 7,7], exclude_cls), 1, 0)
        loss_mask_slope = torch.where(slope[:, 7,7:8] > self.slope_th, 1, 0)
        loss_mask = loss_mask_wc * loss_mask_slope
        loss_mask = 1 - loss_mask
        return normalize(x, MEAN, STD), y, loss_mask