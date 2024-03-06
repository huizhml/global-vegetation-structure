import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from kornia.enhance import normalize

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

    @torch.no_grad()  # disable gradients for effiency
    def forward(self, x) -> Tensor:
        return normalize(x, MEAN, STD)