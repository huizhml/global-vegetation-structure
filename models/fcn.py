import torch
import torch.nn as nn
from typing import Any, List
import lightning as L
from torchsummary import summary
from .modules.util import ConvNormActivation
from utils import get_class
from ._base_pl_model import BaseModel
from models.modules.util import get_nonveg_mask
from const import LAT_MEAN, LAT_STD, LON_SIN_MEAN, LON_SIN_STD, LON_COS_MEAN, LON_COS_STD, SLOPE_MEAN, SLOPE_STD


class DoubleConvSkip(nn.Module):
    def __init__(self, in_channels, out_channels, activation_layer, norm_layer, pool: bool = False,
                 self_attention: bool = False, **kwargs):
        super(DoubleConvSkip, self).__init__()
        self.conv1 = ConvNormActivation(in_channels, out_channels, 3, activation_layer=activation_layer,
                                        norm_layer=norm_layer, **kwargs)
        self.conv2 = ConvNormActivation(out_channels, out_channels, 3, activation_layer=activation_layer,
                                        norm_layer=norm_layer,**kwargs)
        
        self.skip = nn.Conv2d(out_channels, out_channels, 5, padding=0, groups=out_channels, bias=False)
        self.skip.weight.requires_grad = False
        self.skip.weight.data.fill_(0)
        central_idx = self.skip.kernel_size[0] // 2
        self.skip.weight.data[:, :, central_idx, central_idx] = 1
        self.in_channels = in_channels
        self.relu = activation_layer

    def forward(self, x):
        skip = self.skip(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = x + skip
        x = self.relu(x)
        # x = torch.cat([x, skip], dim=1)
        return x


class ConvSkip(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 skip_kernel_size,
                 activation_layer, norm_layer, pool: bool = False,
                 self_attention: bool = False, **kwargs):
        super(ConvSkip, self).__init__()
        self.conv = ConvNormActivation(in_channels, out_channels, kernel_size, activation_layer=activation_layer,
                                        norm_layer=norm_layer, **kwargs)
        
        self.skip = nn.Conv2d(out_channels, out_channels, skip_kernel_size, padding=0, groups=out_channels, bias=False)
        self.skip.weight.requires_grad = False
        self.skip.weight.data.fill_(0)
        central_idx = self.skip.kernel_size[0] // 2
        self.skip.weight.data[:, :, central_idx, central_idx] = 1
        self.in_channels = in_channels
        self.relu = activation_layer

    def forward(self, x):
        skip = self.skip(x)
        x = self.conv(x)
        x = x + skip
        x = self.relu(x)
        # x = torch.cat([x, skip], dim=1)
        return x

class FCN(BaseModel):
    """Fully Convolutional Network (FCN)."""

    def __init__(self, 
                 entry_block: nn.Module,
                 activation_layer: nn.Module,
                 in_channels: int,
                 out_channels: int,
                 patch_size: int,
                 norm_layer: str,
                 feature_sizes: List[int],
                 long_skip: bool = False,
                  *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        self.long_skip = long_skip
        self.entry_block = entry_block

        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        self.patch_size = patch_size
        self.norm_layer = get_class(norm_layer)
        self.out_channels = out_channels

        self.fcs = nn.ModuleList()

        for i in range(len(feature_sizes)-1):
            # self.fcs.append(ConvNormActivation(channels[i], channels[i+1], 3, activation_layer=activation_layer, norm_layer=self.norm_layer, padding=0))
            # if i > 0:
            #     ich = channels[i] + in_channels
            # else:
            #     ich = channels[i]
            self.fcs.append(DoubleConvSkip(feature_sizes[i], feature_sizes[i+1], activation_layer=activation_layer, norm_layer=self.norm_layer, padding=0))

        self.fcs.append(ConvSkip(feature_sizes[-2], feature_sizes[-1], 3, 3, activation_layer=activation_layer, norm_layer=self.norm_layer, padding=0))
        self.last_conv = ConvNormActivation(feature_sizes[-1], out_channels, 1, activation_layer=activation_layer, norm_layer=self.norm_layer, padding=0)
    
        print("FCN model initialized")
        print(summary(self))

    def forward(self, x):
        x = self.entry_block(x)
        if self.long_skip:
            skip = x
        for fc in self.fcs:
            x = fc(x)
        if self.long_skip:
            x = x + skip
        x = self.last_conv(x)
        return x
    