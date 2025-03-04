from typing import List
import torch.nn as nn
from models.modules.base import BaseModule
from models.modules.util import ConvNormActivation
from utils import get_class

__all__ = ['StandardNet', 'standard']


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, activation_layer, norm_layer, kernal_sizes:List[int]=(3,3), pool: bool = False,
                 self_attention: bool = False):
        super(DoubleConv, self).__init__()
        self.pool = nn.MaxPool2d(2, ceil_mode=True) if pool else nn.Sequential()
        self.conv1 = ConvNormActivation(in_channels, out_channels, kernal_sizes[0], activation_layer=activation_layer,
                                        norm_layer=norm_layer)
        self.conv2 = ConvNormActivation(out_channels, out_channels, kernal_sizes[1], activation_layer=activation_layer,
                                        norm_layer=norm_layer, 
                                        self_attention=self_attention)

    def forward(self, x):
        x = self.pool(x)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class DoubleConvSkip(DoubleConv):
    def __init__(self, in_channels, out_channels, activation_layer, norm_layer, pool: bool = False,
                 self_attention: bool = False, **kwargs):
        super(DoubleConvSkip, self).__init__(
            in_channels, out_channels, activation_layer, norm_layer, pool=pool, self_attention=self_attention, **kwargs
        )
        self.skip = ConvNormActivation(in_channels, out_channels, 1) if in_channels != out_channels else nn.Sequential()

    def forward(self, x):
        x = self.pool(x)
        skip = self.skip(x)
        x = self.conv1(x)
        x = self.conv2(x)
        return x + skip


class StandardNet(BaseModule):
    def __init__(self, in_channels: int, activation_layer: nn.Module, norm_layer: str,
                 block: str = 'DoubleConvSkip', initial_stride: int = 1, depths=(64, 128, 256, 512, 1024),
                 scale_factor: int=1,
                 kernal_sizes:List[int]=(3,3)):
        super(StandardNet, self).__init__()
        assert initial_stride in [1, 2], "standard model only supports initial_stride of 1 or 2"
        self.in_channels = in_channels
        self.activation_layer = activation_layer
        block = get_class(block)
        norm_layer = get_class(norm_layer)
        # depths = [in_channels, 64, 128, 256, 512, 1024]
        depths = [d * scale_factor for d in depths]
        layers = [block(in_channels, depths[0], activation_layer, norm_layer, pool=initial_stride==2, kernal_sizes=kernal_sizes)]
        for i in range(len(depths) - 1):
            layers.append(block(depths[i], depths[i + 1], activation_layer, norm_layer, pool=True, kernal_sizes=kernal_sizes))
        self.blocks = nn.ModuleList(layers)


def standard(**kwargs):
    return StandardNet(**kwargs)
