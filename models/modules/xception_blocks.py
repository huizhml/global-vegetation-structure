
import torch.nn as nn
from typing import List
from download.core.utils import get_class

def conv3x3(in_channels, out_channels, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=True, dilation=dilation)


def conv1x1(in_channels, out_channels, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, bias=True)


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1):
        super(SeparableConv2d, self).__init__()

        self.depthwise = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size,
                                   stride=stride, padding=padding, dilation=dilation, groups=in_channels, bias=False)

        self.pointwise = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1,
                                   padding=0, dilation=1, groups=1, bias=False)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class PointwiseBlock(nn.Module):

    def __init__(self, activation_layer: nn.Module, norm_layer: str=None,  in_channels:int=None, filters:List[int]=None, out_channels:int=None):
        super(PointwiseBlock, self).__init__()
        norm_layer = get_class(norm_layer)
        self.in_channels = in_channels
        self.filters = filters
        self.activation = activation_layer
        filters = [in_channels] + filters
        self.layers = nn.Sequential()
        for i in range(len(filters) - 1):
            self.layers.add_module(f'ConvNormActivation{i}', nn.Sequential(
                conv1x1(filters[i], filters[i + 1]),
                norm_layer(filters[i + 1]),
                activation_layer
            ))
        self.layers.add_module('ConvNorm', nn.Sequential(
            conv1x1(filters[-1], out_channels),
            norm_layer(out_channels)
        ))

        self.shortcut = nn.Sequential()
        if self.in_channels != out_channels:
            self.shortcut.add_module('conv_shortcut', conv1x1(in_channels, out_channels))
            self.shortcut.add_module('bn_shortcut', norm_layer(out_channels))

    def forward(self, x):
        shortcut = self.shortcut(x)
        x = self.layers(x)
        out = x + shortcut

        return out



class DoubleConvBlock(nn.Module):

    def __init__(self, activation_layer:nn.Module,norm_layer:str, in_channels:int=None, filters:List[int]=None, kernel_sizes:List[int]=(3,3), padding:int=0):
        super(DoubleConvBlock, self).__init__()
        norm_layer = get_class(norm_layer)
        self.activation = activation_layer
        self.in_channels = in_channels
        self.filters = filters

        self.sepconv1 = nn.Conv2d(in_channels=in_channels, out_channels=filters[0], kernel_size=kernel_sizes[0], padding=padding, bias=False)
        self.bn1 = norm_layer(filters[0])

        self.sepconv2 = nn.Conv2d(in_channels=filters[0], out_channels=filters[1], kernel_size=kernel_sizes[1], padding=padding, bias=False)
        self.bn2 = norm_layer(filters[1])

        
        self.shortcut = nn.Sequential()
        if self.in_channels != self.filters[-1]:
            self.shortcut.add_module('conv_shortcut', conv1x1(in_channels, filters[1]))
            self.shortcut.add_module('bn_shortcut', norm_layer(filters[1]))

    def forward(self, x):
        shortcut = self.shortcut(x)
        out = self.activation(x)
        out = self.sepconv1(out)
        out = self.bn1(out)

        out = self.activation(out)
        out = self.sepconv2(out)
        out = self.bn2(out)
        out = out + shortcut
        return out


class DoubleSepConvBlock(nn.Module):

    def __init__(self, activation_layer:nn.Module,norm_layer:str, in_channels:int=None, filters:List[int]=None, kernel_sizes:List[int]=(3,3)):
        super(DoubleSepConvBlock, self).__init__()
        norm_layer = get_class(norm_layer)
        self.activation = activation_layer
        self.in_channels = in_channels
        self.filters = filters

        self.sepconv1 = SeparableConv2d(in_channels=in_channels, out_channels=filters[0], kernel_size=kernel_sizes[0])
        self.bn1 = norm_layer(filters[0])

        self.sepconv2 = SeparableConv2d(in_channels=filters[0], out_channels=filters[1], kernel_size=kernel_sizes[1])
        self.bn2 = norm_layer(filters[1])

        
        self.shortcut = nn.Sequential()
        if self.in_channels != self.filters[-1]:
            self.shortcut.add_module('conv_shortcut', conv1x1(in_channels, filters[1]))
            self.shortcut.add_module('bn_shortcut', norm_layer(filters[1]))

    def forward(self, x):
        shortcut = self.shortcut(x)
        out = self.activation(x)
        out = self.sepconv1(out)
        out = self.bn1(out)

        out = self.activation(out)
        out = self.sepconv2(out)
        out = self.bn2(out)
        out = out + shortcut
        return out

class SepNonlinConvBlock(nn.Module):
    def __init__(self, activation_layer:nn.Module,norm_layer:str, in_channels:int=None, filters:List[int]=None, kernel_sizes:List[int]=(3,3)):
        super(SepNonlinConvBlock, self).__init__()
        norm_layer = get_class(norm_layer)
        self.activation = activation_layer
        self.in_channels = in_channels
        self.filters = filters

        self.sepconv1 = SeparableConv2d(in_channels=in_channels, out_channels=filters[0], kernel_size=3)
        self.bn1 = norm_layer(filters[0])

        self.conv1x1 = conv1x1(filters[0], filters[1])
        self.bn2 = norm_layer(filters[1])

        
        self.shortcut = nn.Sequential()
        if self.in_channels != self.filters[-1]:
            self.shortcut.add_module('conv_shortcut', conv1x1(in_channels, filters[1]))
            self.shortcut.add_module('bn_shortcut', norm_layer(filters[1]))

    def forward(self, x):
        shortcut = self.shortcut(x)
        out = self.activation(x)
        out = self.sepconv1(out)
        out = self.bn1(out)

        out = self.activation(out)
        out = self.conv1x1(out)
        out = self.bn2(out)
        out = out + shortcut
        return out


class SepConvBlock(nn.Module):

    def __init__(self, activation_layer: nn.Module, norm_layer:str=None, in_channels:int=None, out_channels:int=None):
        super(SepConvBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.sepconv1 = SeparableConv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3)
        self.bn1 = get_class(norm_layer)(out_channels)

        self.activation = activation_layer
        # self.shortcut = nn.Sequential()
        # if self.in_channels != self.out_channels:
        #     self.shortcut.add_module('conv_shortcut', conv1x1(in_channels, filters))
        #     self.shortcut.add_module('bn_shortcut', norm_layer(filters))
        # self.conv_shortcut = conv1x1(in_channels, filters[1])
        # self.bn_shortcut = norm_layer(filters[1])

    def forward(self, x):
        out = self.activation(x)
        out = self.sepconv1(out)
        out = self.bn1(out)
        # out = out + shortcut

        return out

class SepConvBlockSkip(nn.Module):

    def __init__(self, activation_layer: nn.Module, norm_layer:str=None, in_channels:int=None, out_channels:int=None, nconv:int=1):
        super(SepConvBlockSkip, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.layers = nn.Sequential(
                activation_layer,
                SeparableConv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3),
                get_class(norm_layer)(out_channels)
        )
        if nconv > 1:
            self.layers.add_module('ActConvNorm1x1', nn.Sequential(
                activation_layer,
                conv1x1(in_channels=out_channels, out_channels=out_channels),
                get_class(norm_layer)(out_channels)
            ))

        self.shortcut = nn.Sequential()
        if self.in_channels != self.out_channels:
            self.shortcut.add_module('conv_shortcut', conv1x1(in_channels, out_channels))
            self.shortcut.add_module('bn_shortcut', norm_layer(out_channels))

    def forward(self, x):
        shortcut = self.shortcut(x)
        out = self.layers(x)
        out = out + shortcut
        return out

