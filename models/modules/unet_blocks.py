import torch
import torch.nn as nn
import torch.nn.functional as F

from .standard import DoubleConvSkip
from .util import CustomPixelShuffle_ICNR, ConvNormActivation

class UnetBlockDeep(nn.Module):
    "A quasi-UNet block, using upsampling."

    def __init__(self, up_in_c: int, x_in_c: int, activation_layer: nn.Module, norm_layer: nn.Module,
                 blur: bool = False, self_attention: bool = False, upsampling="pixelshuffle",
                 final: bool = False, scale_factor=2, block: nn.Module = DoubleConvSkip):
        super().__init__()
        up_in_shuf, upscaler = get_upscaler(up_in_c, scale_factor, activation_layer, norm_layer, blur, upsampling)
        self.upscaler = upscaler

        channel_in = up_in_shuf + x_in_c
        channel_out = channel_in if final else channel_in // 2
        self.block = block(
            channel_in, channel_out, activation_layer=activation_layer,
            norm_layer=norm_layer, self_attention=self_attention
        )
        # TODO This seems to result in double activations for the up_out part
        self.relu = activation_layer

    def forward(self, up_in: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        up_out = self.upscaler(up_in)
        ssh = skip.shape[-2:]
        if ssh != up_out.shape[-2:]:
            up_out = F.interpolate(up_out, skip.shape[-2:], mode='nearest')

        cat_x = self.relu(torch.cat([up_out, skip], dim=1))

        return self.block(cat_x)


def get_upscaler(in_dim, scale_factor, activation_layer, norm_layer, blur: bool, upsampling: str):
    upscaler = []
    out_dim = in_dim
    for i in range(int(scale_factor // 2)):
        if upsampling == "pixelshuffle":
            upscaler.append(
                CustomPixelShuffle_ICNR(
                    out_dim, activation_layer, norm_layer, out_dim // 2, scale=2
                )
            )
        elif upsampling == "deconv":
            upscaler.append(nn.ConvTranspose2d(out_dim, out_dim // 2, 2, 2))
        elif upsampling == "nearest":
            upscaler.append(nn.Sequential(
                nn.Upsample(scale_factor=2, mode=upsampling),
                ConvNormActivation(
                    out_dim, out_dim // 2, 1, activation_layer=activation_layer, norm_layer=norm_layer
                )
            ))
        else:
            upscaler.append(nn.Sequential(
                nn.Upsample(scale_factor=2, mode=upsampling, align_corners=True),
                ConvNormActivation(
                    out_dim, out_dim // 2, 1, activation_layer=activation_layer, norm_layer=norm_layer
                )
            ))
        # Blurring over (h*w) kernel
        # "Super-Resolution using Convolutional Neural Networks without Any Checkerboard Artifacts"
        # - https://arxiv.org/abs/1806.02658
        if blur:
            upscaler.extend([nn.ReplicationPad2d((1, 0, 1, 0)), nn.AvgPool2d(2, stride=1)])
        out_dim //= 2
    upscaler = nn.Sequential(*upscaler)
    return out_dim, upscaler


class ResBlock(nn.Module):
    def __init__(self, ni, nh, norm_layer, activation_layer):
        super().__init__()
        self.convs = nn.Sequential(
            ConvNormActivation(ni, nh, 3, norm_layer=norm_layer, activation_layer=activation_layer),
            ConvNormActivation(nh, ni, 3, norm_layer=norm_layer, activation_layer=activation_layer)
        )

    def forward(self, x): return self.convs(x) + x


class CatResBlock(ResBlock):
    def forward(self, x, prev_x):
        x = torch.cat([x, prev_x], 1)
        return self.convs(x) + x


class CatBlock(nn.Module):
    def __init__(self, block, **kwargs):
        self.block = block(**kwargs)

    def forward(self, x, prev_x):
        x = torch.cat([x, prev_x], 1)
        return self.block(x)


class PassBlock(nn.Module):
    def forward(self, x, *args):
        return x