import torch
import torch.nn as nn
from typing import Any, Mapping
import lightning as L

from .modules.unet_blocks import UnetBlockDeep, CatResBlock, PassBlock, get_upscaler
from .modules.util import CustomPixelShuffle_ICNR, icnr_init, get_class

class UNet(L.LightningModule):

    @torch.no_grad()
    def __init__(self, 
                in_channels: int,
                out_channels: int,
                patch_size: int,
                norm_layer_up: str,
                up_block: str,
                activation_layer: nn.Module,
                last_opt: nn.Module,
                encoder: nn.Module,
                loss: nn.Module,
                mask_module: nn.Module,
                upsampling: str = "pixelshuffle",
                blur_final: bool = False,
                blur: bool = False,
                final_skip: bool = True,
                self_attention: bool = False,
                 *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        self.blur = blur
        self.blur_final = blur_final
        self.self_attention = self_attention
        self.final_skip = final_skip
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.norm_layer_up = get_class(norm_layer_up)
        self.upsampling = upsampling
        self.activation_layer = activation_layer
        self.up_block = get_class(up_block)
        self.out_channels = out_channels
        self.last_opt = last_opt
        self.apply_mask = mask_module

        self.encoder = encoder
        self.init_decoder()

        self.loss = loss
        if isinstance(last_opt, nn.Identity):
            self.yhat_trasform = lambda x: x
        else:
            self.yhat_trasform =  lambda x: x.cumsum(dim=1)
        torch.set_float32_matmul_precision('high')

    def init_decoder(self):
        self.encoder.eval()

        # create dummy features to test network integrity
        img = torch.rand(2, self.in_channels, *self.patch_size).detach()
        # iterate through encoder to dynamically measure sizes
        x, outs = self.encoder(img)
        self.scale_factors = []
        ps = img.shape[2]  # assumes same reduction over height and width
        for out in outs:
            self.scale_factors.append(ps / out.shape[2])
            ps = out.shape[2]
        self.scale_factors.append(ps / x.shape[2])

        in_sizes = [out.size(1) for out in outs]

        # invert sizes and outputs
        cross_sizes = in_sizes[::-1]
        outs = outs[::-1]
        self.scale_factors = self.scale_factors[::-1]

        # init up scaling
        up_layers = []
        for i, cross_size in enumerate(cross_sizes):
            final = i == len(cross_sizes) - 1 and not self.final_skip
            up_in_c = x.size(1)
            do_blur = self.blur and (not final or self.blur_final)
            sa = i == 0 and self.self_attention
            unet_block = UnetBlockDeep(up_in_c, cross_size, self.activation_layer, norm_layer=self.norm_layer_up,
                                       blur=do_blur, self_attention=sa, upsampling=self.upsampling, final=final,
                                       scale_factor=self.scale_factors[i], block=self.up_block).eval()
            up_layers.append(unet_block)

            x = unet_block(x, outs[i])

        self.up_layers = nn.ModuleList(up_layers).eval()
        last_dim = x.size(1)

        if self.scale_factors[-1] != 1.0:
            last_dim, self.final_upsampling = get_upscaler(
                last_dim, self.scale_factors[-1], self.activation_layer, self.norm_layer_up, self.blur, self.upsampling
            )
            self.final_upsampling.eval()
            x = self.final_upsampling(x)
        else:
            self.final_upsampling = nn.Sequential()

        # init final output layers
        if self.final_skip:
            # add initial feature maps (not downsampled as last long skip connection)
            self.pre_final_conv = CatResBlock(
                last_dim + self.in_channels, last_dim + self.in_channels,
                self.norm_layer_up, self.activation_layer
            ).eval()
        else:
            self.pre_final_conv = PassBlock()
        x = self.pre_final_conv(x, img)

        self.last_conv = nn.Conv2d(x.size(1), self.out_channels, 1).eval()
        x = self.last_conv(x)
        self.last_opt(x)

    def forward(self, img):
        x = img
        x, outs = self.encoder(x)

        outs = outs[::-1]

        for out, up_layer in zip(outs, self.up_layers):
            x = up_layer(x, out)
        x = self.final_upsampling(x)
        x = self.last_conv(self.pre_final_conv(x, img))
        out = self.last_opt(x)
        return out

    def reset_parameters(self):
        def init_weights(m):
            self.encoder.init_weights(m)
            if isinstance(m, CustomPixelShuffle_ICNR):
                m[0][0].weight.data.copy_(icnr_init(m[0][0].weight.data))

        if self.backbone in ["swin-b", "swin-s"]:  # skip pretrained encoder
            self.up_layers.apply(init_weights)
            self.pre_final_conv.apply(init_weights)
            self.last_conv.apply(init_weights)
        else:  # reset everything
            self.apply(init_weights)

    def reset_head(self):
        self.last_conv.apply(self.encoder.init_weights)

    def training_step(self, sample, batch_idx):
        x, y, mask = self.apply_mask(*sample)
        output = self.forward(x.float())
        output = output[mask][..., 7,7] # drop patches with high slope
        y = y[mask].float()
        y_hat = output[:, :101]
        var = output[:, 101:]
        y_hat = self.yhat_trasform(y_hat) # for outputing delta RHs
        losses = self.loss(y_hat, y, var)
        for name, loss in losses.items():
            self.log(f'train_{name}', loss, on_epoch=True, on_step=False)
        return {'loss': losses['loss'], 'pred': y_hat, 'target': y}

    def validation_step(self, sample, batch_idx):
        x, y, mask = self.apply_mask(*sample)
        output = self.forward(x.float())
        output = output[mask][...,7,7] 
        y = y[mask] # drop patches with high slope
        y_hat = output[:, :101]
        var = output[:, 101:]
        y_hat = self.yhat_trasform(y_hat) # for outputing delta RHs
        losses = self.loss(y_hat, y, var)
        for name, loss in losses.items():
            self.log(f'val_{name}', loss, on_epoch=True, on_step=False)
        
        return {'loss': losses['loss'], 'pred': y_hat, 'target': y}