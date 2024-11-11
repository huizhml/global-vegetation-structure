import joblib
import torch
import torch.nn as nn
from typing import Any
import lightning as L


class BaseModel(L.LightningModule):

    def __init__(
            self, 
            feed_slope: bool = False,
            feed_latlon: bool = False,
            loss_fc: nn.Module = None, 
            transform: nn.Module = None, 
            yhat_transform: nn.Module = None, *args: Any, **
            kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.feed_slope = feed_slope
        self.feed_latlon = feed_latlon
        self.loss_fc = loss_fc
        self.transform = transform
        self.yhat_transform = yhat_transform

    def training_step(self, sample, batch_idx):
        x = self.transform(sample[0])
        if self.feed_slope:
            x = torch.cat([x, sample[3].unsqueeze(1)], dim=1)
        if self.feed_latlon:
            import ipdb; ipdb.set_trace()
            lat = sample[4]
            sin_lon = torch.sin(sample[4][:,1]/180)
            cos_lon = torch.cos(sample[4][:,1]/180)
            x = torch.cat([[x,lat, sin_lon.unsqueeze(1), cos_lon], sample[4]], dim=1)
        y_hat = self.forward(x.float())

        error_metrics, error_metrics_veg, output = self.loss_fc(y_hat, *sample[1:])
        for name, err in error_metrics.items():
            self.log(f'train_{name}', err, on_epoch=True, on_step=False, sync_dist=True)
        for name, err in error_metrics_veg.items():
            self.log(f'veg/train_{name}', err, on_epoch=True, on_step=False, sync_dist=True)
        return output

    def validation_step(self, sample, batch_idx):
        x = self.transform(sample[0])
        if self.feed_slope:
            x = torch.cat([x, sample[3].unsqueeze(1)], dim=1)
        if self.feed_latlon:
            lat = sample[4]
            sin_lon = torch.sin(sample[4][:,1]/180)
            cos_lon = torch.cos(sample[4][:,1]/180)
            x = torch.cat([[x,lat, sin_lon, cos_lon], sample[4]], dim=1)
        y_hat = self.forward(x.float())

        error_metrics, error_metrics_veg, output = self.loss_fc(y_hat, *sample[1:])
        for name, err in error_metrics.items():
            self.log(f'val_{name}', err, on_epoch=True, on_step=False, sync_dist=True)
        for name, err in error_metrics_veg.items():
            self.log(f'veg/val_{name}', err, on_epoch=True, on_step=False, sync_dist=True)
        return output