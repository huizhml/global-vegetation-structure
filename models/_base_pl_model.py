import joblib
import torch
import torch.nn as nn
from typing import Any
import lightning as L


class BaseModel(L.LightningModule):

    def __init__(
            self, loss_fc: nn.Module = None, transform: nn.Module = None, yhat_transform: nn.Module = None, *args: Any, **
            kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.loss_fc = loss_fc
        self.transform = transform
        self.yhat_transform = yhat_transform

    def training_step(self, sample, batch_idx):
        x = self.transform(sample[0])
        y_hat = self.forward(x.float())

        error_metrics, output = self.loss_fc(y_hat, *sample[1:])
        for name, err in error_metrics.items():
            self.log(f'train_{name}', err, on_epoch=True, on_step=False, sync_dist=True)
        return output

    def validation_step(self, sample, batch_idx):
        x = self.transform(sample[0])
        y_hat = self.forward(x.float())

        error_metrics, output = self.loss_fc(y_hat, *sample[1:])
        for name, err in error_metrics.items():
            self.log(f'val_{name}', err, on_epoch=True, on_step=False, sync_dist=True)
        return output