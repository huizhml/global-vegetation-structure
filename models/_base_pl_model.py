import joblib
import torch
import torch.nn as nn
from typing import Any
import lightning as L


class BaseModel(L.LightningModule):

    def __init__(self, loss_fc:nn.Module=None, mask_fc:nn.Module=None, yhat_transform:nn.Module=None, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.loss_fc = loss_fc
        self.mask_fc = mask_fc
        self.yhat_transform = yhat_transform
        

    def training_step(self, sample, batch_idx):
        x, y, mask = self.mask_fc(*sample)
        output = self.forward(x.float())
        output = output[mask][..., 7,7] # drop patches with high slope
        y = y[mask].float().unsqueeze(-1)
        batch_size = y.size(0)
        feature_size = y.size(1)
        y_hat = output.reshape(batch_size, feature_size, -1)
        # y_hat = self.yhat_transform(output) # for outputing delta RHs
        losses = self.loss_fc(y_hat, y)
        for name, loss in losses.items():
            self.log(f'train_{name}', loss, on_epoch=True, on_step=False, sync_dist=True)
        return {'loss': losses['loss'], 'pred': y_hat, 'target': y, 'wc': sample[2][mask], 'slope': sample[3][mask], 'sens': sample[5], 'shot_number': sample[-1][mask]}

    def validation_step(self, sample, batch_idx):
        x, y, mask = self.mask_fc(*sample)
        output = self.forward(x.float())
        output = output[mask][...,7,7] 
        y = y[mask].float().unsqueeze(-1) # drop patches with high slope
        batch_size = y.size(0)
        feature_size = y.size(1)
        y_hat = output.reshape(batch_size, feature_size, -1)
        losses = self.loss_fc(y_hat, y)
        for name, loss in losses.items():
            self.log(f'val_{name}', loss, on_epoch=True, on_step=False, sync_dist=True)
        
        return {'loss': losses['loss'], 'pred': y_hat, 'target': y, 'wc': sample[2][mask], 'slope': sample[3][mask], 'sens': sample[5][mask], 'shot_number': sample[-1][mask]}