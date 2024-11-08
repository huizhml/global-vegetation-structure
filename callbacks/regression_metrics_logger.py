import torch
from torch import Tensor
from typing import Any, List, Union, Optional, Mapping
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks.callback import Callback
import wandb
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from .utils import check_if_log

class AverageMeter:
    def __init__(self, xlabel:str='Relative Height (0-100)', **kwargs: Any) -> None:
        self.xlabel = xlabel
        self.me = 0 # mean error for each rh
        self.mae = 0 # mean absolute error for each rh
        self.rmse = 0 # mean squared error for each rh
        self.pred_interval = 0
        self.n = 0
    
    def update(self, pred: Tensor, y: Tensor) -> None:
        residuals = pred - y
        self.n += len(residuals)
        self.me += residuals.sum(dim=0)
        self.mae += residuals.abs().sum(dim=0)
        self.rmse += residuals.pow(2).sum(dim=0)
        if len(residuals.shape) > 2:
            self.pred_interval += (pred[..., 0] - pred[..., -1]).sum(dim=0)

    def reset(self):
        self.me = 0 # mean error for each rh
        self.mae = 0 # mean absolute error for each rh
        self.rmse = 0 # mean squared error for each rh

        self.n = 0

    def plot(self, name:str=None, title:str=None,**kwargs: Any):
        metric = getattr(self, name)
        metric /= self.n
        if name == 'rmse':
            metric = metric.sqrt()
        fig = plt.figure(figsize=(30, 6))
        plt.plot(metric.cpu().numpy())
        plt.xlabel(self.xlabel)
        plt.ylabel(name.upper())
        plt.xticks(np.arange(len(metric)))
        plt.title(title)
        if len(metric.shape) > 1:
            plt.legend([f'Q{i+1}' for i in range(metric.shape[1])])
        return fig


class ErrorMetricsLogger(Callback):

    def __init__(self,
                 log_every: Union[int, List]=None,
                 **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.log_every = log_every
        self.avgmeter = AverageMeter()

    @torch.no_grad()
    def on_train_batch_end(self, trainer: Trainer, pl_module: LightningModule, outputs, batch: Any, batch_idx: int) -> None:
        current_epoch = trainer.current_epoch
        if check_if_log(current_epoch, self.log_every):
            self.avgmeter.update(outputs['rhs_hat'], outputs['rhs'])
        return super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)
    
    def on_validation_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        current_epoch = trainer.current_epoch
        if check_if_log(current_epoch, self.log_every):
            for metric in ['me', 'mae', 'rmse', 'pred_interval']:
                fig = self.avgmeter.plot(metric, f'{metric.upper()} [train]')
                wandb.log({f'{metric} [train]': wandb.Image(fig)})
                plt.close(fig)
            self.avgmeter.reset()
        return super().on_train_epoch_end(trainer, pl_module)
    
    @torch.no_grad()
    def on_validation_batch_end(self, trainer: Trainer, pl_module: LightningModule, outputs, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        current_epoch = trainer.current_epoch
        if check_if_log(current_epoch, self.log_every):
            self.avgmeter.update(outputs['rhs_hat'], outputs['rhs'])
        return super().on_validation_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)
    
    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        current_epoch = trainer.current_epoch
        if check_if_log(current_epoch, self.log_every):
            for metric in ['me', 'mae', 'rmse', 'pred_interval']:
                fig = self.avgmeter.plot(metric, f'{metric.upper()} [val]')
                wandb.log({f'{metric} [val]': wandb.Image(fig)})
                plt.close(fig)
            self.avgmeter.reset()
        return super().on_validation_epoch_end(trainer, pl_module)

    
