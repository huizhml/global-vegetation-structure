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

class ModelSync(Callback):
    def __init__(self, name:str=None, func: callable=None, xlabel:str=None, ylabel:str=None, **kwargs: Any) -> None:
        self.name = name
        self.func = func
        self.state = []
        self.xlabel = xlabel
        self.ylabel = ylabel
    
    def update(self, pred: Tensor, y: Tensor) -> None:
        v = self.func(pred, y)
        self.state.append(v)

    def grouped_cat(self) -> None:
        boxes = []
        for i in range(len(self.state[0])):
            boxes.append(torch.cat([arr[i] for arr in self.state], dim=0).cpu().numpy())
        return boxes

    def reset(self):
        self.state.clear()

    def plot(self, title:str=None, **kwargs: Any):
        if self.name == 'residuals_rh98':
            v = self.grouped_cat()
            w = len(v)
        else:
            v = torch.cat(self.state, dim=0).cpu().numpy()
            w = v.shape[1]
        fig = plt.figure(figsize=(30, 6))
        plt.boxplot(v, whis=[0, 100])
        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)
        
        plt.xticks(np.arange(1,w+1), np.arange(w))
        plt.title(title)
        return fig


def mae(pred: Tensor, y: Tensor) -> Tensor:
    return (pred - y).abs().mean()

def residuals(pred: Tensor, y: Tensor) -> Tensor:
    return pred - y

def delta_rh(pred: Tensor, y: Tensor) -> Tensor:
    return pred[:, 1:] - pred[:, :-1]

def relative_height(pred: Tensor, y: Tensor) -> Tensor:
    return pred

def residuals_rh98(pred: Tensor, y: Tensor) -> Tensor:
    bins = torch.tensor(np.arange(0,90,10), device=pred.device)
    rh98 = y[:, 98]
    res = pred[:, 98] - rh98
    bin_places = (rh98.unsqueeze(1) >= bins).long().sum(1)
    binned_res = [res[bin_places == i] for i in range(1, len(bins)+1)]
    return binned_res


METRICS = {
    'mae': {
        'func': mae,
    },
    'residuals': {
        'func': residuals,
        'xlabel': 'RH0-RH100',
        'ylabel': 'Residuals (m)'
    },
    'delta_rh': {
        'func': delta_rh,
        'xlabel': 'delta_RH1-delta_RH100',
        'ylabel': 'Delta RH (m)'
    },
    'relative_height': {
        'func': relative_height,
        'xlabel': 'RH0-RH100',
        'ylabel': 'Relative Height (m)'
    },
    'residuals_rh98': {
        'func': residuals_rh98,
        'xlabel': 'Residuals (m)',
        'ylabel': 'RH98 (x10m)'
    }
}

class BoxplotLogger(Callback):

    def __init__(self,
                 log_every: Union[int, List]=None,
                 log_metrics: Optional[List[str]]=['residuals', 'delta_rh', 'relative_height', 'residuals_rh98'],
                 **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.log_every = log_every
        self.cols = [f'RH{i}' for i in range(101)]
        self.metrics = []
        for metric in log_metrics:
            self.metrics.append(Visualizer(name=metric, **METRICS[metric]))

    @torch.no_grad()
    def on_train_batch_end(self, trainer: Trainer, pl_module: LightningModule, outputs, batch: Any, batch_idx: int) -> None:
        current_epoch = trainer.current_epoch
        if check_if_log(current_epoch, self.log_every):
            for metric in self.metrics:
                metric.update(outputs['pred'], outputs['target'])
        return super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)
    
    def on_validation_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        current_epoch = trainer.current_epoch
        if check_if_log(current_epoch, self.log_every):
            for metric in self.metrics:
                fig = metric.plot(f'{metric.name} [train]')
                wandb.log({f'{metric.name}.train': wandb.Image(fig)})
                metric.reset()
        return super().on_train_epoch_end(trainer, pl_module)
    
    @torch.no_grad()
    def on_validation_batch_end(self, trainer: Trainer, pl_module: LightningModule, outputs, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        current_epoch = trainer.current_epoch
        if check_if_log(current_epoch, self.log_every):
            for metric in self.metrics:
                metric.update(outputs['pred'], outputs['target'])
        return super().on_validation_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)
    
    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        current_epoch = trainer.current_epoch
        if check_if_log(current_epoch, self.log_every):
            for metric in self.metrics:
                fig = metric.plot(f'{metric.name} [val]')
                wandb.log({f'{metric.name}.val': wandb.Image(fig)})
                plt.close(fig)
                metric.reset()
        return super().on_validation_epoch_end(trainer, pl_module)

    
