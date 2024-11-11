import torch
from torch import Tensor
from typing import Any, List, Union, Optional, Mapping
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks.callback import Callback
import wandb
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from const import ESA_WC_s
from .utils import check_if_log



def plot_prediction(pred: Tensor, target: Tensor, xlabel:str='Relative Height (0-100)', **kwargs: Any):
    fig = plt.figure(figsize=(12, 6))
    plt.plot(pred.cpu().numpy())
    plt.scatter(target.cpu().numpy(), label='target')
    plt.xticks(np.arange(101, 10))
    plt.xlabel(xlabel)
    plt.ylabel('RH')
    if pred.shape[1] > 1:
        plt.legend([f'Q{i+1}' for i in range(pred.shape[1])] + ['target'])
    else:
        plt.legend(['pred', 'target'])
    return fig

SHOT_NUMBERS_TRAIN = [
    # wc 10
    201400200300352775, 
    # wc 20
    98870300300351318, 
    # wc 30
    65781100200204486, 
    # wc 40
    165320500300301582, 
    # wc 50
    76760000200246936, 
    # wc 60
    88020800100038631, 
    # wc 70
    91200000100032552, 
    # wc 80
    135990300200106060, 
    # wc 90
    44160000400290624, 
    # wc 95
    207121100100113911, 
    # wc 100
    127420600200503637, 
]

SHOT_NUMBERS_VAL = [
    # wc 10
    44350500300290822, 
    # wc 20
    31100200100114853,
    # wc 30
    49620000400452946,
    # wc 40
    22170200300146877, 
    # wc 50
    136650500100095174,
    # wc 60
    50960500300181616, 
    # wc 70
    145771100200196539,
    # wc 80
    165870800200125841,
    # wc 90
    133020500200055572,
    # wc 95
    199640600400585013, 
    # wc 100
    214911100300236035,
]


class PredictionLogger(Callback):

    def __init__(self,
                 log_batch:int=0,
                 log_every: Union[int, List]=None,
                 show_n_examples:int=10,
                 shot_numbers_train:List[int]=SHOT_NUMBERS_TRAIN,
                 shot_numbers_val:List[int]=SHOT_NUMBERS_VAL,
                 **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.log_batch = log_batch
        self.log_every = log_every
        self.show_n_examples = show_n_examples
        self.shot_numbers_train = shot_numbers_train
        self.shot_numbers_val = shot_numbers_val
        self.finished = []


    @torch.no_grad()
    def on_train_batch_end(self, trainer: Trainer, pl_module: LightningModule, outputs, batch: Any, batch_idx: int) -> None:
        current_epoch = trainer.current_epoch
        if len(self.finished) == len(self.shot_numbers_train): # all logged, skip the rest batches in this epoch
            return super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)
        shot_numbers = [v for v in self.shot_numbers_train if v in outputs['shot_number']]
        self.finished += shot_numbers
        if check_if_log(current_epoch, self.log_every) and len(shot_numbers) > 0:
            # fig = plot_prediction(outputs['pred'], outputs['target'])
            # wandb.log({f'Prediction [train]': wandb.Image(fig)})
            # plt.close(fig)
            lc = outputs['lc'][outputs['mask'], 7,7].cpu().numpy()
            if outputs.get('lc_pred') is not None:
                lc_pred = outputs['lc_pred'][outputs['mask'], 7,7].cpu().numpy()
            for i in shot_numbers:
                idx = torch.where(outputs['shot_number']==i)[0]
                pred = torch.cat([outputs['rhs_hat'][idx], outputs['rhs'][idx]], dim=2).cpu().numpy()
                n_features = outputs['rhs_hat'].shape[1]
                n_predictions = outputs['rhs_hat'].shape[2]
                slope = outputs['slope'][idx,7,7].cpu().numpy().round(2)
                if outputs.get('lc_pred') is not None:
                    lc_pred_name = ESA_WC_s[lc_pred[idx].item()].replace('/', '_')
                else:
                    lc_pred_name = None
                lc_name = ESA_WC_s[lc[idx].item()].replace('/', '_')
                title = f'[train] epoch{current_epoch}/{lc_name}-pred-{lc_pred_name}_slope{round(slope.item(), 2)} sample{i}'
                line = wandb.plot.line_series(
                            xs=range(n_features),
                            ys=pred[0].T,
                            keys=[f'Q{k}' for k in range(n_predictions)] + ['target'],
                            xname='Relative Height (0-100)',
                            title=title)
                wandb.log({f'prediction/{title}': line})
        return super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)

    @torch.no_grad()
    def on_train_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.finished = []
        return super().on_train_epoch_end(trainer, pl_module)
    
    @torch.no_grad()
    def on_validation_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.finished = []
        return super().on_validation_epoch_end(trainer, pl_module)
    

    @torch.no_grad()
    def on_validation_batch_end(self, trainer: Trainer, pl_module: LightningModule, outputs, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        current_epoch = trainer.current_epoch
        if len(self.finished) == len(self.shot_numbers_val): # all logged, skip the rest batches in this epoch
            return super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)
        shot_numbers = [v for v in self.shot_numbers_val if v in outputs['shot_number']]
        self.finished += shot_numbers
        if check_if_log(current_epoch, self.log_every) and len(shot_numbers) > 0:
            # fig = plot_prediction(outputs['pred'], outputs['target'])
            # wandb.log({f'Prediction [train]': wandb.Image(fig)})
            # plt.close(fig)
            lc = outputs['lc'][outputs['mask'], 7,7].cpu().numpy()
            if outputs.get('lc_pred') is not None:
                lc_pred = outputs['lc_pred'][outputs['mask'], 7,7].cpu().numpy()
            for i in shot_numbers:
                idx = torch.where(outputs['shot_number']==i)[0]
                pred = torch.cat([outputs['rhs_hat'][idx], outputs['rhs'][idx]], dim=2).cpu().numpy()
                n_features = outputs['rhs_hat'].shape[1]
                n_predictions = outputs['rhs_hat'].shape[2]
                slope = outputs['slope'][idx,7,7].cpu().numpy().round(2)
                if outputs.get('lc_pred') is not None:
                    lc_pred_name = ESA_WC_s[lc_pred[idx].item()].replace('/', '_')
                else:
                    lc_pred_name = None
                lc_name = ESA_WC_s[lc[idx].item()].replace('/', '_')
                title = f'[val] epoch{current_epoch}/{lc_name}-pred-{lc_pred_name}_slope{round(slope.item(), 2)} sample{i}'
                line = wandb.plot.line_series(
                            xs=range(n_features),
                            ys=pred[0].T,
                            keys=[f'Q{k}' for k in range(n_predictions)] + ['target'],
                            xname='Relative Height (0-100)',
                            title=title)
                wandb.log({f'prediction/{title}': line})
        return super().on_validation_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)

    
