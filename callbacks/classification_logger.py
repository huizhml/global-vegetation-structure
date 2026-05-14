from typing import Union, List
from functools import partial
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks.callback import Callback
import torch
import torchmetrics
import logging
import numpy as np
import wandb
from .utils import check_if_log
from const import ESA_WC_SHORT


logger = logging.getLogger(__name__)
fields = {
    "Actual": "Actual",
    "Predicted": "Predicted",
    "nPredictions": "nPredictions",
}

class ClassificationLogger(Callback):

    def __init__(self, log_val_every: Union[int, List]=10, 
                 special_name:str='') -> None:
        """ 
        Log confusion matrix for val dataset and test dataset, can be initialized multiple times for different tasks/subtasks in the same run
        Currently requires the batch output contains key 'pred', the key for ground truth is given by cmat_y in batch output
        
        Parameters
        ----------
        * output_name: to differentiate different outputs when multiple datasets with different classes presenting, 
            e.g, output_name='tree', then the batch output has key: pred_tree and y_tree
        """
        super().__init__()
        self.class_names = list(ESA_WC_SHORT.values())
        self.log_val_every = log_val_every

        if log_val_every:
            self.on_validation_start = self.init_cmat
            self.on_validation_batch_end = self.update_cmat
            self.on_validation_epoch_end = partial(self.log_cmat_stats, split='val')
        else:
            self.on_validation_epoch_end = super().on_validation_epoch_end
        
        # self.on_test_start = self.init_cmat
        # self.on_test_batch_end = self.update_cmat
        # self.on_test_epoch_end = partial(self.log_cmat_stats, split='test')

    def init_cmat(self, trainer, pl_module):
        self.conf = torchmetrics.ConfusionMatrix(num_classes=len(self.class_names), task='multiclass').to(pl_module.device)
    

    def update_cmat(self, trainer, pl_module, outputs, batch, batch_idx: int, dataloader_idx: int = 0):
        '''update cmat at each test (val) batch end. log F1, etc., for every epoch
        '''
        lc = outputs['lc'].view(-1).long()
        lc_hat = outputs['lc_pred'].view(-1)
        self.conf.update(lc_hat, lc)

    def log_cmat_stats(self, trainer: "Trainer", pl_module: "LightningModule", split='train'):

        current_epoch = pl_module.current_epoch
        log_epoch = check_if_log(current_epoch, self.log_val_every) or split == 'test'
        if log_epoch:
            cmat = self.conf.compute()
            cmat = cmat.float()
            stats = {}
            numel = cmat.sum(1)
            mask = numel > 0
            if mask.sum() == 0:  # nothing to log
                return stats
            tp = torch.diag(cmat)[mask]
            fp = cmat.sum(0)[mask] - tp
            fn = (cmat.sum(1)[mask] - tp)
            # pl_module.log(f'tp.{split}', tp.sum())
            # pl_module.log(f'fp.{split}', fp.sum())
            pl_module.log(f'accuracy/{split}', (tp.sum() / numel.sum()))
            
            # macro statistics
            acc = (tp / numel[mask])
            precision = tp / (tp + fp + torch.finfo(torch.float32).eps)
            recall = tp / (tp + fn + torch.finfo(torch.float32).eps)
            f1 = 2 * ((precision * recall) / (precision + recall + torch.finfo(torch.float32).eps))
            pl_module.log(f'accuracy/macc.{split}', acc.mean())
            pl_module.log(f'precision/{split}', precision.mean())
            pl_module.log(f'recall/{split}', recall.mean())
            pl_module.log(f'f1.{split}', f1.mean())

            # also log metrics per class stats
            for i, class_name in enumerate(np.array(self.class_names)[mask.cpu().numpy()]):
                pl_module.log(f"accuracy/{split}.{class_name}", acc[i])
                pl_module.log(f"tp.{split}.{class_name}", tp[i])
                pl_module.log(f'fp.{split}.{class_name}', fp[i])
                pl_module.log(f"recall/{split}.{class_name}", recall[i])
                pl_module.log(f"precision/{split}.{class_name}", precision[i])
                pl_module.log(f"f1/{split}.{class_name}", f1[i])