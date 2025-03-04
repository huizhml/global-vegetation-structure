from torch import Tensor
from lightning.pytorch.callbacks import Callback
from lightning.pytorch import Trainer, LightningModule


def conformal_score(pred: Tensor, target: Tensor) -> Tensor:
    """
    Compute the conformal score of the prediction.
    pred: (B, 303)
    target: (B, 101)
    """
    pred = pred.reshape(-1, 101, 3)
    pred_lower = pred[:, :, 0]
    pred_upper = pred[:, :, 2]
    scores = max(pred_lower - target, target - pred_upper)
    return scores
    

    # compute the conformal score

class UncertaintyLogger(Callback):
    def __init__(self,
                 alpha: float = 0.05,
                 ) -> None:
        '''
        alpha: 1- alpha is the coverage of the prediction interval
        '''
        super().__init__()
        self.alpha = alpha
        self.val_size = 0
        self.val_scores = []
    
    
    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        pass


    def on_validation_batch_end(self, trainer: Trainer, pl_module: LightningModule, outputs, batch, batch_idx, dataloader_idx = 0) -> None:
        self.val_size += outputs['rhs_hat'].shape[0]
        self.val_scores.append(conformal_score(outputs['rhs_hat'], batch['rhs']))

    def on_validation_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.val_size = 0
        self.val_scores = []
    
    