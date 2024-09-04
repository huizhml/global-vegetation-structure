import torch
import torch.nn as nn
from torch import Tensor
from models.losses.base import Loss

def hinge_loss(delta_rh: Tensor):
    """
    Penalize negative delta relative heights (rh_{i+1} - rh_i) values.
    Parameters:
    ------------
        * delta_rh (Tensor): rh_i - rh_{i+1}.
    """
    return torch.mean(torch.max(torch.tensor(0), delta_rh)) #? mean or sum?

class HingeAddedLoss(Loss):

    def __init__(self, name:str=None, alpha: float = 1.0):
        super().__init__()
        self.name = name
        self.hinge_loss_fn = hinge_loss
        self.alpha = alpha

    def forward(self, y_hat, y, *args) -> Tensor:
        losses = super().forward(y_hat, y)
        losses['hinge_loss'] = self.hinge_loss_fn(y_hat[:-1]- y_hat[1:])
        losses['loss'] = losses[self.name] + self.alpha*losses['hinge_loss']
        return losses