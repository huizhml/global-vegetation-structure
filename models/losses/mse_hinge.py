import torch
import torch.nn as nn
from torch import Tensor

def hinge_loss(delta_rh: Tensor):
    """
    Penalize negative delta relative heights (rh_{i+1} - rh_i) values.
    Parameters:
    ------------
        * delta_rh (Tensor): rh_i - rh_{i+1}.
    """
    return torch.mean(torch.max(torch.tensor(0), delta_rh)) #? mean or sum?

class MSEHinge(nn.Module):

    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.mse_loss_fn = nn.MSELoss()
        self.hinge_loss_fn = hinge_loss
        self.alpha = alpha

    def forward(self, y_hat, y, *args) -> Tensor:
        mse_loss = self.mse_loss_fn(y_hat, y)
        rmse = torch.sqrt(mse_loss)
        hinge_loss = self.hinge_loss_fn(y_hat[:-1]- y_hat[1:])
        return {
            'loss': mse_loss + self.alpha * hinge_loss,
            'mse_loss': mse_loss, 
            'hinge_loss': self.alpha * hinge_loss,
            'rmse': rmse,
        }