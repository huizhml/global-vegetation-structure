from typing import Dict, Tuple
import torch
import torch.nn as nn
from torch import Tensor


class MaskedBaseLoss(nn.Module):
    """Base class for masked loss functions."""

    def __init__(self, loss_fn: nn.Module, **kwargs) -> None:
        super().__init__()
        self.loss_fn = loss_fn(**kwargs)

    def forward(
        self,
        rhs_hat: Tensor,
        slope_mask: Tensor,
        rhs: Tensor,
        lc: Tensor,
        slope: Tensor,
        latlon: Tensor,
        sens: Tensor,
        shot_number: Tensor,
        predict_high_slope: bool = False,
    ) -> Tuple[Dict[str, Tensor], Dict[str, Tensor], Dict[str, Tensor]]:
        # Common preprocessing logic
        rhs_hat = rhs_hat[:, :, 7, 7].unsqueeze(-1)
        rhs = rhs.float().unsqueeze(-1)

        # Compute the loss using the provided loss function
        loss = self.loss_fn(rhs_hat[slope_mask], rhs[slope_mask])

        # Return losses, predictions, and targets
        losses = {"loss": loss}
        pred = {"rhs_hat": rhs_hat}
        target = {"rhs": rhs}
        return losses, pred, target


class MaskedL1Loss(MaskedBaseLoss):
    def __init__(self, **kwargs) -> None:
        super().__init__(loss_fn=nn.L1Loss, **kwargs)


class MaskedL2Loss(MaskedBaseLoss):
    def __init__(self, **kwargs) -> None:
        super().__init__(loss_fn=nn.MSELoss, **kwargs)


class MaskedHuberLoss(MaskedBaseLoss):
    def __init__(self, **kwargs) -> None:
        super().__init__(loss_fn=nn.HuberLoss, **kwargs)