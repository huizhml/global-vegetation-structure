import torch
import torchmetrics
from torchmetrics.regression import MeanAbsoluteError, MeanSquaredError, MeanAbsolutePercentageError
from torchmetrics.utilities.checks import _check_same_shape

class MAE(MeanAbsoluteError):

    def __init__(self,  threshold:int=None, **kwargs):
        super().__init__(num_outputs=101, **kwargs)
        self.th = threshold
        
    def update(self, preds, target):
        if self.th is not None:
            mask = target[:, 98] > self.th
            return super().update(preds[mask], target[mask])
        return super().update(preds, target)
    
    def compute(self):
        errors = super().compute()
        return {
            'FullProfile': errors.mean(),
            'RH98': errors[98],
            'RH100': errors[100]
        }

class RMSE(MeanSquaredError):
    def __init__(self, squared = True, num_outputs = 101, threshold:int=None, **kwargs):
        super().__init__(squared, num_outputs, **kwargs)
        self.th = threshold
    
    def update(self, preds, target):
        if self.th is not None:
            mask = target[:, 98] > self.th
            return super().update(preds[mask], target[mask])
        return super().update(preds, target)

    def compute(self):
        mse = super().compute()
        return {
            'FullProfile': torch.sqrt(mse.mean()),
            'RH98': torch.sqrt(mse[98]),
            'RH100': torch.sqrt(mse[100])
        }
    
class ME(torchmetrics.MeanMetric):
    def __init__(self, num_outputs: int = 101, threshold:int=None, **kwargs):
        super().__init__(**kwargs)
        self.th = threshold

        self.add_state("sum_per_error", default=torch.zeros(num_outputs), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor, epsilon: float = 1.17e-06) -> None:
        """Update and returns variables required to compute Mean Percentage Error.

        Check for same shape of input tensors.

        Args:
            preds: Predicted tensor
            target: Ground truth tensor
            epsilon: Specifies the lower bound for target values. Any target value below epsilon
                is set to epsilon (avoids ``ZeroDivisionError``).

        """
        if self.th is not None:
            mask = target[:, 98] > self.th
            preds = preds[mask]
            target = target[mask]
        _check_same_shape(preds, target)

        diff = preds - target
        sum_per_error = torch.sum(diff, dim=0)
        num_obs = target.shape[0]

        self.sum_per_error += sum_per_error
        self.total += num_obs

    def compute(self) -> torch.Tensor:
        errors = self.sum_per_error / self.total
        return {
            'FullProfile': errors.mean(),
            'RH98': errors[98],
            'RH100': errors[100]
        }


class MAPE(MeanAbsolutePercentageError):
    def __init__(self, num_outputs: int = 101, **kwargs):
        super().__init__(**kwargs)

        self.add_state("sum_abs_per_error", default=torch.zeros(num_outputs), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor, epsilon: float = 1.17e-06) -> None:
        """Update and returns variables required to compute Mean Percentage Error.

        Check for same shape of input tensors.

        Args:
            preds: Predicted tensor
            target: Ground truth tensor
            epsilon: Specifies the lower bound for target values. Any target value below epsilon
                is set to epsilon (avoids ``ZeroDivisionError``).

        """
        _check_same_shape(preds, target)

        abs_diff = torch.abs(preds - target)
        abs_per_error = abs_diff / torch.clamp(torch.abs(target), min=epsilon)

        sum_abs_per_error = torch.sum(abs_per_error, dim=0)

        num_obs = target.shape[0]

        self.sum_abs_per_error += sum_abs_per_error
        self.total += num_obs

    def compute(self) -> torch.Tensor:
        errors = super().compute()
        return {
            'MAPE': errors.mean(),
            'MAPE_RH98': errors[98],
            # 'MAPE_RH100': errors[100]
        }