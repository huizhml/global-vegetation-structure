import torch
from typing import Any, List, Union
from lightning.pytorch.callbacks.callback import Callback
import pandas as pd


class PredictionLogger(Callback):

    def __init__(self,
                 output_rh_idxs: List[int]=None,
                 **kwargs: Any) -> None:
        super().__init__(**kwargs)
        
        self.output_cols = [f'RH{i}' for i in output_rh_idxs] + ['RH95_GEDI', 'RH98_GEDI', 'RH100_GEDI', 'slope_mask', 'veg_mask']
    
    def on_test_epoch_start(self, trainer, pl_module):
        self.canopy_heights = []
        self.outfile_suffix = '_corrected' if pl_module.correct_bias else ''
    
    @torch.no_grad()
    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx = 0):
        y_hat = outputs[0][:, self.output_rh_idxs]
        canopy_heights = torch.cat([y_hat, outputs[1:]], dim=1)
        self.canopy_heights.append(canopy_heights.cpu())
        return super().on_test_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)
    

    def on_test_epoch_end(self, trainer, pl_module):
        self.canopy_heights = torch.cat(self.canopy_heights)
        df = pd.DataFrame(self.canopy_heights, columns=self.output_cols)
        df.to_parquet(f'output/canopy_height_predictions_{trainer.logger._experiment.id}{self.outfile_suffix}.parquet')
        print(f'saved canopy height predictions to output/canopy_height_predictions_{trainer.logger._experiment.id}{self.outfile_suffix}.parquet')
        return super().on_test_epoch_end(trainer, pl_module)
    

