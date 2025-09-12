import torch
from typing import Any, List, Union
from lightning.pytorch.callbacks.callback import Callback
import pandas as pd
from pathlib import Path

class PredictionLogger(Callback):

    def __init__(self,
                 output_rh_idxs: Union[List[int], range]=range(101),
                 outfile_suffix: str='',
                 save_dir: str='output/canopy_height_predictions',
                 **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.output_rh_idxs = output_rh_idxs
        self.outfile_suffix = outfile_suffix
        self.output_cols = [f'RH{i}_{j}' for i in output_rh_idxs for j in range(3)] + [f'RH{j}_GEDI' for j in output_rh_idxs] + ['slope_mask', 'veg_mask']
        self.save_dir = Path(save_dir).expanduser()
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
    def on_test_epoch_start(self, trainer, pl_module):
        self.canopy_heights = []
        
    
    @torch.no_grad()
    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx = 0):
        y_hat = outputs[0]#[:, self.output_rh_idxs]
        y = outputs[1]#[:, self.output_rh_idxs]
        canopy_heights = torch.cat([y_hat, y, *outputs[2:]], dim=1)
        self.canopy_heights.append(canopy_heights.cpu())
        return super().on_test_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)
    

    def on_test_epoch_end(self, trainer, pl_module):
        self.canopy_heights = torch.cat(self.canopy_heights)
        df = pd.DataFrame(self.canopy_heights, columns=self.output_cols)
        df.to_parquet(self.save_dir / f'canopy_height_predictions_{trainer.logger._experiment.id}{self.outfile_suffix}.parquet')
        print(f'saved canopy height predictions to {self.save_dir / f"canopy_height_predictions_{trainer.logger._experiment.id}{self.outfile_suffix}.parquet"}')
        return super().on_test_epoch_end(trainer, pl_module)
    

