import torch
from torch import Tensor
from typing import Any, List, Union, Optional, Mapping
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks.callback import Callback
import wandb
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import xarray as xr
import rioxarray
import rasterio

from const import ESA_WC_SHORT, VSM_NODATA

RH100_idx = 301
RH98_idx = 295

class ImageWriter(Callback):

    def __init__(self,
                 mask_with_scl: bool = False,
                 predict_full_profile: bool = True,
                 output_format: str = 'cog',
                 **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.mask_with_scl = mask_with_scl
        self.rh_idx = slice(0, 303) if predict_full_profile else (RH100_idx, RH98_idx)
        self.prediction_cache = {}
        self.nodata_value = VSM_NODATA
        self.scl_exclude_labels = np.array([0, 3, 8, 9, 11, 6, self.nodata_value], dtype=np.uint16)
        self.scl_exclude_labels_tensor = None

    def _apply_masks(self, prediction, scl, x_topleft, y_topleft):
        prediction_no_border = prediction[:, self.rh_idx,
                                          self.border:self.patch_size - self.border,
                                          self.border:self.patch_size - self.border
                                          ]
        
        
        location_key = f'{y_topleft}_{x_topleft}'
        if location_key not in self.prediction_cache:
            self.prediction_cache[location_key] = prediction_no_border
            return None
        else: # ready to write
            prediction_no_border = torch.cat([self.prediction_cache[location_key], prediction_no_border], dim=0)
            
            # Prepare masks before applying them to reduce repeated operations
            masks_to_apply = []
            
            # if self.mask_empty:
            #     # pixels where all RGB values equal zero are empty (bands B02, B03, B04)
            #     # note self.image has shape: (height, width, channels)
            #     img = self.store[f'{self.tile_id}/s2'][:, 1:4, y_topleft:y_topleft + self.patch_size_no_border, x_topleft:x_topleft + self.patch_size_no_border]
            #     # Use torch operations directly instead of numpy sum
            #     img_tensor = torch.from_numpy(img).to(prediction_no_border.device)
            #     invalid_mask = torch.sum(img_tensor, dim=1, keepdim=True) == 0
            #     masks_to_apply.append(invalid_mask)
            
            if self.mask_with_scl:
                # mask snow and cloud (medium and high density). In some cases the probability cloud mask might miss some clouds
                scl = scl[:, :, self.border:self.patch_size - self.border,
                            self.border:self.patch_size - self.border]
                # Convert to torch tensor once and use isin equivalent with pre-computed tensor
                scl_tensor = torch.from_numpy(scl).to(prediction_no_border.device)
                if self.scl_exclude_labels_tensor is None or self.scl_exclude_labels_tensor.device != prediction_no_border.device:
                    self.scl_exclude_labels_tensor = torch.tensor(self.scl_exclude_labels, device=prediction_no_border.device)
                scl_mask = torch.isin(scl_tensor, self.scl_exclude_labels_tensor)
                masks_to_apply.append(scl_mask)
            
            # Apply all masks at once using logical_or to combine them
            # if len(masks_to_apply) > 1:
            #     combined_mask = torch.logical_or(*masks_to_apply)
            # else:
            #     combined_mask = masks_to_apply[0]
            # prediction_no_border = torch.where(combined_mask, torch.nan, prediction_no_border)
            
            # aggregate predictions with median
            prediction_no_border, _ = torch.nanmedian(prediction_no_border, dim=0)          
            prediction_no_border.mul_(100).round_()  # In-place operations
            prediction_no_border = torch.nan_to_num(prediction_no_border, nan=VSM_NODATA).to(torch.int16)
            
            # Move to CPU once and do all numpy operations together
            prediction_no_border = prediction_no_border.cpu().numpy()
            return prediction_no_border
    
    @torch.no_grad()
    def on_predict_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        
        self.img_height = trainer.datamodule.pred_dataset.img_height
        self.img_width = trainer.datamodule.pred_dataset.img_width
        self.border = trainer.datamodule.pred_dataset.border
        self.patch_size = trainer.datamodule.pred_dataset.patch_size
        self.patch_size_no_border = trainer.datamodule.pred_dataset.patch_size_no_border
        self.patch_coords_dict = trainer.datamodule.pred_dataset.patch_coords_dict
        print(f'img_height: {self.img_height}, img_width: {self.img_width}, border: {self.border}, patch_size_no_border: {self.patch_size_no_border}')
        self.full_pred = np.full((303, self.img_height, self.img_width), VSM_NODATA, dtype=np.int16)
        


    @torch.no_grad()
    def on_predict_batch_end(self, trainer: Trainer, pl_module: LightningModule, outputs, batch: Any, batch_idx: int) -> None:
        y_topleft, x_topleft = self.patch_coords_dict[batch_idx][1:]
        y_topleft = y_topleft + self.border
        x_topleft = x_topleft + self.border     
        prediction_no_border = self._apply_masks(outputs, batch[1], x_topleft, y_topleft)
        if prediction_no_border is None:
            return
        
        self.full_pred[:,y_topleft:y_topleft + self.patch_size_no_border, x_topleft:x_topleft + self.patch_size_no_border] = prediction_no_border
        del self.prediction_cache[f'{y_topleft}_{x_topleft}']
        print(self.prediction_cache.keys())

