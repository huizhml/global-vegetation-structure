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

from const import ESA_WC_s

class ImageWriter(Callback):

    def __init__(self,
                 **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.patches = []


    @torch.no_grad()
    def on_test_batch_end(self, trainer: Trainer, pl_module: LightningModule, outputs, batch: Any, batch_idx: int) -> None:
        self.patches.append(outputs.cpu().numpy())
        return super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)

    @torch.no_grad()
    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        tile = trainer.datamodel.test_dataset.recompose_predictions(self.patches)
        ds = xr.open_zarr(trainer.datamodel.test_fp, group=trainer.datamodel.tile_id)
        coords = ds.s2.isel(time=0).coords
        bands = []
        for i in range(101):
            for p in range(3, 0, -1):
                bands.append(f'rh{i}_q{p}')
        coords = coords.assign(band=bands)
        pred = xr.DataArray(tile, coords=coords, dims=('band', 'y', 'x'))
        pred.rio.to_raster('masked_pred.tif')
