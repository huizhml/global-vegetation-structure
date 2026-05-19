import torch
from torch import Tensor
from typing import Any, List, Union, Optional, Mapping
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks.callback import Callback
import wandb
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from .utils import check_if_log
from const import BIOMES_BY_VALUE, FONT_SIZES, set_plot_fonts

set_plot_fonts()

    
def boxplot_from_stats(name, table):
    boxplot_data = {}
    for col in table.columns:
        boxplot_data[col] = {
            'whislo': table.loc['10%', col],
            'whishi': table.loc['90%', col],
            'q1': table.loc['25%', col],
            'q3': table.loc['75%', col],
            'med': table.loc['50%', col]
        }
    fig, ax = plt.subplots()
    for idx, (column, stats) in enumerate(boxplot_data.items()):
        ax.bxp(
            [stats],
            positions=[idx + 1],
            widths=0.5,
            showfliers=False  # Exclude outliers
        )
    ax.set_xticklabels(boxplot_data.keys())
    ax.set_xlabel("GEDI reference RH98 (m)" if '-' in table.columns[0] else "BIOME")
    ax.set_ylabel(f"{name}(m)")
    plt.tight_layout()
    return fig

class BoxplotLogger(Callback):

    def __init__(self,
                 log_every: Union[int, List]=1,
                 **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.log_every = log_every
        self.rh_idx = [0, 25, 50, 75, 98, 100]
        self.intervals = [float('-inf')]+ np.arange(0, 55, 5).tolist() + [float('inf')]
        self.labels = [f"{self.intervals[i]}-{self.intervals[i+1]}" for i in range(len(self.intervals)-1)]
        self.cols = [f'Residuals RH{i}' for i in self.rh_idx]

    
    def on_validation_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.val_samples_biome = pd.read_parquet(trainer.datamodule.val_fp.with_suffix('.parquet'), columns=['BIOME', 'lat', 'lon'])
        # self.val_samples_biome = self.val_samples_biome.drop_duplicates(subset=['shot_number', 'rh98'])
        self.residuals = []
        self.avg_residuals = []
        self.rh98 = []
        self.lat = []
        self.lon = []
    
    @torch.no_grad()
    def on_validation_batch_end(self, trainer: Trainer, pl_module: LightningModule, outputs, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        current_epoch = trainer.current_epoch
        if check_if_log(current_epoch, self.log_every):
            mask = (outputs['veg_mask'] & outputs['slope_mask']).bool()
            rhs_hat = outputs['rhs_hat'][mask, :, 1] if len(outputs['rhs_hat'].shape) > 2 else outputs['rhs_hat'][mask, :]
            residuals =  rhs_hat - outputs['rhs'][mask]
            avg_residuals = residuals.mean(dim=1)
            
            self.residuals.append(residuals[:, self.rh_idx])
            self.avg_residuals.append(avg_residuals)
            self.rh98.append(outputs['rhs'][mask, 98])
            self.lat.append(outputs['lat'][mask]) 
            self.lon.append(outputs['lon'][mask])
        return super().on_validation_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)
    
    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        current_epoch = trainer.current_epoch
        if check_if_log(current_epoch, self.log_every):
            residuals = torch.cat(self.residuals, dim=0).cpu().numpy()
            rh98 = torch.cat(self.rh98, dim=0).cpu().numpy()
            avg_residuals = torch.cat(self.avg_residuals, dim=0).cpu().numpy()
            lat = torch.cat(self.lat, dim=0).cpu().numpy()
            lon = torch.cat(self.lon, dim=0).cpu().numpy()
            df = pd.DataFrame(residuals.squeeze(), columns=self.cols)
            df['rh98'] = rh98
            df['Residuals RH_all'] = avg_residuals
            df['lat'] = lat
            df['lon'] = lon
            df = pd.merge(df, self.val_samples_biome, on=['lat', 'lon'], how='left')
            df['interval'] = pd.cut(df['rh98'], bins=self.intervals, include_lowest=True, labels=self.labels)
            stats = df.groupby('interval').describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9])
            stats_biome = df.groupby('BIOME').describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9])
            for rh in self.rh_idx + ['_all', 'rh98']:
                if rh == 'rh98':
                    name = 'rh98'
                else:
                    name = f'Residuals RH{rh}'
                table = stats[name].T
                table_biome = stats_biome[name].T
                table_biome = table_biome.drop(columns=[98,99])
                table_biome.columns = [BIOMES_BY_VALUE[col]['abbr'] for col in table_biome.columns]
                
                wandb.log({f'Table RH98_intervals/{name}': wandb.Table(dataframe=table.reset_index())})
                wandb.log({f'Table BIOME/{name}': wandb.Table(dataframe=table_biome.reset_index())})
                fig = boxplot_from_stats(name, table)
                wandb.log({f'Residuals by RH98 intervals/{name}': wandb.Image(fig)})
                plt.close(fig)
                fig = boxplot_from_stats(name, table_biome)
                plt.xticks(rotation=45, ha='right', fontsize=FONT_SIZES['ticks'])
                wandb.log({f'Residuals by biome/{name}_biome': wandb.Image(fig)})
                plt.close(fig)


        return super().on_validation_epoch_end(trainer, pl_module)

