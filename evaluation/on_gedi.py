import re
import dask
import dask.dataframe as dd
from dask.diagnostics import ProgressBar
import pandas as pd
import geopandas as gpd
import dask_geopandas as dgp
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from const import FONT_SIZES, set_plot_fonts

set_plot_fonts()

# -------------------------------------------------------------
#  Plot functions
# -------------------------------------------------------------
def make_residual_plot(residuals, save_path):
    fig, ax = plt.subplots(figsize=(15, 3))
    ax.axhline(0, color='red', linewidth=1)
    residuals.boxplot(ax=ax, showfliers=False)
    plt.xticks(range(1, 102, 5), np.arange(0, 101, 5))
    plt.xlabel("Relative Height (0-100)", fontsize=FONT_SIZES['label'])
    plt.ylabel("Residuals (m)", fontsize=FONT_SIZES['label'])
    plt.title(f"Residuals of VSM on GEDI", fontsize=FONT_SIZES['title'])
    ax.grid(False)  # Remove grid
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# -------------------------------------------------------------
#  Runable
# -------------------------------------------------------------

def evaluate_vsm_on_gedi(
        ref_and_ours_dir: str = None,
        save_dir: str = None,
        slope_lt20: bool = True,
        **kwargs):
    '''
    Evaluate the VSM performance on GEDI
    '''
    ref_and_ours_dir = Path(f'{ref_and_ours_dir}').expanduser()
    save_dir = Path(f'{save_dir}').expanduser()
    save_dir.mkdir(parents=True, exist_ok=True)
    files = list(ref_and_ours_dir.glob('*.parquet'))#[:10]
    ours_rh_cols = [f'RH{i}_Q1' for i in range(101)]
    gedi_rh_cols = [f'rh{i}' for i in range(101)]
    # Read ONLY the columns the residuals need. dgp.read_parquet would decode
    # the WKB geometry into a shapely object per row (millions, unused here)
    # and load every column; plain dd + column projection avoids both. With
    # one 2.8 GB file, split_row_groups gives parallel partitions for compute.
    needed_cols = ours_rh_cols + gedi_rh_cols + (['slope'] if slope_lt20 else [])
    ddf = dd.read_parquet(files, columns=needed_cols, split_row_groups=True,
                          dataset={"partitioning": None})
    if slope_lt20:
        ddf = ddf[ddf['slope'] < 20]
    rename_map = dict(zip(ours_rh_cols, gedi_rh_cols)) # dataframe substract matches the columns, need to rename the columns
    residuals = ddf[ours_rh_cols].rename(columns=rename_map) - ddf[gedi_rh_cols]
    residuals = residuals.compute()
    std_residuals = residuals.std(axis=0)
    suffix = '_slope_lt20' if slope_lt20 else ''
    make_residual_plot(residuals, save_dir / f'residual_plot{suffix}.pdf')
    residuals = residuals/std_residuals
    make_residual_plot(residuals, save_dir / f'residual_plot_normalized{suffix}.pdf')
    
    
    
if __name__ == '__main__':
    ref_and_ours_dir = '/projects/dereeco/data/gvs/gedi/veg_sensitivity_gt0p95/subset_test/original_with_sota_chms_biome_and_ours_full/2020'
    save_dir = '/projects/dereeco/data/gvs/evaluation/with_gedi'
    evaluate_vsm_on_gedi(ref_and_ours_dir=ref_and_ours_dir,
                         save_dir=save_dir)