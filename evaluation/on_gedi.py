import re
from typing import List
import dask
import dask.dataframe as dd
from dask.diagnostics import ProgressBar
from dask.distributed import Client
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
def compute_bxp_stats(ddf_residuals):
    # Fuse quantile + std into a single parallel traversal of the data.
    q_lazy = ddf_residuals.quantile([0.25, 0.5, 0.75])
    std_lazy = ddf_residuals.std(axis=0)
    q, std = dask.compute(q_lazy, std_lazy)
    stats = []
    for col in ddf_residuals.columns:
        q1 = q[col].loc[0.25]
        med = q[col].loc[0.5]
        q3 = q[col].loc[0.75]
        iqr = q3 - q1
        stats.append(dict(med=med, q1=q1, q3=q3,
                          whislo=q1 - 1.5 * iqr, whishi=q3 + 1.5 * iqr,
                          fliers=[]))
    return stats, std


def make_residual_plot(stats, save_path):
    fig, ax = plt.subplots(figsize=(15, 3))
    ax.axhline(0, color='red', linewidth=1)
    ax.bxp(stats, showfliers=False, positions=range(1, 102))
    plt.xticks(range(1, 102, 5), np.arange(0, 101, 5))
    plt.xlabel("Relative Height (0-100)", fontsize=FONT_SIZES['label'])
    plt.ylabel("Residuals (m)", fontsize=FONT_SIZES['label'])
    plt.title(f"Residuals of VSM on GEDI", fontsize=FONT_SIZES['title'])
    ax.grid(False)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# -------------------------------------------------------------
#  Runable
# -------------------------------------------------------------

def evaluate_vsm_on_gedi(
        ref_dir: str = None,
        ours_dir: str = None,
        save_dir: str = None,
        align_cols: List[str] = ('geometry', 'shot_number'),
        slope_lt20: bool = True,
        **kwargs):
    '''
    Evaluate the VSM performance on GEDI
    '''
    ref_dir = Path(ref_dir).expanduser()
    ours_dir = Path(ours_dir).expanduser()


    files = list(ref_and_ours_dir.glob('*.parquet'))#[:10]
    ours_rh_cols = [f'RH{i}_Q1' for i in range(101)]
    gedi_rh_cols = [f'rh{i}' for i in range(101)]
    # Processes-based local cluster: numeric reductions over 101 columns hit
    # the GIL hard under the default threaded scheduler.
    with Client() as client:
        print(client.dashboard_link)
        # Read ONLY the columns the residuals need. dgp.read_parquet would
        # decode WKB geometry into shapely objects per row (millions, unused)
        # and load every column. With one 2.8 GB file, split_row_groups gives
        # parallel partitions. Pushing the slope cut into `filters` prunes row
        # groups via parquet stats AND row-filters internally, so 'slope'
        # never lands in the output frame.
        filters = [('slope', '<', 20)] if slope_lt20 else None
        ddf = dd.read_parquet(files, columns=ours_rh_cols + gedi_rh_cols,
                              split_row_groups=True, filters=filters,
                              dataset={"partitioning": None})
        print(f'npartitions: {ddf.npartitions}')
        rename_map = dict(zip(ours_rh_cols, gedi_rh_cols)) # dataframe substract matches the columns, need to rename the columns
        residuals = ddf[ours_rh_cols].rename(columns=rename_map) - ddf[gedi_rh_cols]
        # One parallel pass yields quantiles + std; the normalized plot is
        # then derived arithmetically, so the full residuals frame never has
        # to land in memory.
        stats, std_residuals = compute_bxp_stats(residuals)
    suffix = '_slope_lt20' if slope_lt20 else ''
    
    save_dir = Path(f'{save_dir}').expanduser()
    save_dir.mkdir(parents=True, exist_ok=True)
    make_residual_plot(stats, save_dir / f'residual_plot{suffix}.pdf')
    norm_stats = [
        {'med': s['med'] / std_residuals.iloc[i],
         'q1': s['q1'] / std_residuals.iloc[i],
         'q3': s['q3'] / std_residuals.iloc[i],
         'whislo': s['whislo'] / std_residuals.iloc[i],
         'whishi': s['whishi'] / std_residuals.iloc[i],
         'fliers': []}
        for i, s in enumerate(stats)
    ]
    make_residual_plot(norm_stats, save_dir / f'residual_plot_normalized{suffix}.pdf')
    
    
    
if __name__ == '__main__':
    ref_and_ours_dir = '/projects/dereeco/data/gvs/gedi/veg_sensitivity_gt0p95/subset_test/original_with_sota_chms_biome_and_ours_full/2020'
    save_dir = '/projects/dereeco/data/gvs/evaluation/with_gedi'
    evaluate_vsm_on_gedi(ref_and_ours_dir=ref_and_ours_dir,
                         save_dir=save_dir)