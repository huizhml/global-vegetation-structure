import dask
import dask.dataframe as dd
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

def make_residual_plot(residuals, save_path, slope_lt20=True):
    fig, ax = plt.subplots(figsize=(15, 3))
    residuals.boxplot(ax=ax, showfliers=False)
    plt.xticks(range(1, 102, 5), np.arange(0, 101, 5))
    plt.xlabel("Relative Height (0-100)", fontsize=14)
    plt.ylabel("Residuals (m)", fontsize=14)
    plt.title(f"Residuals of VSM on GEDI (slope < 20)" if slope_lt20 else "Residuals of VSM on GEDI", fontsize=14)
    ax.grid(False)  # Remove grid
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

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
    ddf = dd.read_parquet(files)
    if slope_lt20:
        ddf = ddf[ddf['slope'] < 20]
    ours_rh_cols = [f'RH{i}_Q1_raw' for i in range(101)]
    gedi_rh_cols = [f'rh{i}' for i in range(101)]
    rename_map = dict(zip(ours_rh_cols, gedi_rh_cols)) # dataframe substract matches the columns, need to rename the columns
    residuals = ddf[ours_rh_cols].rename(columns=rename_map) - ddf[gedi_rh_cols]
    residuals = residuals.compute()
    std_residuals = residuals.std(axis=0)
    make_residual_plot(residuals, save_dir / 'residual_plot.pdf', slope_lt20=slope_lt20)
    residuals = residuals/std_residuals
    make_residual_plot(residuals, save_dir / 'residual_plot_normalized.pdf', slope_lt20=slope_lt20)
    
    
    
if __name__ == '__main__':
    ref_and_ours_dir = '/projects/dereeco/data/gvs/gedi/veg_sensitivity_gt0p95/subset_test/original_with_sota_chms_biome_and_ours_full/2020'
    save_dir = '/projects/dereeco/data/gvs/evaluation/with_gedi'
    evaluate_vsm_on_gedi(ref_and_ours_dir=ref_and_ours_dir,
                         save_dir=save_dir)