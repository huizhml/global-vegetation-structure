"""
Compute per-pixel Foliage Height Diversity (Shannon entropy of RH profile)
from 101-band GEDI-like rasters.

Fully vectorized — no for loops, no JAX, no numba.
Uses VRT + rasterio windowed reads for fast I/O and
ProcessPoolExecutor for true parallel CPU compute.

Dependencies: numpy, rasterio, gdal, scipy, psutil
Optional: stackstac, pystac, xarray, rioxarray (for STAC-based workflow)
"""

from osgeo import gdal
import glob
import rasterio
from rasterio.windows import Window
import numpy as np
from pathlib import Path
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import time
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from evaluation.utils import ProgressMonitor, batch_binning
import dask
from dask.diagnostics import ProgressBar
import dask.dataframe as dd
import re
from sklearn.metrics import r2_score
import seaborn as sns
from matplotlib.colors import LogNorm
import colorsys
from const import BIOMES
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from scipy.stats import gaussian_kde, pearsonr

MAX_HEIGHT = 150.0 # NOTE: GEDI L2B FHD was derived from [0, 150] m height range, with 5 m step size
NODATA_IN = 32767
NODATA_OUT = -9999.0
GEDI_META_COLS =['digital_elevation_model', 'digital_elevation_model_srtm', 'pft_class', 'sensitivity', 'beam',
                 'elevation_bias_flag', 'energy_total', 'landsat_treecover', 'landsat_water_persistence', 
                 'leaf_off_doy', 'leaf_off_flag', 'leaf_on_doy', 'leaf_on_cycle', 'modis_nonvegetated', 
                 'modis_nonvegetated_sd', 'modis_treecover', 'modis_treecover_sd', 'num_detectedmodes',
                 'solar_azimuth', 'solar_elevation', 'surface_flag', 'urban_focal_window_size', 'urban_proportion', 'slope', 'lc', 
                 'geometry',
                 'BIOME', 'ECO_NAME']
key_rhs = [25, 50, 75, 95, 98]
RH_COLS = [f'rh{rh}' for rh in key_rhs] + [f'RH{rh}_Q1_raw' for rh in key_rhs]
stac_collection_dir = '~/data/gvs/products/gvsm_stac_catalog/vsm_local'

# ------------------------------------------------------------------------------------------------
# Plotting functions
# ------------------------------------------------------------------------------------------------
def scatter_plot(df: pd.DataFrame, var: str, metrics: dict, biome_value: int = None, save_dir: Path = None, max_value: int = 3) -> None:
    if biome_value == 98:
        biome_value = 15
    elif biome_value == 99:
        biome_value = 16
    biome_value = int(biome_value) - 1
    plot_title = f'{BIOMES[biome_value]['name']}' if biome_value is not None else 'All Biomes'
    file_name = f'scatter_plot_gedi_vs_ours_{var}_{BIOMES[biome_value]["abbr"].replace(".", "")}.pdf'
    fig, ax = plt.subplots(1, 1, figsize=(7, 6))
    sns.histplot(df, x=f'{var}_gedi', y = f'{var}_ours', bins=50, cbar=False, cmap='viridis', ax=ax)
    ax.collections[0].set_norm(LogNorm(vmin=1, vmax=1000))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    fig.colorbar(ax.collections[0], cax=cax)
    ax.set_xlim(0, max_value)
    ax.set_ylim(0, max_value)
    ax.plot(np.arange(max_value), np.arange(max_value), color='black', linestyle='dashed')
    ax.set_title(plot_title)
    ax.text(0.05, 0.95, f'R$^2$ = {metrics[f"{var}_r2"]:.2f}', ha='left', va='top', transform=ax.transAxes, fontsize=14)
    ax.text(0.05, 0.90, f'Corr = {metrics[f"{var}_corr"]:.2f}', ha='left', va='top', transform=ax.transAxes, fontsize=14)
    ax.text(0.05, 0.85, f'N = {metrics[f"{var}_n"]}', ha='left', va='top', transform=ax.transAxes, fontsize=14)
    ax.set_aspect('equal')
    fig.savefig(save_dir / file_name, bbox_inches='tight')
    plt.close(fig)  # explicitly close THIS figure

def boxplot_with_marginal_histograms(df: pd.DataFrame, var: str, rh_col: str='rh98', bin_width:int=5, max_height:int=50, max_value:int=50, save_dir: Path = None) -> None:
    '''
    NOTE: bin width here means the canopy top height bin width, not the vertical resolution width
    '''
    biome_value = df['BIOME'].unique()[0]
    df = df[df[var] <= max_value]
    if biome_value == 98:
        biome_value = 15
    elif biome_value == 99:
        biome_value = 16
    biome_value = int(biome_value) - 1
    plot_title = f'{BIOMES[biome_value]['name']}' if biome_value is not None else 'All Biomes'
    file_name = f'boxplot_{var}_{rh_col}_{BIOMES[biome_value]["abbr"].replace(".", "")}.pdf'
    # Create 5m height bins based on rh98
    bin_edges = np.arange(0, max_height+bin_width, bin_width)
    df['height_bin'] = pd.cut(df[rh_col], bins=bin_edges, right=False)
    df['bin_label'] = df['height_bin'].apply(
        lambda x: f"{int(x.left)}-{int(x.right)}" if pd.notna(x) else None
    )
    df = df.dropna(subset=['bin_label'])
    ordered_bin = sorted(df['bin_label'].unique(), key=lambda x: int(x.split('-')[0]))
    counts = df.groupby('bin_label').size()
    
    # Set up grid: main axes + marginal on the right
    fig = plt.figure(figsize=(10, 8))
    gs = gridspec.GridSpec(2, 2, width_ratios=[5, 1], height_ratios=[1, 5],
                           wspace=0.05, hspace=0.05)

    ax_top  = fig.add_subplot(gs[0, 0])          # top marginal
    ax_main = fig.add_subplot(gs[1, 0])           # main boxplot
    ax_marg = fig.add_subplot(gs[1, 1], sharey=ax_main)  # right marginal
    ax_empty = fig.add_subplot(gs[0, 1])          # empty corner
    ax_empty.axis('off')

    # --- Main boxplot ---
    data = [df[df['bin_label'] == b][var].values for b in ordered_bin]
    bp = ax_main.boxplot(
        data,
        tick_labels=ordered_bin,
        patch_artist=True,
        widths=0.6,
        showfliers=True,
        flierprops=dict(marker='o', markersize=3, alpha=0.5, markerfacecolor='grey'),
        medianprops=dict(color='darkred', linewidth=1.5),
        boxprops=dict(facecolor='#4C72B0', alpha=0.7, edgecolor='black'),
        whiskerprops=dict(color='black'),
        capprops=dict(color='black'),
    )

    # Annotate sample sizes
    ymax = df[var].max()
    for i, b in enumerate(ordered_bin):
        n = counts.get(b, 0)
        ax_main.text(i + 1, ymax + 0.08, f'n={n}',
                    ha='center', va='bottom', fontsize=7, color='grey')

    ax_main.set_xlabel(f'Canopy Top Height (m)', fontsize=12)
    ax_main.set_ylabel(var, fontsize=12)
    ax_main.tick_params(axis='x', rotation=45)
    ax_main.grid(axis='y', alpha=0.3)

    # --- Right marginal KDE (distribution of var) ---
    vals_var = df[var].dropna().values
    kde_var = gaussian_kde(vals_var)
    y_grid = np.linspace(vals_var.min() - 0.2, vals_var.max() + 0.2, 300)
    density_var = kde_var(y_grid)

    ax_marg.fill_betweenx(y_grid, density_var, alpha=0.4, color='#4C72B0')
    ax_marg.plot(density_var, y_grid, color='#4C72B0', linewidth=1.2)
    ax_marg.tick_params(labelleft=False, left=False)
    ax_marg.set_xlabel('Density', fontsize=10)
    ax_marg.grid(axis='y', alpha=0.3)
    ax_marg.spines['top'].set_visible(False)
    ax_marg.spines['right'].set_visible(False)
    ax_marg.spines['left'].set_visible(False)

    # --- Top marginal KDE (distribution of rh_col) ---
    vals_rh = df[rh_col].dropna().values
    kde_rh = gaussian_kde(vals_rh)
    x_grid = np.linspace(vals_rh.min() - 0.2, vals_rh.max() + 0.2, 300)
    density_rh = kde_rh(x_grid)

    ax_top.fill_between(x_grid, density_rh, alpha=0.4, color='#4C72B0')
    ax_top.plot(x_grid, density_rh, color='#4C72B0', linewidth=1.2)
    ax_top.tick_params(labelbottom=False, bottom=False)
    ax_top.set_ylabel('Density', fontsize=10)
    ax_top.grid(axis='x', alpha=0.3)
    ax_top.spines['top'].set_visible(False)
    ax_top.spines['right'].set_visible(False)
    ax_top.spines['bottom'].set_visible(False)

    # Align the top marginal x-axis with the main boxplot x-axis
    # The boxplot x-axis goes from 0.5 to len(ordered_bin)+0.5 (categorical positions)
    # Map the continuous rh_col range to those positions
    ax_top.set_xlim(bin_edges[0], bin_edges[-1])
    ax_main.set_xlim(0.5, len(ordered_bin) + 0.5)
    
    # Pearson correlation (top-left of main axes)
    r, p = pearsonr(df[rh_col], df[var])
    ax_main.text(0.02, 0.9, f'r = {r:.3f} (p = {p:.1e})',
                transform=ax_main.transAxes, fontsize=10,
                verticalalignment='top', fontstyle='italic',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    # Within-bin std (annotate below each box)
    for i, b in enumerate(ordered_bin):
        std = df[df['bin_label'] == b][var].std()
        ax_main.text(i + 1, ax_main.get_ylim()[0] + 0.05, f'σ={std:.2f}',
                    ha='center', va='bottom', fontsize=6, color='steelblue')

    fig.suptitle(f'{plot_title}', fontsize=14, y=0.98)
    fig.savefig(save_dir / file_name, dpi=200, bbox_inches='tight')
    plt.close(fig)

def boxplot_with_marginal_histograms_combined(
    df: pd.DataFrame,
    var: str,
    rh_col: str = 'rh98',
    bin_width: int = 5,
    max_height: int = 50,
    max_value: int = 50,
    biome_col: str = 'BIOME',
    save_dir: Path = None,
) -> None:
    """
    Single plot with grouped boxplots: one box per biome within each height bin.
    """
    df = df.copy()
    df = df.dropna(subset=[var, rh_col, biome_col])
    df = df[df[biome_col] < 90] # NOTE: exclude biome 98 and 99
    df = df[df[var] <= max_value]

    # Map biome codes to readable names
    def biome_label(code):
        code = int(code)
        if code == 98:
            idx = 14
        elif code == 99:
            idx = 15
        else:
            idx = code - 1
        return BIOMES[idx]['name']

    df['biome_name'] = df[biome_col].apply(biome_label)
    biome_names = sorted(df['biome_name'].unique())
    n_biomes = len(biome_names)

    # Height bins
    bin_edges = np.arange(0, max_height + bin_width, bin_width)
    df['height_bin'] = pd.cut(df[rh_col], bins=bin_edges, right=False)
    df['bin_label'] = df['height_bin'].apply(
        lambda x: f"{int(x.left)}-{int(x.right)}" if pd.notna(x) else None
    )
    df = df.dropna(subset=['bin_label'])
    ordered_bins = sorted(df['bin_label'].unique(), key=lambda x: int(x.split('-')[0]))
    n_bins = len(ordered_bins)

    # Color palette
    cmap = plt.cm.get_cmap('tab10', n_biomes)
    # colors = [cmap(i) for i in range(n_biomes)]
    colors = [
        colorsys.hls_to_rgb(i / n_biomes, 0.45, 0.75)
        for i in range(n_biomes)
    ]

    # --- Layout ---
    fig = plt.figure(figsize=(max(14, n_bins * 2), 9))
    gs = gridspec.GridSpec(2, 2, width_ratios=[5, 1], height_ratios=[1, 5],
                           wspace=0.05, hspace=0.05)
    ax_top = fig.add_subplot(gs[0, 0])
    ax_main = fig.add_subplot(gs[1, 0])
    ax_marg = fig.add_subplot(gs[1, 1], sharey=ax_main)
    fig.add_subplot(gs[0, 1]).axis('off')

    # --- Grouped boxplots ---
    total_width = 0.75  # total width allocated per height bin
    box_width = total_width / n_biomes
    positions_map = {}  # (bin_label, biome) -> x position

    for i, b in enumerate(ordered_bins):
        center = i + 1
        start = center - total_width / 2 + box_width / 2
        for j, biome in enumerate(biome_names):
            pos = start + j * box_width
            positions_map[(b, biome)] = pos

    bp_artists = []
    for j, biome in enumerate(biome_names):
        positions = []
        data = []
        for b in ordered_bins:
            subset = df[(df['bin_label'] == b) & (df['biome_name'] == biome)][var].values
            data.append(subset if len(subset) > 0 else [np.nan])
            positions.append(positions_map[(b, biome)])

        bp = ax_main.boxplot(
            data,
            positions=positions,
            widths=box_width * 0.85,
            patch_artist=True,
            showfliers=True,
            flierprops=dict(marker='o', markersize=2, alpha=0.4, markerfacecolor=colors[j]),
            medianprops=dict(color='darkred', linewidth=1.2),
            boxprops=dict(facecolor=colors[j], alpha=0.7, edgecolor='black', linewidth=0.5),
            whiskerprops=dict(color='black', linewidth=0.5),
            capprops=dict(color='black', linewidth=0.5),
            manage_ticks=False,
        )
        bp_artists.append((bp, biome))

    # X-axis ticks at bin centers
    ax_main.set_xticks([i + 1 for i in range(n_bins)])
    ax_main.set_xticklabels(ordered_bins, rotation=45, ha='right')
    ax_main.set_xlim(0.5, n_bins + 0.5)
    ax_main.set_xlabel('Canopy Top Height (m)', fontsize=12)
    ax_main.set_ylabel(var, fontsize=12)
    ax_main.grid(axis='y', alpha=0.3)

    # Legend
    legend_patches = [mpatches.Patch(facecolor=colors[j], alpha=0.7, edgecolor='black',
                                      label=biome_names[j]) for j in range(n_biomes)]
    ax_main.legend(handles=legend_patches, fontsize=7, loc='upper left',
                   ncol=max(1, n_biomes // 4), framealpha=0.8)

    # --- Right marginal: one KDE per biome ---
    for j, biome in enumerate(biome_names):
        vals = df[df['biome_name'] == biome][var].dropna().values
        if len(vals) < 2:
            continue
        kde = gaussian_kde(vals)
        y_grid = np.linspace(df[var].min() - 0.2, df[var].max() + 0.2, 300)
        ax_marg.plot(kde(y_grid), y_grid, color=colors[j], linewidth=1, alpha=0.7)
    ax_marg.tick_params(labelleft=False, left=False)
    ax_marg.set_xlabel('Density', fontsize=10)
    ax_marg.spines['top'].set_visible(False)
    ax_marg.spines['right'].set_visible(False)
    ax_marg.spines['left'].set_visible(False)

    # --- Top marginal: one KDE per biome ---
    for j, biome in enumerate(biome_names):
        vals = df[df['biome_name'] == biome][rh_col].dropna().values
        if len(vals) < 2:
            continue
        kde = gaussian_kde(vals)
        x_grid = np.linspace(0, max_height, 300)
        ax_top.plot(x_grid, kde(x_grid), color=colors[j], linewidth=1, alpha=0.7)
    ax_top.set_xlim(bin_edges[0], bin_edges[-1])
    ax_top.tick_params(labelbottom=False, bottom=False)
    ax_top.set_ylabel('Density', fontsize=10)
    ax_top.spines['top'].set_visible(False)
    ax_top.spines['right'].set_visible(False)
    ax_top.spines['bottom'].set_visible(False)

    fig.suptitle(f'{var} by Canopy Height across Biomes', fontsize=14, y=0.98)
    file_name = f'boxplot_combined_{var}_{rh_col}.pdf'
    if save_dir:
        fig.savefig(save_dir / file_name, dpi=200, bbox_inches='tight')
        print(f'Saved to {save_dir / file_name}')
    plt.close(fig)

def plot_residuals_rh98_bined(df: pd.DataFrame, save_dir: Path = None, max_height: int = 50, rh98_interval: int = 5, **kwargs):
    df['rh98_bins'] = pd.cut(df['rh98'], bins=np.arange(0, max_height + rh98_interval, rh98_interval), right=False)

    for var, name in [('fhd', 'FHD'), ('enl1d', 'ENL 1D'), ('enl2d', 'ENL 2D'), ('cr', 'CR')]:
        grouped = df.groupby('rh98_bins', observed=True)[f'{var}_diff']
        labels = [str(k) for k in grouped.groups.keys()]
        data = [g.dropna().values for _, g in grouped]
        counts = [len(d) for d in data]

        fig, ax = plt.subplots(figsize=(10, 6))

        # --- background count bars on secondary axis ---
        ax_bar = ax.twinx()
        ax_bar.bar(range(1, len(counts) + 1), counts, color='lightgrey', alpha=0.3, width=0.6, zorder=1)
        ax_bar.set_ylabel('Count', color='grey')
        ax_bar.tick_params(axis='y', labelcolor='grey')
        ax_bar.set_ylim(0, max(counts) * 3)  # push bars down to ~1/3 of plot height

        # count labels: format large numbers with comma separator
        for i, n in enumerate(counts, start=1):
            ax_bar.text(i, n + max(counts) * 0.02, f'{n:,}', ha='center', va='bottom',
                        fontsize=7, color='grey', fontweight='light')

        # --- dashed zero line (behind boxes, in front of bars) ---
        ax.axhline(y=0, color='dimgrey', linestyle='--', linewidth=0.8, zorder=2)

        # --- boxplot on top ---
        ax.boxplot(data, labels=labels, showfliers=False, zorder=3)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_xlabel('Canopy Top Height (m)')
        ax.set_ylabel(f'{name} Residual')
        ax.set_title(f'{name} Residuals by Canopy Top Height')

        # keep boxplot axis in front
        ax.set_zorder(ax_bar.get_zorder() + 1)
        ax.patch.set_visible(False)

        fig.tight_layout()
        for ext in ('pdf', 'png'):
            fig.savefig(save_dir / f'residuals_{var}_rh98_binned.{ext}', dpi=300, bbox_inches='tight')
        print(f"Saved to {save_dir / f'residuals_{var}_rh98_binned.[pdf|png]'}")
        plt.close(fig)

# ------------------------------------------------------------------------------------------------
# Diversity indices
# ------------------------------------------------------------------------------------------------
def pixel_diversity_indices(rhs, bin_width=5, max_height=MAX_HEIGHT):
    """
    Compute per-pixel FHD using a simple histogram approach. NOTE!!!: rhs should be in meters!!!
    """
    if isinstance(rhs, pd.Series):
        rhs = rhs.values
    rhs_arr = np.asarray(rhs, dtype=np.float32)
    valid = np.isfinite(rhs_arr) & (rhs_arr > 0)
    rhs_valid = rhs_arr[valid]
    if rhs_valid.size == 0:
        return np.nan, np.nan, np.nan, np.nan
    rhs_valid = np.clip(rhs_valid, None, max_height)
    n_bins = int(max_height / bin_width)
    hist, bins = np.histogram(rhs_valid, bins=n_bins, range=(0, max_height)) # negative values are ignored
    p = hist / hist.sum() # NOTE: normalize the histogram to get the probability
    mask = p > 0
    fhd = -np.sum(p[mask] * np.log(p[mask])).astype(np.float32)
    enl1d = np.exp(fhd)
    enl2d = np.float32(1.0 / np.sum(p[mask] ** 2))
    if np.isinf(enl2d):
        print(f'ENL2D is inf, setting to NaN, rhs: {rhs}, p: {p}')
        enl2d = np.nan
    rh25 = max(rhs_arr[25], 0)
    if rhs_arr[98] <= 0:
        cr = np.nan
    else:
        cr = (rhs_arr[98] - rh25)/rhs_arr[98]
    
    return fhd, enl1d, enl2d, cr


def pixel_vertical_profile(rhs, min_rh=-20, max_rh=50, step=0.1, window=3):
    
    rhs_arr = np.asarray(rhs, dtype=np.float32)
    rhs_arr = rhs_arr[np.isfinite(rhs_arr)]
    x = np.arange(min_rh, max_rh + step, step, dtype=np.float32)

    if rhs_arr.size < 3:
        return x, np.zeros_like(x, dtype=np.float32)

    rhs_unique = np.unique(np.sort(rhs_arr))
    if rhs_unique.size < 3:
        return x, np.zeros_like(x, dtype=np.float32)

    ones = np.arange(rhs_unique.size, dtype=np.float32)
    grad = np.gradient(ones, rhs_unique)
    grad = np.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
    grad_inter = interp1d(rhs_unique, grad, kind='linear', fill_value=0, bounds_error=False)
    grad_resampled = grad_inter(x)

    max_window = int(grad_resampled.size) if grad_resampled.size % 2 == 1 else int(grad_resampled.size) - 1
    safe_window = min(window if window % 2 == 1 else window + 1, max_window)
    if safe_window < 3:
        smoothed_grad = grad_resampled
    else:
        smoothed_grad = savgol_filter(grad_resampled, safe_window, 1)
    return x, smoothed_grad.astype(np.float32)

def _chunk_diversity(data, bin_width=5, max_height=MAX_HEIGHT):
    """
    Vectorized Shannon entropy for a single spatial chunk. data is in decimeters.

    Parameters
    ----------
    tile : ndarray, shape (101, rows, cols)

    Returns
    -------
    out : ndarray, shape (rows, cols), float32
    """
    if data.size == 3:
        data = data[None, ...]  # add band dimension for consistency

    n_batch, n_bands, n_rows, n_cols = data.shape
    hist, nodata_mask = batch_binning(data[None, ...], bin_width=bin_width, max_height=max_height)

    # Normalize
    total = hist.sum(axis=-1, keepdims=True)
    total = np.where(total > 0, total, 1.0)
    p = hist / total

    # FHD
    log_p = np.where(p > 0, np.log(p), 0.0)
    fhd = -np.sum(p * log_p, axis=-1).astype(np.float32)

    # ENL1D
    enl1d = np.exp(fhd).astype(np.float32)

    # ENL2D
    sum_p2 = np.sum(np.where(p > 0, p ** 2, 0.0), axis=-1)
    enl2d = np.where(sum_p2 > 0, 1.0 / sum_p2, np.nan).astype(np.float32)

    # CR
    # TODO: check order of RHs? at least rh25 and rh98
    rh25 = np.maximum(data[24, :], 0)
    rh98 = data[97, :]
    cr = np.where(rh98 > 0, (rh98 - rh25) / rh98, np.nan).astype(np.float32)

    fhd = fhd.reshape(n_batch, n_rows, n_cols)
    enl1d = enl1d.reshape(n_batch, n_rows, n_cols)
    enl2d = enl2d.reshape(n_batch, n_rows, n_cols)
    cr = cr.reshape(n_batch, n_rows, n_cols)
    # Apply nodata
    fhd[nodata_mask] = np.nan
    enl1d[nodata_mask] = np.nan
    enl2d[nodata_mask] = np.nan
    cr[nodata_mask] = np.nan
    return fhd, enl1d, enl2d, cr

def _process_tile(args):
    """
    Worker function for ProcessPoolExecutor.
    Each worker opens its own file handle (required for multiprocessing).
    Reads all 101 bands for one window in a single call, computes entropy.
    """
    vrt_path, col_off, row_off, w, h, bin_width, max_height = args
    with rasterio.open(vrt_path, "r") as src:
        window = Window(col_off, row_off, w, h)
        tile = src.read(window=window).astype(np.float32)  # (101, h, w)
    ent, enl1d, enl2d, cr = _chunk_diversity(tile, bin_width=bin_width, max_height=max_height)
    return ent, enl1d, enl2d, cr, col_off, row_off, w, h


def compute_entropy(output_dir, tile_id, year, vrt_path=None,
                         chunk_size=512, max_workers=8, bin_width=5, max_height=MAX_HEIGHT, **kwargs):
    """
    Compute per-pixel FHD entropy — fast version.

    Two key speedups over the stackstac version:
    1. VRT + rasterio windowed read: single I/O call reads all 101 bands
       for a spatial window (vs stackstac opening 101 separate COGs)
    2. ProcessPoolExecutor: true CPU parallelism for entropy computation
       (vs ThreadPoolExecutor which is GIL-limited for numpy)

    Parameters
    ----------
    output_dir : str
        Output single-band GeoTIFF path.
    tile_id : str
        Tile identifier (e.g. '36NTF').
    year : int
        Year for file lookup.
    vrt_path : str, optional
        Path to 101-band VRT. If None, auto-creates from tile directory.
    chunk_size : int
        Spatial chunk size in pixels.
    max_workers : int
        Number of parallel processes.
    """
    output_dir = Path(output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f'{tile_id}.tif'
    # if output_path.exists():
    #     return

    # Create VRT if not provided
    if vrt_path is None:
        tile_dir = f"~/data/gvs/products/vsm/{year}/original/tiles/cog/{tile_id}"
        vrt_path = f"~/data/gvs/products/vsm/{year}/original/vrt/{tile_id}_Q1.vrt"
        create_vrt(tile_dir, vrt_path)

    vrt_path = str(Path(vrt_path).expanduser())

    # Get metadata from VRT
    with rasterio.open(vrt_path, "r") as src:
        ny = src.height
        nx = src.width
        crs = src.crs
        transform = src.transform
        n_bands = src.count
        print(f"VRT: {nx}x{ny}, {n_bands} bands, dtype={src.dtypes[0]}")

    assert n_bands == 100, f"Expected 100 bands, got {n_bands}"

    out_profile = {
        "driver": "GTiff",
        "dtype": "float32",
        "height": ny,
        "width": nx,
        "count": 4,
        "crs": crs,
        "transform": transform,
        "nodata": NODATA_OUT,
        "compress": None,
        "tiled": True,
        "blockxsize": 512,
        "blockysize": 512,
    }

    # Build work items
    work_items = []
    for row_off in range(0, ny, chunk_size):
        for col_off in range(0, nx, chunk_size):
            h = min(chunk_size, ny - row_off)
            w = min(chunk_size, nx - col_off)
            ent, enl1d, enl2d, cr = _process_tile((vrt_path, col_off, row_off, w, h, bin_width, max_height))
            work_items.append((vrt_path, col_off, row_off, w, h, bin_width, max_height))

    print(f"Processing {len(work_items)} tiles with {max_workers} processes")
    print(f"Estimated peak RAM: ~{max_workers * 100 * chunk_size**2 * 4 / 1e9:.1f} GB")

    monitor = ProgressMonitor(total_tiles=len(work_items), interval=5.0)

    with (rasterio.open(output_path, "w", **out_profile) as dst):
        monitor.start()
        try:
            with ProcessPoolExecutor(max_workers=max_workers) as pool:
                futures = {
                    pool.submit(_process_tile, item): i
                    for i, item in enumerate(work_items)
                }
                for fut in as_completed(futures):
                    ent, enl1d, enl2d, cr, col_off, row_off, w, h = fut.result()
                    win = Window(col_off, row_off, w, h)
                    dst.write(np.stack([ent, enl1d, enl2d, cr], axis=0), window=win)
                    monitor.tick()
        finally:
            dst.set_band_description(1, "fhd")
            dst.set_band_description(2, "enl1d")
            dst.set_band_description(3, "enl2d")
            dst.set_band_description(4, "cr")
            monitor.stop()

    print(f"Done: {output_path}")
    return output_path

# ------------------------------------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------------------------------------
def create_vrt(tile_dir, vrt_path, q_idx="1"):
    tile_dir = Path(tile_dir).expanduser()
    vrt_path = Path(vrt_path).expanduser()
    vrt_path.parent.mkdir(parents=True, exist_ok=True)

    tile_id = tile_dir.stem
    files = sorted(
        glob.glob(f"{tile_dir}/RH*_Q{q_idx}.tif"),
        key=lambda x: int(x.split("RH")[-1].split("_")[0]),
    )
    files = files[1:] # remove RH0_Q1.tif
    print(f"Creating VRT for {tile_id} with {len(files)} files")
    assert len(files) == 100, f"Expected 100 files, found {len(files)}"

    vrt_options = gdal.BuildVRTOptions(separate=True)
    vrt = gdal.BuildVRT(str(vrt_path), files, options=vrt_options)

    for i in range(1, len(files)):
        band = vrt.GetRasterBand(i)
        band.SetDescription(f"RH{i - 1}")

    vrt.FlushCache()
    vrt = None
    return vrt_path

# ------------------------------------------------------------------------------------------------
# Orchestrator
# ------------------------------------------------------------------------------------------------

def vertical_profile_biome_analysis(points_file, save_dir, s2_grid_file=None, gedi_ref_dir=None, max_distance=1000):
    points_file = Path(points_file).expanduser()
    save_dir = Path(save_dir).expanduser()
    s2_grid_file = Path(s2_grid_file).expanduser()
    gedi_ref_dir = Path(gedi_ref_dir).expanduser()
    save_dir.mkdir(parents=True, exist_ok=True)
    points = gpd.read_file(points_file)
    s2_grid = gpd.read_parquet(s2_grid_file)
    points.loc[points['Biomes'] == 'TmpBroMixF', 'Biomes'] = 'TempBrdMix'
    points.loc[points['Biomes'] == 'TempConFo', 'Biomes'] = 'TmpConF'
    points.loc[points['Biomes'] == 'TmpConiF', 'Biomes'] = 'TmpConF'
    points.loc[points['Biomes'] == 'TrSuMoBrF', 'Biomes'] = 'TrpSbMoBrF'
    points.loc[points['Biomes'] == 'TrSbDrBrF', 'Biomes'] = 'TrpSbDrBrF'
    points.loc[points['Biomes'] == 'TrSbGrSvSh', 'Biomes'] = 'TrpSbGrSvh'
    
    points = points.sjoin(s2_grid, how='left', predicate='within')
    points = points.drop(columns=['index_right'])
    vertical_prof = {biome: [] for biome in points['Biomes'].unique()}
    gedi_ref_points = {biome: [] for biome in points['Biomes'].unique()}
    rh_cols = [f'rh{i}' for i in range(101)]
    for idx, point in points.iterrows():
        tile_id = point['Name']
        if not (gedi_ref_dir / f'{tile_id}.parquet').exists():
            continue
        gedi_ref = gpd.read_parquet(gedi_ref_dir / f'{tile_id}.parquet')
        point_gdf = gpd.GeoDataFrame([point], geometry='geometry', crs=points.crs)
        point_gdf = point_gdf.to_crs(epsg=4087)
        gedi_proj = gedi_ref.to_crs(epsg=4087)
        intersects = point_gdf.sjoin_nearest(gedi_proj, how='left', max_distance=max_distance)
        if intersects['sensitivity'].isnull().all() or intersects['sensitivity'].max() < 0.95:
            print(f'{tile_id} has no intersects or sensitivity less than 0.95, skipping')
            continue
        closest = intersects.iloc[intersects['sensitivity'].argmax()]
        gedi_ref_points[point['Biomes']].append(closest)
        x, smoothed_grad = pixel_vertical_profile(closest[rh_cols], min_rh=-20, max_rh=50, step=0.1, window=3)
        vertical_prof[point['Biomes']].append(smoothed_grad)
    
    gedi_ref_list = []
    for biome in gedi_ref_points.keys():
        if len(gedi_ref_points[biome]) == 0:
            print(f'{biome} found no GEDI reference points, skipping')
            continue
        gedi_ref_list.append(pd.DataFrame(gedi_ref_points[biome]))
    gedi_ref_points = pd.concat(gedi_ref_list, ignore_index=True)
    gedi_ref_points = gpd.GeoDataFrame(gedi_ref_points, geometry='geometry', crs="EPSG:4326")
    gedi_ref_points.to_parquet(save_dir / 'gedi_ref_points.parquet')
    avg_metrics = {biome: {'fhd': -1, 'enl1d': -1, 'enl2d': -1} for biome in vertical_prof.keys()}
    for biome in vertical_prof.keys():
        if len(vertical_prof[biome]) == 0:
            print(f'{biome} found no vertical profiles, skipping')
            continue
        prof = np.stack(vertical_prof[biome], axis=1)
        indices = [pixel_diversity_indices(prof[:, i]) for i in range(prof.shape[1])]
        avg_fhd = np.mean([indices[0] for indices in indices])
        avg_enl1d = np.mean([indices[1] for indices in indices])
        avg_enl2d = np.mean([indices[2] for indices in indices])
        avg_metrics[biome]['fhd'] = avg_fhd
        avg_metrics[biome]['enl1d'] = avg_enl1d
        avg_metrics[biome]['enl2d'] = avg_enl2d
        avg_prof = prof.mean(axis=1)
        std_prof = prof.std(axis=1)
        plt.plot(avg_prof, x, label=f'{biome}')
        plt.fill_betweenx(x, avg_prof - std_prof, avg_prof + std_prof, alpha=0.2)
        plt.legend()
        plt.title(f'{biome}: FHD={avg_fhd:.2f}, ENL1D={avg_enl1d:.2f}, ENL2D={avg_enl2d:.2f}, N={prof.shape[1]}')
        plt.savefig(f"{save_dir}/{biome}.png")
        plt.close()
        
    avg_metrics = pd.DataFrame(avg_metrics).T
    avg_metrics.to_csv(save_dir / 'avg_metrics.csv')
    
def vertical_profile_biome_analysis_ours(points_file, save_dir, s2_grid_file=None, gedi_ref_points_file=None, pred_dir=None, max_distance=1000, vrt_path=None, year=2020):
    points_file = Path(points_file).expanduser()
    save_dir = Path(save_dir).expanduser()
    s2_grid_file = Path(s2_grid_file).expanduser()
    pred_dir = Path(pred_dir).expanduser()
    gedi_ref_points_file = Path(gedi_ref_points_file).expanduser()
    gedi_ref_points = gpd.read_parquet(gedi_ref_points_file)
    save_dir.mkdir(parents=True, exist_ok=True)
    s2_grid = gpd.read_parquet(s2_grid_file)
    tile_ids = s2_grid['Name'].unique()
    rh_cols = [f'rh{i}' for i in range(101)]
    for tile_id in tile_ids:
        if not (pred_dir / f'{tile_id}/RH98_Q1.tif').exists():
            continue
        # extract the vertical profile from the predicted RH98
            # Create VRT if not provided
        if vrt_path is None:
            tile_dir = f"~/data/gvs/products/vsm/{year}/original/tiles/cog/{tile_id}"
            vrt_path = f"~/data/gvs/products/vsm/{year}/original/vrt/{tile_id}_Q1.vrt"
            create_vrt(tile_dir, vrt_path)
        vrt_path = str(Path(vrt_path).expanduser())
        #TODO: ...

def cal_diversity_indices(save_dir, gedi_ours_dir, bin_width=5, max_height=MAX_HEIGHT, year=2020, **kwargs):
    '''Evaluate the diversity indices against the GEDI reference points on the test set'''
    save_dir = Path(save_dir).expanduser()
    save_dir.mkdir(parents=True, exist_ok=True)
    gedi_ours_dir = Path(gedi_ours_dir).expanduser()
    gedi_ours_dir.mkdir(parents=True, exist_ok=True)
    gedi_ours_files = sorted(gedi_ours_dir.glob('*.parquet'))
    gedi_ref_cols = [f'rh{i}' for i in range(101)]
    ours_cols = [f'RH{i}_Q1_raw' for i in range(101)]
    
    def _tile_level_indices(gedi_ours_file: Path):
        tile_id = gedi_ours_file.stem
        if (save_dir / f'{tile_id}.parquet').exists():
            return None
        gedi_ours = gpd.read_parquet(gedi_ours_file)
        gedi_ours = gedi_ours.dropna(subset=gedi_ref_cols + ours_cols)
        mask = (gedi_ours['pft_class'].isin(np.arange(1,11)) & gedi_ours['sensitivity'] >= 0.95)
        gedi_ours = gedi_ours[mask]
        if gedi_ours.empty:
            return None
        
        indices_gedi = gedi_ours[gedi_ref_cols].apply(lambda x: pixel_diversity_indices(x, bin_width=bin_width, max_height=max_height), axis=1)
        indices_gedi = pd.DataFrame(indices_gedi.tolist(), index=indices_gedi.index, columns=['fhd_gedi', 'enl1d_gedi', 'enl2d_gedi', 'cr_gedi'])
        indices_ours = gedi_ours[ours_cols].apply(lambda x: pixel_diversity_indices(x, bin_width=bin_width, max_height=max_height), axis=1)
        indices_ours = pd.DataFrame(indices_ours.tolist(), index=indices_ours.index, columns=['fhd_ours', 'enl1d_ours', 'enl2d_ours', 'cr_ours'])
        df = pd.concat([indices_gedi, indices_ours, gedi_ours[GEDI_META_COLS + RH_COLS]], axis=1)
        df = gpd.GeoDataFrame(df, geometry='geometry', crs="EPSG:4326")
        df.to_parquet(save_dir / f'{tile_id}.parquet')
    
    tasks = []
    # gedi_ours_files = [ f for f in gedi_ours_files if f.stem=='32QND']
    for gedi_ours_file in gedi_ours_files:
        # _tile_level_indices(gedi_ours_file)
        tasks.append(dask.delayed(_tile_level_indices)(gedi_ours_file))
    with ProgressBar():
        dask.compute(*tasks)


def eval_diversity_indices(indices_dir:str=None, bin_width:int=5, group_by=None, year=2020, save_dir=None, filter_steep_slope=False, plot_scatter=False, plot_boxplot=False, max_height=50, **kwargs):
    # Extract bin_width from indices_dir string (looks for "bin_width_" followed by digits)

    indices_dir = Path(indices_dir).expanduser()
    save_dir = Path(save_dir).expanduser()
    save_dir.mkdir(parents=True, exist_ok=True)
    indices_files = sorted(indices_dir.glob('*.parquet'))
    # indices_files = indices_files[100:110] # for testing
    ddf = dd.read_parquet(indices_files)
    if filter_steep_slope:
        ddf = ddf[ddf['slope'] <= 20]
        # save_dir = save_dir.parent / f'steep_slope_filtered_bin_{bin_width}m'
        # save_dir.mkdir(parents=True, exist_ok=True)
    ddf['fhd_diff'] = ddf['fhd_gedi'] - ddf['fhd_ours']
    ddf['enl1d_diff'] = ddf['enl1d_gedi'] - ddf['enl1d_ours']
    ddf['enl2d_diff'] = ddf['enl2d_gedi'] - ddf['enl2d_ours']
    ddf['cr_diff'] = ddf['cr_gedi'] - ddf['cr_ours']

    n_bins = int(100/bin_width)
    max_metrics = {
        'fhd': np.log(n_bins),
        'enl1d': n_bins,
        'enl2d': n_bins,
        'cr': 5.0,
    }
    # Group and compute metrics
    def compute_metrics(group):
        metrics = {}
        for var in ['fhd', 'enl1d', 'enl2d', 'cr']:
            gedi = group[f'{var}_gedi']
            ours = group[f'{var}_ours']
            diff = group[f'{var}_diff'] / max_metrics[var]
            metrics[f'{var}_corr'] = np.corrcoef(gedi, ours)[0, 1]
            metrics[f'{var}_r2'] =  r2_score(gedi, ours)
            metrics[f'{var}_rmse'] = np.sqrt(np.mean(diff**2))
            metrics[f'{var}_mae'] = np.mean(np.abs(diff))
            metrics[f'{var}_me'] = np.mean(diff)
            metrics[f'{var}_n'] = len(gedi)
            if plot_scatter:
                scatter_plot(group, var, metrics, biome_value=group['BIOME'].iloc[0], save_dir=save_dir, max_value=max_metrics[var])
            if plot_boxplot:
                boxplot_with_marginal_histograms(group, f'{var}_gedi', rh_col=f'rh98', save_dir=save_dir, max_height=max_height)
                boxplot_with_marginal_histograms(group, f'{var}_ours', rh_col=f'RH98_Q1_raw', save_dir=save_dir, max_height=max_height)
        return pd.Series(metrics)
    if group_by is not None:
        meta = pd.DataFrame({
            "fhd_corr": pd.Series(dtype="float32"),
            "fhd_r2": pd.Series(dtype="float32"),
            "fhd_rmse": pd.Series(dtype="float32"),
            "fhd_mae": pd.Series(dtype="float32"),
            "fhd_me": pd.Series(dtype="float32"),
            "fhd_n": pd.Series(dtype="int32"),
            
            "enl1d_corr": pd.Series(dtype="float32"),
            "enl1d_r2": pd.Series(dtype="float32"),
            "enl1d_rmse": pd.Series(dtype="float32"),
            "enl1d_mae": pd.Series(dtype="float32"),
            "enl1d_me": pd.Series(dtype="float32"),
            "enl1d_n": pd.Series(dtype="int32"),
            "enl2d_corr": pd.Series(dtype="float32"),
            "enl2d_r2": pd.Series(dtype="float32"),
            "enl2d_rmse": pd.Series(dtype="float32"),
            "enl2d_mae": pd.Series(dtype="float32"),
            "enl2d_me": pd.Series(dtype="float32"),
            "enl2d_n": pd.Series(dtype="int32"),
            "cr_corr": pd.Series(dtype="float32"),
            "cr_r2": pd.Series(dtype="float32"),
            "cr_rmse": pd.Series(dtype="float32"),
            "cr_mae": pd.Series(dtype="float32"),
            "cr_me": pd.Series(dtype="float32"),
            "cr_n": pd.Series(dtype="int32"),
        })
        # ddf = ddf.compute()
        # ddf.groupby('BIOME').apply(compute_metrics)
        result = ddf.groupby(group_by).apply(compute_metrics, meta=meta).compute()
        result.to_csv(save_dir / f'diversity_indices_evaluation_by_biomes_bin_width_{bin_width}m_{year}.csv')
    else:
        ddf = ddf.compute()
        plot_biome_combined_boxplot = kwargs.get('plot_biome_combined_boxplot', False)
        if plot_biome_combined_boxplot:
            boxplot_with_marginal_histograms_combined(
                ddf, var='fhd_gedi', rh_col='rh98', save_dir=save_dir
            )
        plot_residuals_rh98_bined(ddf, save_dir=save_dir, max_height=50)
        result = compute_metrics(ddf)
        records = {}
        for var, label in [('fhd', 'FHD'), ('enl1d', 'ENL 1D'), ('enl2d', 'ENL 2D'), ('cr', 'CR')]:
            records[label] = {
                'R2': result[f'{var}_r2'],
                'Corr': result[f'{var}_corr'],
                'RMSE': result[f'{var}_rmse'],
                'MAE': result[f'{var}_mae'],
                'ME': result[f'{var}_me'],
            }
        result_df = pd.DataFrame(records).T
        result_df.to_csv(save_dir / f'diversity_indices_evaluation_all_tiles_bin_width_{bin_width}m_{year}.csv')
  
def diversity_indices_distribution(indices_dir, save_dir, group_by=None, filter_steep_slope=False, year=2020, **kwargs):      
    indices_dir = Path(indices_dir).expanduser()
    save_dir = Path(save_dir).expanduser()
    save_dir.mkdir(parents=True, exist_ok=True)
    indices_files = sorted(indices_dir.glob('*.parquet'))
    for indices_file in indices_files:
        indices = gpd.read_parquet(indices_file)
        indices = indices[RH_COLS]
        indices = indices.apply(lambda x: pixel_diversity_indices(x, bin_width=5), axis=1)

if __name__ == "__main__":
    save_dir = '/projects/dereeco/data/gvs/evaluation/with_gedi_on_diversity_indices/indices_by_tile/bin_width_5m/2020'
    gedi_ours_dir = '/projects/dereeco/data/gvs/gedi/veg_sensitivity_gt0p95/subset_test/original_with_sota_chms_biome_and_ours_full/2020'
    # cal_diversity_indices(save_dir, gedi_ours_dir)
    tile_id = '36NTF'
    year = 2020
    vrt_path = f"~/data/gvs/products/vsm/{year}/original/vrt/{tile_id}_Q1.vrt"
    output_dir = f"~/data/gvs/products/profile_entropy/{year}/tiles/geotiff"

    # # Create VRT if needed
    tile_dir = f"~/data/gvs/products/vsm/{year}/original/tiles/cog/{tile_id}"
    vrt_resolved = Path(vrt_path).expanduser()
    if not vrt_resolved.exists():
        create_vrt(tile_dir, vrt_path)

    start = time.time()
    compute_entropy(
        output_dir=output_dir,
        tile_id=tile_id,
        year=year,
        vrt_path=vrt_path,
        chunk_size=512,
        max_workers=8,
        bin_width=5,
    )
    elapsed = time.time() - start
    print(f"Time taken: {elapsed:.2f} seconds")
    # # vertical_profile_per_biome(
    # #     points_file='/projects/dereeco/data/gvs/analysis/typical_forests/typical_forests.zip',
    # #     save_dir='/projects/dereeco/data/gvs/analysis/typical_forests/vertical_profile_gedi_ref',
    # #     s2_grid_file='/projects/dereeco/data/gvs/state/s2_tiles_with_growing_months.parquet',
    # #     gedi_ref_dir='/projects/dereeco/data/gvs/gedi/veg_sensitivity_gt0p95/all_valid/2020',
    # #     max_distance=1000
    # # )