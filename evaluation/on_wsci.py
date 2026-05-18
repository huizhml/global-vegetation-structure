from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import dask_geopandas as dgp
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import rasterio
from rasterio.crs import CRS
from rasterio.transform import rowcol
from rasterio.warp import transform as warp_transform
from rasterio.windows import Window
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm

from const import BIOMES

# Band order of the diversity raster (used as df columns / file slugs).
DIVERSITY_BANDS = ['fhd', 'enl1d', 'enl2d', 'cr']
# Human-readable labels for plot titles, axis labels and the CSV.
DIVERSITY_LABELS = {'fhd': 'FHD', 'enl1d': '1D ENL',
                    'enl2d': '2D ENL', 'cr': 'CR'}


def _biome_label(biome_value) -> tuple[str, str]:
    '''Map a raw BIOME value to its (name, file-safe abbr), matching the
    98->15 / 99->16 convention used in on_diversity_indices.scatter_plot.'''
    bv = int(biome_value)
    idx = 14 if bv == 98 else 15 if bv == 99 else bv - 1
    return BIOMES[idx]['name'], BIOMES[idx]['abbr'].replace('.', '')


def rasterio_read_locs(file: Path, gdf: gpd.GeoDataFrame,
                       block_size: int = 512) -> np.ndarray:
    '''
    Sample every band of a raster at the point locations in `gdf`.

    Points are reprojected to the raster CRS before sampling, then sampled
    block-by-block: each occupied `block_size`-pixel block of the raster is
    read once and indexed vectorised, instead of one windowed read per point.
    The result is identical to `src.sample` (nearest pixel containing the
    point) but with orders of magnitude fewer I/O calls. Pixels that fall
    outside the raster footprint or equal the raster nodata value are returned
    as NaN so they can be dropped pairwise during the correlation analysis.

    Returns an array of shape (n_points, n_bands).
    '''
    xs = gdf.geometry.x.to_numpy()
    ys = gdf.geometry.y.to_numpy()
    with rasterio.open(file, 'r') as src:
        out = np.full((len(xs), src.count), np.nan, dtype='float64')

        pts_epsg = gdf.crs.to_epsg() if gdf.crs is not None else None
        src_epsg = src.crs.to_epsg() if src.crs is not None else None
        if gdf.crs is not None and src.crs is not None and pts_epsg != src_epsg:
            xs, ys = warp_transform(
                CRS.from_user_input(gdf.crs), src.crs, xs.tolist(), ys.tolist()
            )
            xs, ys = np.asarray(xs), np.asarray(ys)

        rows, cols = rowcol(src.transform, xs, ys)  # vectorised, floor
        rows = np.asarray(rows, dtype=np.int64)
        cols = np.asarray(cols, dtype=np.int64)

        # Out-of-footprint points stay NaN (do NOT clip onto edge pixels).
        in_bounds = (
            (rows >= 0) & (rows < src.height) &
            (cols >= 0) & (cols < src.width)
        )
        idx = np.flatnonzero(in_bounds)
        if len(idx) == 0:
            return out
        r, c = rows[idx], cols[idx]

        # Group points by raster block; read each occupied block once.
        n_block_cols = (src.width + block_size - 1) // block_size
        block_id = (r // block_size) * n_block_cols + (c // block_size)
        order = np.argsort(block_id, kind='stable')
        idx, r, c = idx[order], r[order], c[order]
        _, starts = np.unique(block_id[order], return_index=True)
        starts = np.append(starts, len(idx))

        for i in tqdm(range(len(starts) - 1), desc=f'sampling {Path(file).name}'):
            sl = slice(starts[i], starts[i + 1])
            pr, pc = r[sl], c[sl]
            r0, c0 = int(pr.min()), int(pc.min())
            window = Window(c0, r0,
                            int(pc.max()) - c0 + 1, int(pr.max()) - r0 + 1)
            block = src.read(window=window)  # (n_bands, h, w)
            out[idx[sl]] = block[:, pr - r0, pc - c0].T

        if src.nodata is not None:
            out[out == src.nodata] = np.nan
    return out


def _corr_metrics(df: pd.DataFrame, x_name: str, y_name: str) -> dict:
    '''Pearson r, Spearman rho and R² for two columns (NaNs already dropped).'''
    r, p_r = pearsonr(df[x_name], df[y_name])
    rho, p_rho = spearmanr(df[x_name], df[y_name])
    return {
        'n': len(df),
        'pearson_r': r,
        'pearson_p': p_r,
        'spearman_rho': rho,
        'spearman_p': p_rho,
        'r2': r ** 2,
    }


def _scale(v: np.ndarray, how: str) -> np.ndarray:
    '''Standardize a 1-D array for *display only* (correlations are scale
    invariant, so this never touches the reported stats).'''
    v = np.asarray(v, dtype='float64')
    if how == 'zscore':
        s = v.std()
        return (v - v.mean()) / (s if s else 1.0)
    if how == 'minmax':
        lo, hi = v.min(), v.max()
        return (v - lo) / ((hi - lo) if hi > lo else 1.0)
    if how in (None, 'none'):
        return v
    raise ValueError(f"unknown scale '{how}' (use 'zscore', 'minmax', 'none')")


def correlation_plot(df: pd.DataFrame, x_name: str, y_name: str,
                     metrics: dict, save_path: Path, title: str,
                     scale: str = 'zscore',
                     x_label: str = None, y_label: str = None,
                     pclip: float = 1.0) -> None:
    '''Log-scaled density scatter of the two products with a 1:1 reference.

    Axes are standardized per `scale` ('zscore'|'minmax'|'none') so the two
    products are visually comparable despite different value ranges, and the
    view is clipped to a shared [`pclip`, 100-`pclip`] percentile box so a
    long tail in one product can't blow out the panel. The clip is *display
    only* — the Pearson/Spearman/R² annotation is computed on the raw values
    upstream over all points and is unaffected by this rescaling/clipping.
    '''
    x = _scale(df[x_name].to_numpy(), scale)
    y = _scale(df[y_name].to_numpy(), scale)
    unit = {'zscore': ' (z-score)', 'minmax': ' (min-max)'}.get(scale, '')

    # Shared limits at the [pclip, 100-pclip] percentile of each axis, so
    # neither axis clips its own bulk and the box stays square / 1:1 valid.
    lo = float(min(np.percentile(x, pclip), np.percentile(y, pclip)))
    hi = float(max(np.percentile(x, 100 - pclip),
                   np.percentile(y, 100 - pclip)))

    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    hb = ax.hist2d(x, y, bins=60, cmap='viridis', norm=LogNorm(vmin=1),
                   range=[[lo, hi], [lo, hi]])
    ax.plot([lo, hi], [lo, hi], color='black', linestyle='dashed', linewidth=1)

    # Square plot box + colorbar locked to the axes' height.
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect('equal')
    cax = make_axes_locatable(ax).append_axes('right', size='5%', pad=0.1)
    fig.colorbar(hb[3], cax=cax, label='count')

    ax.set_xlabel((x_label or x_name) + unit)
    ax.set_ylabel((y_label or y_name) + unit)
    ax.set_title(title)
    ax.text(0.05, 0.95,
            f"Pearson r = {metrics['pearson_r']:.3f}\n"
            f"Spearman ρ = {metrics['spearman_rho']:.3f}\n"
            f"R² = {metrics['r2']:.3f}\n"
            f"N = {metrics['n']}",
            ha='left', va='top', transform=ax.transAxes, fontsize=12,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    fig.savefig(save_path, bbox_inches='tight')
    plt.close(fig)


def _correlate(df: pd.DataFrame, wsci_name: str, index_name: str,
               save_dir: Path, group_by: str, min_points: int,
               scale: str = 'zscore') -> list[dict]:
    '''Correlate WSCI against one diversity index, overall and per group.'''
    rows = []
    label = DIVERSITY_LABELS.get(index_name, index_name)
    sub = df.dropna(subset=[wsci_name, index_name])

    m = _corr_metrics(sub, wsci_name, index_name)
    rows.append({'diversity_index': label, 'group': 'all',
                 'biome_value': -1, **m})
    correlation_plot(sub, wsci_name, index_name, m,
                     save_dir / f'wsci_vs_{index_name}_scatter_all.pdf',
                     title=f'{wsci_name} vs {label} (all biomes)',
                     scale=scale, y_label=label)
    print(f'{label} all:', m)

    if group_by:
        for biome_value, group in sub.groupby(group_by):
            name, abbr = _biome_label(biome_value)
            if len(group) < min_points:
                print(f'  skipping {index_name}/{name}: {len(group)} points')
                continue
            gm = _corr_metrics(group, wsci_name, index_name)
            rows.append({'diversity_index': label, 'group': name,
                         'biome_value': int(biome_value), **gm})
            correlation_plot(
                group, wsci_name, index_name, gm,
                save_dir / f'wsci_vs_{index_name}_scatter_{abbr}.pdf',
                title=f'{wsci_name} vs {label} ({name})',
                scale=scale, y_label=label,
            )
            print(f'  {label}/{name}:', gm)
    return rows


def eval_on_wsci(wsci_file: str, diversity_file: str, test_point_dir: str,
                 save_dir: str = None, wsci_name: str = 'WSCI',
                 diversity_bands: list = None, group_by: str = 'BIOME',
                 min_points: int = 30, plot_scale: str = 'zscore', **kwargs):
    '''
    Correlation analysis between WSCI and each band of a 4-band diversity
    raster (fhd, enl1d, enl2d, cr) sampled at the test point locations,
    both overall and per biome.

    Parameters:
        wsci_file: path to the single-band WSCI raster
        diversity_file: path to the multi-band diversity-index raster
        test_point_dir: directory of parquet files holding the test points
        save_dir: where to write the stats CSV and scatter plots
                  (defaults to `test_point_dir`)
        wsci_name: column / axis label for the WSCI product
        diversity_bands: names for the diversity raster bands, in band order
                         (defaults to fhd, enl1d, enl2d, cr)
        group_by: point column to break the correlation down by (default
                  'BIOME'); set to None to skip the per-group breakdown
        min_points: minimum valid points required to report a group
        plot_scale: axis standardization for the scatter plots only —
                    'zscore' | 'minmax' | 'none' (stats stay on raw values)
    Returns:
        results: DataFrame with one row per (diversity_index, group)
    '''
    wsci_file = Path(wsci_file).expanduser()
    diversity_file = Path(diversity_file).expanduser()
    test_point_dir = Path(test_point_dir).expanduser()
    save_dir = Path(save_dir).expanduser() if save_dir else test_point_dir
    save_dir.mkdir(parents=True, exist_ok=True)
    diversity_bands = diversity_bands or DIVERSITY_BANDS

    files = sorted(test_point_dir.glob('*.parquet'))
    columns = ['geometry'] + ([group_by] if group_by else [])
    gdf = dgp.read_parquet(
        files, columns=columns, gather_spatial_partitions=False
    ).compute()
    print(f'Loaded {len(gdf)} test points from {len(files)} parquet files')

    wsci = rasterio_read_locs(wsci_file, gdf)[:, 0]
    diversity = rasterio_read_locs(diversity_file, gdf)
    if diversity.shape[1] != len(diversity_bands):
        raise ValueError(
            f'diversity raster has {diversity.shape[1]} bands but '
            f'{len(diversity_bands)} band names were given: {diversity_bands}'
        )

    df = pd.DataFrame({wsci_name: wsci})
    for i, band in enumerate(diversity_bands):
        df[band] = diversity[:, i]
    if group_by:
        df[group_by] = gdf[group_by].to_numpy()
    df.to_parquet(save_dir / 'wsci_vs_diversity_sampled.parquet')

    rows = []
    for band in diversity_bands:
        rows += _correlate(df, wsci_name, band, save_dir, group_by,
                            min_points, scale=plot_scale)
    results = pd.DataFrame(rows)
    results.to_csv(save_dir / 'wsci_vs_diversity_correlation.csv', index=False)
    return results


if __name__ == '__main__':
    wsci_file = '~/data/gvs/products/wsci/wsci.tif'
    diversity_file = '~/data/gvs/products/diversity/diversity.tif'
    test_point_dir = '~/data/gvs/evaluation/test_points'
    save_dir = '~/data/gvs/evaluation/on_wsci'
    eval_on_wsci(wsci_file=wsci_file,
                 diversity_file=diversity_file,
                 test_point_dir=test_point_dir,
                 save_dir=save_dir)
