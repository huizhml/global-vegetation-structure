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
from evaluation.utils import ProgressMonitor
import dask
from dask.diagnostics import ProgressBar

MAX_HEIGHT = 1000.0
N_BINS = 20
BIN_WIDTH = MAX_HEIGHT / N_BINS
NODATA_IN = 32767
NODATA_OUT = -9999.0
GEDI_META_COLS =['digital_elevation_model', 'digital_elevation_model_srtm', 'pft_class', 'sensitivity', 'beam',
                 'elevation_bias_flag', 'energy_total', 'landsat_treecover', 'landsat_water_persistence', 
                 'leaf_off_doy', 'leaf_off_flag', 'leaf_on_doy', 'leaf_on_cycle', 'modis_nonvegetated', 
                 'modis_nonvegetated_sd', 'modis_treecover', 'modis_treecover_sd', 'num_detectedmodes',
                 'solar_azimuth', 'solar_elevation', 'surface_flag', 'urban_focal_window_size', 'urban_proportion', 'slope', 'lc', 
                 'geometry',
                 'BIOME', 'ECO_NAME']
stac_collection_dir = '~/data/gvs/products/gvsm_stac_catalog/vsm_local'


def pixel_diversity_indices(rhs, bin_width=5, max_height=None):
    """
    Compute per-pixel FHD using a simple histogram approach.
    """
    if isinstance(rhs, pd.Series):
        rhs = rhs.values
    if max_height is None:
        max_height = MAX_HEIGHT
    rhs_arr = np.asarray(rhs, dtype=np.float32)
    rhs_arr = rhs_arr[np.isfinite(rhs_arr)]
    n_bins = int(MAX_HEIGHT / bin_width)
    hist, bins = np.histogram(rhs_arr, bins=n_bins, range=(0, MAX_HEIGHT)) # negative values are ignored
    p = hist / hist.sum()
    mask = p > 0
    fhd = -np.sum(p[mask] * np.log(p[mask])).astype(np.float32)
    enl1d = np.exp(fhd)
    enl2d = np.float32(1.0 / np.sum(p[mask] ** 2))
    if rhs[98] <= 0:
        cr = 0
    else:
        cr = (rhs[98] - rhs[25])/rhs[98]
    
    return fhd, enl1d, enl2d, cr

def _chunk_diversity(tile, bin_width=50):
    """
    Vectorized Shannon entropy for a single spatial chunk.

    Parameters
    ----------
    tile : ndarray, shape (101, rows, cols)

    Returns
    -------
    out : ndarray, shape (rows, cols), float32
    """
    n_bands, n_rows, n_cols = tile.shape
    n_pixels = n_rows * n_cols
    n_bins = int(MAX_HEIGHT / bin_width)
    valid = np.isfinite(tile) & (tile != NODATA_IN) & (tile > 0)
    nodata_mask = valid.sum(axis=0) == 0

    tile_clean = np.where(valid, tile, 0.0)
    bin_idx = np.clip((tile_clean / bin_width).astype(np.int32), 0, n_bins - 1)
    bin_idx = np.where(valid, bin_idx, -1)

    bin_flat = bin_idx.reshape(n_bands, n_pixels)
    pixel_indices = np.broadcast_to(
        np.arange(n_pixels)[np.newaxis, :], (n_bands, n_pixels)
    ) # (101, n_pixels), each row is the pixel index for the corresponding band, e.g, 0,1,2,3,..., 512*512-1

    hist = np.zeros((n_pixels, n_bins), dtype=np.float32)
    flat_valid = bin_flat != -1 #(101, 512*512)
    np.add.at(hist, (pixel_indices[flat_valid], bin_flat[flat_valid]), 1.0) #

    total = hist.sum(axis=-1, keepdims=True)
    p = hist / total # (n_pixels, n_bins)
    log_p = np.where(p > 0, np.log(p), 0.0)
    entropy = -np.sum(p * log_p, axis=-1).astype(np.float32)
    enl1d = np.exp(entropy)
    enl2d = (1/ (p**2).sum(axis=-1)).astype(np.float32) # 2D ENL

    entropy = entropy.reshape(n_rows, n_cols)
    enl1d = enl1d.reshape(n_rows, n_cols)
    enl2d = enl2d.reshape(n_rows, n_cols)

    entropy[nodata_mask] = NODATA_OUT
    enl1d[nodata_mask] = NODATA_OUT
    enl2d[nodata_mask] = NODATA_OUT
    cr = (tile[98] - tile[25])/(tile[98] + 1e-6)
    cr[nodata_mask] = NODATA_OUT
    return entropy, enl1d, enl2d, cr


def _process_tile(args):
    """
    Worker function for ProcessPoolExecutor.
    Each worker opens its own file handle (required for multiprocessing).
    Reads all 101 bands for one window in a single call, computes entropy.
    """
    vrt_path, col_off, row_off, w, h, bin_width = args
    with rasterio.open(vrt_path, "r") as src:
        window = Window(col_off, row_off, w, h)
        tile = src.read(window=window).astype(np.float32)  # (101, h, w)
    ent, enl1d, enl2d, cr = _chunk_diversity(tile, bin_width=bin_width)
    return ent, enl1d, enl2d, cr, col_off, row_off, w, h


def compute_entropy(output_dir, tile_id, year, vrt_path=None,
                         chunk_size=512, max_workers=8, bin_width=50, **kwargs):
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
    if output_path.exists():
        return

    # Create VRT if not provided
    if vrt_path is None:
        tile_dir = f"~/data/gvs/predictions/{year}/original/tiles/cog/{tile_id}"
        vrt_path = f"~/data/gvs/predictions/{year}/original/vrt/{tile_id}_Q1.vrt"
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
            # ent, enl1d, enl2d = _process_tile((vrt_path, col_off, row_off, w, h, bin_width))
            work_items.append((vrt_path, col_off, row_off, w, h, bin_width))

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


def create_vrt(tile_dir, vrt_path, q_idx="1"):
    tile_dir = Path(tile_dir).expanduser()
    vrt_path = Path(vrt_path).expanduser()
    vrt_path.parent.mkdir(parents=True, exist_ok=True)

    tile_id = tile_dir.stem
    files = sorted(
        glob.glob(f"{tile_dir}/RH*_Q{q_idx}.tif"),
        key=lambda x: int(x.split("RH")[-1].split("_")[0]),
    )
    files = files[:100]
    print(f"Creating VRT for {tile_id} with {len(files)} files")
    assert len(files) == 100, f"Expected 100 files, found {len(files)}"

    vrt_options = gdal.BuildVRTOptions(separate=True)
    vrt = gdal.BuildVRT(str(vrt_path), files, options=vrt_options)

    for i in range(1, len(files) + 1):
        band = vrt.GetRasterBand(i)
        band.SetDescription(f"RH{i - 1}")

    vrt.FlushCache()
    vrt = None
    return vrt_path


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
            tile_dir = f"~/data/gvs/predictions/{year}/original/tiles/cog/{tile_id}"
            vrt_path = f"~/data/gvs/predictions/{year}/original/vrt/{tile_id}_Q1.vrt"
            create_vrt(tile_dir, vrt_path)
        vrt_path = str(Path(vrt_path).expanduser())
        #TODO: ...


def cal_diversity_indices(save_dir, gedi_ours_dir, bin_width=5, year=2020, **kwargs):
    '''Evaluate the diversity indices against the GEDI reference points on the test set'''
    import os
    save_dir = Path(f'{save_dir}/bin_width_{bin_width}m/{year}').expanduser()
    save_dir.mkdir(parents=True, exist_ok=True)
    gedi_ours_dir = Path(gedi_ours_dir).expanduser()
    gedi_ours_dir.mkdir(parents=True, exist_ok=True)
    gedi_ours_files = sorted(gedi_ours_dir.glob('*.parquet'))
    gedi_ref_cols = [f'rh{i}' for i in range(101)]
    ours_cols = [f'RH{i}_Q1_raw' for i in range(101)]
    
    def _tile_level_indices(gedi_ours_file: Path):
        tile_id = gedi_ours_file.stem
        # if (save_dir / f'{tile_id}.parquet').exists():
        #     return None
        gedi_ours = gpd.read_parquet(gedi_ours_file)
        gedi_ours = gedi_ours.dropna(subset=gedi_ref_cols + ours_cols)
        mask = (gedi_ours['pft_class'].isin(np.arange(1,11)) & gedi_ours['sensitivity'] >= 0.95)
        gedi_ours = gedi_ours[mask]
        if gedi_ours.empty:
            if (save_dir / f'{tile_id}.parquet').exists():
                os.remove(save_dir / f'{tile_id}.parquet')
            return None
        
        indices_gedi = gedi_ours[gedi_ref_cols].apply(lambda x: pixel_diversity_indices(x, bin_width=bin_width), axis=1)
        indices_gedi = pd.DataFrame(indices_gedi.tolist(), index=indices_gedi.index, columns=['fhd_gedi', 'enl1d_gedi', 'enl2d_gedi', 'cr_gedi'])
        indices_ours = gedi_ours[ours_cols].apply(lambda x: pixel_diversity_indices(x, bin_width=5), axis=1)
        indices_ours = pd.DataFrame(indices_ours.tolist(), index=indices_ours.index, columns=['fhd_ours', 'enl1d_ours', 'enl2d_ours', 'cr_ours'])
        df = pd.concat([indices_gedi, indices_ours, gedi_ours[GEDI_META_COLS]], axis=1)
        df = gpd.GeoDataFrame(df, geometry='geometry', crs="EPSG:4326")
        df.to_parquet(save_dir / f'{tile_id}.parquet')
        # import ipdb; ipdb.set_trace()
        # N = len(indices_gedi)
        # metrics = {}
        # for col in indices_gedi.columns: # fhd, enl1d, enl2d, cr
        #     diff = indices_gedi[col] - indices_ours[col]
        #     metrics[col] = {
        #         'r2': np.corrcoef(indices_gedi[col], indices_ours[col])[0, 1],
        #         'rmse': np.sqrt(np.mean(diff**2)),
        #         'mae': np.mean(np.abs(diff)),
        #         'me': np.mean(diff),
        #         'n': N,
        #     }
        # metrics_df = pd.DataFrame(metrics).T
        # all_metrics = pd.concat({tile_id: metrics_df}, names=['tile', 'metric'])
        # return all_metrics
    
    tasks = []
    # gedi_ours_files = [ f for f in gedi_ours_files if f.stem=='32QND']
    for gedi_ours_file in gedi_ours_files:
        # _tile_level_indices(gedi_ours_file)
        tasks.append(dask.delayed(_tile_level_indices)(gedi_ours_file))
    with ProgressBar():
        dask.compute(*tasks)
    # all_metrics = pd.concat([r for r in results if r is not None])
    # all_metrics.to_parquet(save_dir / f'all_metrics_by_{group_by}_bin_{bin_width}m.parquet')
        

if __name__ == "__main__":
    save_dir = '/projects/dereeco/data/gvs/evaluation/with_gedi_on_diversity_indices/indices_by_tile/bin_width_5m/2020'
    gedi_ours_dir = '/projects/dereeco/data/gvs/gedi/veg_sensitivity_gt0p95/subset_test/original_with_sota_chms_biome_and_ours_full/2020'
    cal_diversity_indices(save_dir, gedi_ours_dir)
    # tile_id = '36NTF'
    # year = 2020
    # vrt_path = f"~/data/gvs/predictions/{year}/original/vrt/{tile_id}_Q1.vrt"
    # output_dir = f"~/data/gvs/products/profile_entropy/{year}/tiles/geotiff"

    # # Create VRT if needed
    # tile_dir = f"~/data/gvs/predictions/{year}/original/tiles/cog/{tile_id}"
    # vrt_resolved = Path(vrt_path).expanduser()
    # if not vrt_resolved.exists():
    #     create_vrt(tile_dir, vrt_path)

    # start = time.time()
    # compute_entropy(
    #     output_dir=output_dir,
    #     tile_id=tile_id,
    #     year=year,
    #     vrt_path=vrt_path,
    #     chunk_size=512,
    #     max_workers=8,
    #     bin_width=50,
    # )
    # elapsed = time.time() - start
    # print(f"Time taken: {elapsed:.2f} seconds")
    # # vertical_profile_per_biome(
    # #     points_file='/projects/dereeco/data/gvs/analysis/typical_forests/typical_forests.zip',
    # #     save_dir='/projects/dereeco/data/gvs/analysis/typical_forests/vertical_profile_gedi_ref',
    # #     s2_grid_file='/projects/dereeco/data/gvs/state/s2_tiles_with_growing_months.parquet',
    # #     gedi_ref_dir='/projects/dereeco/data/gvs/gedi/veg_sensitivity_gt0p95/all_valid/2020',
    # #     max_distance=1000
    # # )