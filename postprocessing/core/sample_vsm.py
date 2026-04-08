import os
from osgeo import gdal
from typing import List, Union
import geopandas as gpd
import pandas as pd
from dataclasses import dataclass
import hydra
from hydra.core.config_store import ConfigStore
from pathlib import Path
import glob
import h5py
import numpy as np
import rasterio
import time
import matplotlib.pyplot as plt
import dask
from dask.diagnostics import ProgressBar
import pystac
import stackstac
from rasterio.io import MemoryFile
from rasterio.warp import transform
from rasterio.transform import rowcol
import xarray as xr
import numpy as np
from download.core.constants import gedi_attr_dtype
from download.core.utils import check_unfinished_files
import dask.dataframe as dd

NODATA = np.int16(32767)
# ------------------------------------------------------------
#  Helpers

def _resolve_vsm_path(stac_col_dir: Path, tile_id: str, year: int, return_stac_item: bool = False):
    '''
    Resolve the VSM path for the tile
    Parameters:
        tile_id: id of the tile
        year: year of the tile
    Returns:
        vsm_dir: path to the VSM directory for a given tile
    '''
    if not (stac_col_dir / f'{tile_id}_{year}').exists():
        print(f'{tile_id} not in stac collection')
        return    
    stac_item = pystac.Item.from_file(str(stac_col_dir / f'{tile_id}_{year}/{tile_id}_{year}.json'))
    vsm_dir = Path(stac_item.assets[f'RH98_Q1'].href.replace('file://', '')).parent
    if return_stac_item:
        return stac_item
    return vsm_dir

def _sample_tile_points_loop(pred_dir: Path, out_file: Path, loc_file: Path, rh_idxs: list[int] = [98], q_idxs: list[int] = [1]):
    '''
    Extract the RH profile from the tif files
    Args:
        out_file: path to save the dataframe
        pred_dir: path to the prediction directory
        loc_file: path to the location file
        rh_idxs: list of RH indices
        q_idxs: list of Q indices
    Returns:
        df: dataframe with the predicted RH values
    '''
    # if out_file.exists():
    #     return
    
    loc_df = gpd.read_parquet(loc_file)
    loc_df = loc_df.to_crs(epsg=4326)
    lon = loc_df.geometry.x.values
    lat = loc_df.geometry.y.values
    
    with rasterio.open(pred_dir / f'RH{rh_idxs[0]}_Q{q_idxs[0]}.tif') as src:
        xs, ys = transform('EPSG:4326', src.crs, lon, lat)
        coords = list(zip(xs, ys))
        nodata = src.nodata
    
    n_points = len(lon)
    n_rhs = len(rh_idxs)
    n_q = len(q_idxs)
    preds = np.full((n_points, n_rhs,  n_q), np.nan, dtype=np.float32)
    for i, rh_idx in enumerate(rh_idxs):
        for j, q_idx in enumerate(q_idxs):
            with rasterio.open(pred_dir / f'RH{rh_idx}_Q{q_idx}.tif') as src:
                # xs, ys = transform('EPSG:4326', src.crs, lon, lat)
                # coords = list(zip(xs, ys))
                rh = list(rasterio.sample.sample_gen(src, coords))
                pred = np.concatenate(rh, axis=0) # (n_points, )
                preds[:, i, j] = pred  # (n_points, n_rhs, n_q)
            
    preds = np.where(preds == nodata, np.nan, preds) # mask nodata (non-vegetation) for pred and ref
    preds = preds.reshape(n_points, -1)
    vsm_cols = [f'RH{i}_Q{j}' for i in rh_idxs for j in q_idxs]
    _df = pd.DataFrame(preds/10, index=loc_df.index, columns=vsm_cols)
    df = loc_df.join(_df)
    df = gpd.GeoDataFrame(df, geometry='geometry', crs="EPSG:4326")
    df.to_parquet(out_file)
    return df

def _sample_tile_points_xarray(loc_file: Path, stac_item: pystac.Item, out_file: Path,  rh_idxs: list[int] = [98], q_idxs: list[int] = [1]):
    '''
    Extract the RH profile from the tif files
    NOTE: single tile sampling is faster than rasterio windowed sampling, but slower when it comes to large number of tiles.
    '''
    if out_file.exists():
        return
    loc_df = gpd.read_parquet(loc_file)
    loc_df = loc_df.to_crs(epsg=stac_item.properties['proj:epsg'])
    n_points = len(loc_df)
    target_x = xr.DataArray(loc_df.geometry.x.values, dims="points")
    target_y = xr.DataArray(loc_df.geometry.y.values, dims="points")
    
    assets = [f'RH{i}_Q{j}' for i in rh_idxs for j in q_idxs]
    da = stackstac.stack([stac_item], assets=assets, epsg=stac_item.properties['proj:epsg'], resolution=10, dtype='int16',rescale=False, fill_value=np.int16(32767))
    preds = da.sel(x=target_x, y=target_y, method='nearest')
    
    
    preds = xr.where(preds == 32767, np.nan, preds)
    # preds = preds.values # (1, n_rhs*n_q, n_points)
    preds = preds.data.compute()
    preds = preds.reshape(-1, n_points).T # (n_points, n_rhs*n_q)
    vsm_cols = [f'RH{i}_Q{j}' for i in rh_idxs for j in q_idxs]
    _df = pd.DataFrame(preds/10., index=loc_df.index, columns=vsm_cols)
    df = loc_df.join(_df)
    df = gpd.GeoDataFrame(df, geometry='geometry', crs=stac_item.properties['proj:epsg'])
    df = df.to_crs(epsg=4326)
    df.to_parquet(out_file)
    return df

def _sample_tile_points(loc_file: Path, pred_dir: Path, out_file: Path, rh_idxs: list[int] = [98], q_idxs: list[int] = [1]):
    '''
    Extract the RH profile from the tif files
    '''
    if out_file.exists():
        return
    loc_df = gpd.read_parquet(loc_file)
    loc_df = loc_df.to_crs(epsg=4326)
    n_points = len(loc_df)
    n_rhs = len(rh_idxs)
    n_q = len(q_idxs)
    lon = loc_df.geometry.x.values
    lat = loc_df.geometry.y.values

    with rasterio.open(pred_dir / f'RH{rh_idxs[0]}_Q{q_idxs[0]}.tif') as src:
        xs, ys = transform('EPSG:4326', src.crs, lon, lat)
        rows, cols = rasterio.transform.rowcol(src.transform, xs, ys)
        rows = np.array(rows)
        cols = np.array(cols)
        nodata = src.nodata
        H, W = src.height, src.width

    valid = (rows >= 0) & (rows < H) & (cols >= 0) & (cols < W)

    r_min, r_max = rows[valid].min(), rows[valid].max()
    c_min, c_max = cols[valid].min(), cols[valid].max()
    window = rasterio.windows.Window(c_min, r_min, c_max - c_min + 1, r_max - r_min + 1)

    rows_local = rows - r_min
    cols_local = cols - c_min

    n_bands = len(rh_idxs) * len(q_idxs)
    preds = np.full((n_points, n_bands), np.nan, dtype=np.float32)

    for band_idx, (rh_idx, q_idx) in enumerate(
        [(rh, q) for rh in rh_idxs for q in q_idxs]
    ):
        with rasterio.open(pred_dir / f'RH{rh_idx}_Q{q_idx}.tif') as src:
            data = src.read(1, window=window)
        preds[valid, band_idx] = data[rows_local[valid], cols_local[valid]]

    preds = np.where(preds == nodata, np.nan, preds)
    vsm_cols = [f'RH{i}_Q{j}' for i in rh_idxs for j in q_idxs]
    _df = pd.DataFrame(preds/10., index=loc_df.index, columns=vsm_cols)
    df = loc_df.join(_df)
    df = gpd.GeoDataFrame(df, geometry='geometry', crs="EPSG:4326")
    df = df.to_crs(epsg=4326)
    df.to_parquet(out_file)
    return df

def _sample_tile_patches_xarray(loc_file: Path, stac_item: pystac.Item, out_file: Path,
                          rh_idxs: List[int] = np.arange(101), q_idxs: List[int] = [1],
                          patch_size: int = 11, chunk_size: int = 10, **kwargs):
    '''
    NOTE: single tile sampling is faster than rasterio windowed sampling, but slower when it comes to large number of tiles.
    '''
    if out_file.exists():
        return
    half = patch_size // 2
    loc_df = gpd.read_parquet(loc_file)
    loc_df = loc_df.to_crs(epsg=stac_item.properties['proj:epsg'])
    n_points = len(loc_df)

    assets = [f'RH{i}_Q{j}' for i in rh_idxs for j in q_idxs]
    n_bands = len(assets)
    patches = np.full((n_points, n_bands, patch_size, patch_size), np.nan, dtype=np.float32)

    # Get grid info from first chunk
    da_first = stackstac.stack([stac_item], assets=assets[:1],
                                epsg=stac_item.properties['proj:epsg'],
                                resolution=10, dtype='int16', rescale=False,
                                fill_value=np.int16(32767)).squeeze('time')
    x_coords = da_first.x.values
    y_coords = da_first.y.values
    H, W = len(y_coords), len(x_coords)

    cx = loc_df.geometry.x.values
    cy = loc_df.geometry.y.values

    # Nearest pixel index for each point
    col_idxs = np.abs(x_coords[np.newaxis, :] - cx[:, np.newaxis]).argmin(axis=1)
    row_idxs = np.abs(y_coords[np.newaxis, :] - cy[:, np.newaxis]).argmin(axis=1)

    # Build patch indices: (n_points, patch_size)
    offsets = np.arange(-half, half + 1)
    row_patches = row_idxs[:, np.newaxis] + offsets[np.newaxis, :]
    col_patches = col_idxs[:, np.newaxis] + offsets[np.newaxis, :]

    # Out-of-bounds mask: (n_points, patch_size, patch_size)
    oob_mask = (
        (row_patches < 0) | (row_patches >= H)
    )[:, :, np.newaxis] | (
        (col_patches < 0) | (col_patches >= W)
    )[:, np.newaxis, :]

    row_patches = np.clip(row_patches, 0, H - 1)
    col_patches = np.clip(col_patches, 0, W - 1)

    # Process bands in chunks to avoid too-many-open-files
    for band_start in range(0, n_bands, chunk_size):
        asset_chunk = assets[band_start:band_start + chunk_size]
        n_chunk = len(asset_chunk)

        da = stackstac.stack([stac_item], assets=asset_chunk,
                             epsg=stac_item.properties['proj:epsg'],
                             resolution=10, dtype='int16', rescale=False,
                             fill_value=np.int16(32767))
        da = da.squeeze('time')
        data = da.compute().values  # (n_chunk, H, W) as numpy

        # Vectorized fancy indexing on numpy array
        extracted = data[:, row_patches[:, :, np.newaxis], col_patches[:, np.newaxis, :]]
        # (n_chunk, n_points, patch_size, patch_size) -> (n_points, n_chunk, patch_size, patch_size)
        patches[:, band_start:band_start + n_chunk, :, :] = extracted.transpose(1, 0, 2, 3)

    # Mask out-of-bounds and nodata
    oob_broadcast = np.broadcast_to(oob_mask[:, np.newaxis, :, :],
                                     (n_points, n_bands, patch_size, patch_size))
    patches[oob_broadcast] = NODATA
    patches = patches.astype(np.float32)
    patches = np.where(patches == NODATA, np.nan, patches)
    patches = patches / 10.0
    np.savez(out_file, patches=patches)
    return patches


def _sample_tile_patches(loc_file: Path, pred_dir: Path, out_file: Path,
                          rh_idxs: List[int] = np.arange(101), q_idxs: List[int] = [1],
                          patch_size: int = 11, **kwargs):
    if out_file.exists():
        return
    half = patch_size // 2
    loc_df = gpd.read_parquet(loc_file)
    loc_df = loc_df.to_crs(epsg=4326)
    n_points = len(loc_df)
    lon = loc_df.geometry.x.values
    lat = loc_df.geometry.y.values

    assets = [f'RH{i}_Q{j}' for i in rh_idxs for j in q_idxs]
    n_bands = len(assets)

    with rasterio.open(pred_dir / f'RH{rh_idxs[0]}_Q{q_idxs[0]}.tif') as src:
        xs, ys = transform('EPSG:4326', src.crs, lon, lat)
        rows, cols = rasterio.transform.rowcol(src.transform, xs, ys)
        rows = np.array(rows)
        cols = np.array(cols)
        nodata = src.nodata
        H, W = src.height, src.width

    offsets = np.arange(-half, half + 1)
    row_patches = rows[:, np.newaxis] + offsets[np.newaxis, :]  # (n_points, patch_size)
    col_patches = cols[:, np.newaxis] + offsets[np.newaxis, :]

    oob_mask = (
        (row_patches < 0) | (row_patches >= H)
    )[:, :, np.newaxis] | (
        (col_patches < 0) | (col_patches >= W)
    )[:, np.newaxis, :]  # (n_points, patch_size, patch_size)

    row_patches = np.clip(row_patches, 0, H - 1)
    col_patches = np.clip(col_patches, 0, W - 1)

    # Compute bounding window across all points — read only this region
    r_min, r_max = row_patches.min(), row_patches.max()
    c_min, c_max = col_patches.min(), col_patches.max()
    window = rasterio.windows.Window(c_min, r_min, c_max - c_min + 1, r_max - r_min + 1)

    # Shift indices to be relative to the window
    row_patches_local = row_patches - r_min
    col_patches_local = col_patches - c_min

    patches = np.full((n_points, n_bands, patch_size, patch_size), np.nan, dtype=np.float32)

    for band_idx, (rh_idx, q_idx) in enumerate(
        [(rh, q) for rh in rh_idxs for q in q_idxs]
    ):
        with rasterio.open(pred_dir / f'RH{rh_idx}_Q{q_idx}.tif') as src:
            data = src.read(1, window=window)  # only the bounding box of all points

        # Vectorized extraction
        extracted = data[row_patches_local[:, :, np.newaxis], col_patches_local[:, np.newaxis, :]]
        patches[:, band_idx, :, :] = extracted

    # Mask out-of-bounds and nodata
    patches[np.broadcast_to(oob_mask[:, np.newaxis, :, :], patches.shape)] = np.nan
    patches = np.where(patches == nodata, np.nan, patches) / 10.0
    # NOTE:
    #  np.nanmean((loc_df.loc[:, [f'rh{i}' for i in range(101)]].values -  patches[:, :, 5,5])) gives 0.23693679634738254 for 32MNE
    #  which is the same as the one given by the _sample_tile_points method
    da = xr.DataArray(patches, dims=['points', 'bands', 'y', 'x'], coords=[range(n_points), range(n_bands), range(patch_size), range(patch_size)])
    da = da.assign_coords(points=loc_df.ID, bands=assets, 
                     lat=('points', loc_df.geometry.y), lon=('points', loc_df.geometry.x), 
                     flag=('points', loc_df.flag.values), land_use_id=('points', loc_df.Land_use_ID.values),
                     rowid=('points', loc_df.rowid.values))
    da.name = 'data'
    da.to_netcdf(out_file.with_suffix('.h5'), format='NETCDF4', engine='h5netcdf')

# ------------------------------------------------------------
#  Main functions
# ------------------------------------------------------------

def add_biome(parq_dir: str, biome_file: str, save_dir: str, **kwargs):
    '''
    Add biome info to the dataframe
    '''
    parq_dir = Path(parq_dir).expanduser()
    parquet_files = list(parq_dir.glob('*.parquet'))
    biome_file = Path(biome_file).expanduser()
    ecoregions = gpd.read_file(biome_file)
    save_dir = Path(save_dir).expanduser()
    save_dir.mkdir(parents=True, exist_ok=True)
    
    
    def _sjoin(parquet_file: Path):
        df = gpd.read_parquet(parquet_file)
        df = df.set_crs(epsg=4326)
        df = df.to_crs(epsg=3857)
        df = gpd.sjoin(df, ecoregions, how='left', predicate='intersects')
        df = df.drop(columns=['index_right'])
        df.to_parquet(save_dir / f'{parquet_file.stem}.parquet')
        return df
    
    tasks = []
    # parquet_files = [f for f in parquet_files if '32MQE' in f.stem]
    for parquet_file in parquet_files:
        if (save_dir / f'{parquet_file.stem}.parquet').exists():
            continue
        # _sjoin(parquet_file)
        tasks.append(dask.delayed(_sjoin)(parquet_file))
    with ProgressBar():
        res = dask.compute(*tasks)
    return res
    
def sample_patches(loc_dir: str,  stac_col_dir: str, save_dir: str, year: int = 2020, patch_size: int = 11, rh_idxs: List[int] = np.arange(101), q_idxs: List[int] = [1], **kwargs):
    '''
    Parameters:
        loc_dir: path to the location directory
        stac_col_dir: path to the STAC collection directory
        save_dir: path to save the extracted patches
        **kwargs: additional arguments
    Returns:
        None
    '''
    loc_dir = Path(loc_dir).expanduser()
    stac_col_dir = Path(stac_col_dir).expanduser()
    save_dir = Path(save_dir).expanduser()
    save_dir.mkdir(parents=True, exist_ok=True)
    
    tiles_list_file = kwargs.get('tiles_list_file', None)
    if tiles_list_file is not None: # for european tiles
        with open(tiles_list_file, 'r') as f:
            all_tiles = f.read().splitlines()
    else:
        all_tiles = [tile.stem for tile in loc_dir.glob('*.parquet')]
    # all_tiles = ['32SNA']
    print(f'{len(all_tiles)} tiles have predictions')
    
    # all_tiles = ['48RWN']
    tasks = []
    for tile_id in all_tiles:
        pred_dir = dask.delayed(_resolve_vsm_path)(stac_col_dir, tile_id, year)
        loc_file = loc_dir / f'{tile_id}.parquet'
        tasks.append(dask.delayed(_sample_tile_patches)(loc_file, pred_dir, save_dir / f'{tile_id}.h5', rh_idxs=rh_idxs, q_idxs=q_idxs, patch_size=patch_size))
    with ProgressBar():
        dask.compute(*tasks)

def sample_points(
        loc_dir: str = None, 
        save_dir: str = None,
        stac_col_dir: str = None,
        year: int = 2020,
        rh_idxs: List[int] = np.arange(101),
        **kwargs):
    '''extract predictions and add biome info for conformal prediction
    '''
    loc_dir = Path(loc_dir).expanduser()
    stac_col_dir = Path(stac_col_dir).expanduser()
    save_dir = Path(save_dir).expanduser()
    save_dir.mkdir(parents=True, exist_ok=True)
    
    tiles_list_file = kwargs.get('tiles_list_file', None)
    if tiles_list_file is not None: # for european tiles
        with open(tiles_list_file, 'r') as f:
            all_tiles = f.read().splitlines()
    else:
        all_tiles = [tile.stem for tile in loc_dir.glob('*.parquet')]
    # all_tiles = ['32SNA']
    print(f'{len(all_tiles)} tiles have predictions')
    
    # all_tiles = ['48RWN']
    tasks = []
    for tile_id in all_tiles:
        pred_dir = dask.delayed(_resolve_vsm_path)(stac_col_dir, tile_id, year, return_stac_item=False)
        out_file = save_dir / f'{tile_id}.parquet'
        loc_file = loc_dir / f'{tile_id}.parquet'
        tasks.append(dask.delayed(_sample_tile_points)(loc_file, pred_dir, out_file, rh_idxs))
    with ProgressBar():
        dask.compute(*tasks)


def pair_predictions_with_gedi_ref_data(
        gedi_chm_reference_dir: str = None, year: int = None, tiles_list_file: str = None, save_dir: str = None,
        stac_collection_dir: str = None, pred_parent_dir: str = None, rh_idxs: list[int] = [98], **kwargs):
    '''
    Check the correction performance (RMSE, MAE and ME) with GEDI reference data and sota chm
    Split the correction data into 2 parts:
    - Part 1: used for correction
    - Part 2: used for evaluation
    '''
    gedi_chm_reference_dir = Path(f'{gedi_chm_reference_dir}').expanduser()
    pred_parent_dir = pred_parent_dir and Path(f'{pred_parent_dir}').expanduser()
    stac_collection_dir = stac_collection_dir and Path(f'{stac_collection_dir}').expanduser()
    
    save_dir = Path(save_dir).expanduser()
    save_dir.mkdir(parents=True, exist_ok=True)
    if tiles_list_file is not None: # for european tiles
        with open(tiles_list_file, 'r') as f:
            all_tiles = f.read().splitlines()
    else:
        all_tiles = [tile.stem for tile in gedi_chm_reference_dir.glob('*.parquet')]
        
    
    # all_tiles = ['32SNA']
    unfinished_tiles = check_unfinished_files(all_tiles, save_dir, output_format='parquet')
    print(f'{len(all_tiles)} tiles have predictions')
    print(f'{len(unfinished_tiles)} tiles need to be processed')

    ours_rh_cols = [f'RH{i}_Q1_raw' for i in rh_idxs]
    
    @dask.delayed
    def _get_tile_pred_dir(tile_id: str):
        if pred_parent_dir is not None and (pred_parent_dir/f'{tile_id}').exists():
            return pred_parent_dir / f'{tile_id}'

        if stac_collection_dir is not None and (stac_collection_dir / f'{tile_id}_{year}').exists():
            stac_item = pystac.Item.from_file(str(stac_collection_dir / f'{tile_id}_{year}/{tile_id}_{year}.json'))
            tile_pred_dir = Path(stac_item.assets[f'RH98_Q1'].href.replace('file://', '')).parent
            return tile_pred_dir
        print(f'{tile_id} not found in {pred_parent_dir}, or in stac collection')
        return None

    @dask.delayed
    def _process_tile(pred_dir: Union[Path, None]):
        if pred_dir is None:
            return None
        tile_id = pred_dir.stem
        if (save_dir / f'{tile_id}.parquet').exists():
            return
        
        gedi_chm_ref_df = gpd.read_parquet(gedi_chm_reference_dir / f'{tile_id}.parquet')
        lon = gedi_chm_ref_df.lon.values
        lat = gedi_chm_ref_df.lat.values

        preds = _sample_tile_points(pred_dir, lon, lat, rh_idxs)

        # apply bias correction for all rhs
        _df = pd.DataFrame(preds/10, index=gedi_chm_ref_df.index, columns=ours_rh_cols)
        gedi_chm_ref_df = gedi_chm_ref_df.join(_df)
        gedi_chm_ref_df.to_parquet(save_dir / f'{tile_id}.parquet')

    # all_tiles = ['35NLJ']
    tasks = []
    for tile_id in unfinished_tiles:
        pred_file = _get_tile_pred_dir(tile_id)
        tasks.append(_process_tile(pred_file))
    with ProgressBar():
        res = dask.compute(*tasks)
 
    
def extract_gedi_from_h5(h5_file: str, index_table_dir: str, save_dir: str, year: int, keep_columns: list = None):
    """
    Extract GEDI data (with, slope and lc info) from the H5 file and save it to a parquet file.
    """
    h5_file = Path(h5_file).expanduser()
    index_table_dir = Path(index_table_dir).expanduser()
    assert index_table_dir.stem == h5_file.stem.split('_')[1], f'index_table_dir {index_table_dir} and h5_file {h5_file} should come from the same split'
    save_dir = Path(save_dir).expanduser()
    save_dir.mkdir(parents=True, exist_ok=True)
    keep_columns = keep_columns or []
    index_table_fps = list(index_table_dir.glob('*.parquet'))
    rh_cols = [f'rh{i}' for i in range(101)]
    gedi_attr_cols = list(gedi_attr_dtype.keys())
    cols = rh_cols + gedi_attr_cols + ['slope', 'lc', 'lat', 'lon', 'shot_number']  + keep_columns
    dtype = {**gedi_attr_dtype, 'slope': 'float32', 'lc': 'int8', 'shot_number': 'uint64'}
    
    @dask.delayed
    def _extract(index_table_fp: str, year: int):
        df = pd.read_parquet(index_table_fp)
        tile_id = index_table_fp.stem
        save_fp = save_dir / f'{tile_id}.parquet'
        if save_fp.exists():
            print(f'{save_fp} already exists')
            return
    
        groups = df['path'].unique()
        groups = [group for group in groups if f'{year}' in group]
        if len(groups) == 0:
            print(f'No groups found for {tile_id}, maybe no data sampled for this split for this tile, or from other splits')
            return
        data_list = []
        with h5py.File(h5_file, 'r') as data:
            for group in groups:
                in_partition_idx = df[df['path'] == f'{group}']['in_partition_idx'].values
                rhs = data[f'{group}/rhs'][in_partition_idx]
                gedi_attrs = data[f'{group}/gedi_attrs'][in_partition_idx]
                slope = data[f'{group}/slope'][in_partition_idx, 7,7:8]
                lc = data[f'{group}/image'][in_partition_idx, 13, 7, 7:8]
                latlon = data[f'{group}/latlon'][in_partition_idx]
                shot_number = data[f'{group}/shot_number'][in_partition_idx][:, np.newaxis]
                index_df_keep = df[df['path'] == f'{group}'][keep_columns].values
                _data = np.concatenate([rhs, gedi_attrs, slope, lc, latlon, shot_number, index_df_keep], axis=1)
                data_list.append(_data)
            da = np.concatenate(data_list, axis=0)
            df_gedi = pd.DataFrame(da, columns=cols)
            assert df_gedi.lon.max() - df_gedi.lon.min() < 6, f'lon range is too large for tile {tile_id}, {df_gedi.lon.max()} - {df_gedi.lon.min()}'
            assert df_gedi.lat.max() - df_gedi.lat.min() < 2, f'lat range is too large for tile {tile_id}, {df_gedi.lat.max()} - {df_gedi.lat.min()}'
            df_gedi = df_gedi.astype(dtype)
            df_gedi.to_parquet(save_fp)
    unfinished_files = check_unfinished_files(index_table_fps, save_dir, output_format='parquet')
    # unfinished_files = [fp for fp in unfinished_files if '37T' in fp.stem]
    tasks = [_extract(index_table_fp, year) for index_table_fp in unfinished_files]
    dask.compute(*tasks)

    
if __name__ == '__main__':
    import time
    
    tile_id = '32MNE'
    year = 2020
    stac_item = pystac.Item.from_file(str(f'/projects/dereeco/data/gvs/products/gvsm_stac_catalog/vsm_local/{tile_id}_{year}/{tile_id}_{year}.json'))
    out_file = Path(f'~/data/gvs/gedi/veg_sensitivity_gt0p95/subset_test/original_with_sota_chms_ours_test/{year}/{tile_id}').expanduser()
    # loc_dir = '~/data/gvs/downstream_tasks/naturalness/loc_by_tile/'
    # 
    loc_dir = '/projects/dereeco/data/gvs/gedi/veg_sensitivity_gt0p95/subset_test/original_with_sota_chms/2020'
    rh_idxs = np.arange(101)
    q_idxs = [1]    
    
    stac_col_dir = '/projects/dereeco/data/gvs/products/gvsm_stac_catalog/vsm_local'
    save_dir = '~/data/gvs/downstream_tasks/naturalness/vsm_patches_ps11'
    loc_file = Path(f'{loc_dir}/{tile_id}.parquet').expanduser()
    pred_dir = Path(stac_item.assets[f'RH98_Q1'].href.replace('file://', '')).parent
    # df = sample_patches(loc_dir, stac_col_dir, save_dir, year=year, rh_idxs=rh_idxs, q_idxs=q_idxs, patch_size=11, chunk_size=2)
    t0 = time.time()
    df = _sample_tile_points(loc_file, pred_dir, out_file, rh_idxs=rh_idxs, q_idxs=q_idxs)
    t1 = time.time()
    print(f'Time taken: {t1 - t0} seconds')
    # t0 = time.time()
    
    # for i in range(10):
    #     patches = sample_points(loc_file, pred_dir, out_file, rh_idxs=rh_idxs, q_idxs=q_idxs)
    # t1 = time.time()
    # print(f'Time taken: {t1 - t0} seconds')
    