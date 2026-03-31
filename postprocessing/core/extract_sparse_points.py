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
    

def extract_pred_add_biome(
        gedi_ref_dir: str = None, 
        tiles_list_file: str = None,
        biome_file: str=None,
        save_dir: str = None,
        year: int = None, 
        vsm_dir: str = None,    
        rh_idxs: List[int] = (0, 10, 25, 50, 75, 95, 98, 100),
        **kwargs):
    '''extract predictions and add biome info for conformal prediction
    '''
    gedi_ref_dir = Path(f'{gedi_ref_dir}').expanduser()
    vsm_dir = Path(f'{vsm_dir}').expanduser()
    biome_file = Path(f'{biome_file}').expanduser()
    ecoregions = gpd.read_file(biome_file)
    
    save_dir = Path(save_dir).expanduser()
    save_dir.mkdir(parents=True, exist_ok=True)
    if tiles_list_file is not None: # for european tiles
        with open(tiles_list_file, 'r') as f:
            all_tiles = f.read().splitlines()
    else:
        all_tiles = [tile.stem for tile in gedi_ref_dir.glob('*.parquet')]
    # all_tiles = ['32SNA']
    print(f'{len(all_tiles)} tiles have predictions')
    
    ours_rh_cols = [f'RH{i}_Q{j}' for i in rh_idxs for j in range(3)]

    @dask.delayed
    def exe_one_tile(tile_id: str):
        if not (vsm_dir / f'{tile_id}').exists():
            print(f'{tile_id} not blended')
            return        
        gedi_ref_df = pd.read_parquet(gedi_ref_dir / f'{tile_id}.parquet')
        lon = gedi_ref_df.lon.values
        lat = gedi_ref_df.lat.values

        pred_fp0 = vsm_dir / f'{tile_id}/RH0_Q1.tif'
        with rasterio.open(pred_fp0) as src:
            xs, ys = transform('EPSG:4326', src.crs, lon, lat)
            coords = list(zip(xs, ys))
            nodata = src.nodata
        
        preds = []
        for rh_idx in rh_idxs:
            for q_idx in range(3):
                pred_fp = vsm_dir / f'{tile_id}/RH{rh_idx}_Q{q_idx}.tif'
                with rasterio.open(pred_fp) as src:
                    # xs, ys = transform('EPSG:4326', src.crs, lon, lat)
                    # coords = list(zip(xs, ys))
                    rh = list(rasterio.sample.sample_gen(src, coords))
                    pred = np.concatenate(rh, axis=0).reshape(-1, 1)
                    preds.append(pred)
                
        preds = np.concatenate(preds, axis=1) # (n_points, 303)
        preds = preds.astype(np.float32)
        preds = np.where(preds == nodata, np.nan, preds) # mask nodata (non-vegetation) for pred and ref
        
        # apply bias correction for all rhs
        _df = pd.DataFrame(preds/10, index=gedi_ref_df.index, columns=ours_rh_cols)
        df = gedi_ref_df.join(_df)
        df = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat), crs="EPSG:4326")
        df = df.to_crs(epsg=3857)
        df_ecoregions = gpd.sjoin(df, ecoregions, how='left', predicate='within')
        if len(df_ecoregions) != len(df):
            print(f'{tile_id} has {len(df_ecoregions)} points after sjoin, but {len(df)} points before sjoin')
            raise ValueError(f'{tile_id} has {len(df_ecoregions)} points after sjoin, but {len(df)} points before sjoin')
        df_ecoregions.to_parquet(save_dir / f'{tile_id}.parquet')

    # all_tiles = ['48RWN']
    tasks = []
    for tile_id in all_tiles:
        tasks.append(exe_one_tile(tile_id))
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

    
# ------------------------------------------------------------
#  Helpers

def _extract_rh_profile_from_tif(Pred_dir: Path, lon: np.ndarray, lat: np.ndarray, rh_idxs: list[int] = [98]):
    '''
    Extract the RH profile from the tif files
    Args:
        Pred_dir: list of tif files
        lon: longitude array
        lat: latitude array
        rh_size: number of RHs
    Returns:
        preds: array of shape (n_points, rh_size)
    '''
    
    with rasterio.open(Pred_dir / f'RH{rh_idxs[0]}_Q1.tif') as src:
        xs, ys = transform('EPSG:4326', src.crs, lon, lat)
        coords = list(zip(xs, ys))
        nodata = src.nodata
    
    preds = []
    for rh_idx in rh_idxs:
        with rasterio.open(Pred_dir / f'RH{rh_idx}_Q1.tif') as src:
            # xs, ys = transform('EPSG:4326', src.crs, lon, lat)
            # coords = list(zip(xs, ys))
            rh = list(rasterio.sample.sample_gen(src, coords))
            pred = np.concatenate(rh, axis=0).reshape(-1, 1)
            preds.append(pred)  # (n_points, 1)
            
    preds = np.concatenate(preds, axis=1)  # (n_points, rh_size)
    preds = preds.astype(np.float32)
    preds = np.where(preds == nodata, np.nan, preds) # mask nodata (non-vegetation) for pred and ref
    return preds


# ------------------------------------------------------------
#  Main functions
# ------------------------------------------------------------


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
    print(f'{len(all_tiles)} tiles have predictions')

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
        gedi_chm_ref_df = gpd.read_parquet(gedi_chm_reference_dir / f'{tile_id}.parquet')
        lon = gedi_chm_ref_df.lon.values
        lat = gedi_chm_ref_df.lat.values

        preds = _extract_rh_profile_from_tif(pred_dir, lon, lat, rh_idxs)

        # apply bias correction for all rhs
        _df = pd.DataFrame(preds/10, index=gedi_chm_ref_df.index, columns=ours_rh_cols)
        gedi_chm_ref_df = gedi_chm_ref_df.join(_df)
        gedi_chm_ref_df.to_parquet(save_dir / f'{tile_id}.parquet')

    # all_tiles = ['35NLJ']
    tasks = []
    for tile_id in all_tiles:
        pred_file = _get_tile_pred_dir(tile_id)
        tasks.append(_process_tile(pred_file))
    with ProgressBar():
        res = dask.compute(*tasks)
 
    