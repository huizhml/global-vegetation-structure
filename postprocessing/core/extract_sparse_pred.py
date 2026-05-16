import os
from osgeo import gdal
from typing import List
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
import dask.dataframe as dd
from rasterio.crs import CRS


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
        # df_ecoregions.to_parquet(save_dir / f'{tile_id}.parquet')

    # all_tiles = ['48RWN']
    tasks = []
    for tile_id in all_tiles:
        tasks.append(exe_one_tile(tile_id))
    with ProgressBar():
        res = dask.compute(*tasks)
