import wandb
import os
import ee
import h5py
import numpy as np
import dask
import dask.dataframe as dd
import dask_geopandas as dgp
from dask.utils import natural_sort_key
from shapely import wkt
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field
from hydra.core.config_store import ConfigStore
import hydra
from retry import retry
import requests
from io import StringIO
from shapely.geometry import shape
import json
import geopandas as gpd
import rasterio
import matplotlib.pyplot as plt
from rasterio.warp import transform
from dask.diagnostics import ProgressBar
from utils import get_geom_for_countries




def compare_result_precision(run_id, corrected=False):
    suffix = '_corrected' if corrected else ''
    ddf_ours = pd.read_parquet(f'output/canopy_height_predictions_{run_id}{suffix}.parquet')
    ddf_ours = ddf_ours.astype(float)
    ddf_ours = ddf_ours[(ddf_ours['slope_mask']==1) & (ddf_ours['veg_mask']==1)]
    columns = ddf_ours.columns
    pred_cols = [c for c in columns if c.startswith('RH') and '_GEDI' not in c]
    gedi_cols = [c for c in columns if c.startswith('RH') and '_GEDI' in c]
    gedi_ref = ddf_ours[gedi_cols].rename(columns=lambda x: x.replace('_GEDI', ''))
    comp_dfs = []
    for round_precision in range(0,3):
        preds = ddf_ours[pred_cols].round(round_precision)
        diff = preds - gedi_ref
        rmse = ((diff)**2).mean()**0.5
        mae = diff.abs().mean()
        me = diff.mean()
        df = pd.DataFrame({'RMSE': rmse, 'MAE': mae, 'ME': me}, index=pred_cols)
        # unstack the index to make grouped columns
        df_unstacked = df.unstack(level=1)
        df_unstacked = pd.DataFrame(df_unstacked).T
        comp_dfs.append(df_unstacked)
    comp_dfs = pd.concat(comp_dfs, keys=[f'precision_{i}' for i in range(0,3)])
    comp_dfs.to_csv(f'output/evaluation/compare_round_precision_val_{run_id}{suffix}.csv')
        

def compare_with_sota_maps_eu(sota_chm_dir, pred_df_fp: str, countries_file: str=None):
    from utils import get_geom_for_countries
    countries_df = get_geom_for_countries(countries_file)
    
    sota_chm_df_fp = Path(sota_chm_dir).expanduser()
    sota_chm_df = dgp.read_parquet(sota_chm_df_fp, gather_spatial_partitions=False)
    sota_chm_df = sota_chm_df.compute()
    sota_chm_df = sota_chm_df[sota_chm_df.intersects(countries_df.union_all())]
    
    pred_df_fp = Path(pred_df_fp).expanduser()
    pred_df = pd.read_parquet(pred_df_fp, columns=['RH95_1', 'RH98_1', 'RH100_1', 'RH95_GEDI', 'RH98_GEDI', 'RH100_GEDI', 'slope_mask', 'veg_mask', 'wc', 'slope', 'lon', 'lat', 'BIOME'])
    pred_df[['lat', 'lon']] = pred_df[['lat', 'lon']].astype(float)
    pred_df = gpd.GeoDataFrame(pred_df, geometry=gpd.points_from_xy(pred_df.lon, pred_df.lat, crs="EPSG:4326"))
    pred_df = pred_df[pred_df.intersects(countries_df.union_all())]
    
    import ipdb; ipdb.set_trace()
    pred_df = pred_df.sjoin(sota_chm_df, how='left', predicate='intersects')
    import ipdb; ipdb.set_trace()
    comp_dfs = []
    for product, name in [('RH95_1', 'RH95_GEDI'), ('RH98_1', 'RH98_GEDI'), ('RH100_1', 'RH100_GEDI')]:
        rmse = ((pred_df[product] - pred_df[name])**2).mean()**0.5
        mae = (pred_df[product] - pred_df[name]).abs().mean()
        me = (pred_df[product] - pred_df[name]).mean()
        df = pd.DataFrame({'RMSE': rmse, 'MAE': mae, 'ME': me}, index=[product])
        comp_dfs.append(df)

def compare_with_sota_maps(sota_chm_df_fp, run_id, corrected=False):
    # df_dir = Path(df_dir).expanduser()

    # ddf = dd.read_parquet(f'{df_dir}/*.parquet', index=False)
    # ddf = ddf.compute()
    sota_chm_df_fp = Path(sota_chm_df_fp).expanduser()
    ddf = pd.read_parquet(sota_chm_df_fp)
    suffix = '_corrected' if corrected else ''
    ddf_ours = pd.read_parquet(f'output/canopy_height_predictions_{run_id}{suffix}.parquet')
    ddf_ours = ddf_ours.rename(columns=lambda x: x+'_ours' if x.startswith('RH') and '_' not in x else x)
    ddf[['RH95_ours', 'RH98_ours', 'RH100_ours', 'slope_mask', 'veg_mask']] = ddf_ours[['RH95_ours', 'RH98_ours', 'RH100_ours', 'slope_mask', 'veg_mask']]
    
    ddf = ddf.dropna(subset=['RH95_META', 'RH95_UMD', 'RH98_ETH', 'RH100_UM'])
    comp_dfs = []
    for filter in ['Base', 'slope', 'veg']:
        if filter == 'Base':
            ddf = ddf
        else:
            ddf = ddf[ddf[f'{filter}_mask']==1]
        rmse = []
        mae = []
        me = []
        for product, name in [('RH95_UMD', 'RH95_GEDI'), ('RH98_ETH', 'RH98_GEDI'), ('RH100_UM', 'RH100_GEDI'), ('RH95_META', 'RH95_GEDI')]:
            rmse.append(((ddf[product] - ddf[name])**2).mean()**0.5)
            mae.append((ddf[product] - ddf[name]).abs().mean())
            me.append((ddf[product] - ddf[name]).mean())
        for name in ['RH95', 'RH98', 'RH100']:
            rmse.append(((ddf[f'{name}_ours'] - ddf[f'{name}_GEDI'])**2).mean()**0.5)
            mae.append((ddf[f'{name}_ours'] - ddf[f'{name}_GEDI']).abs().mean())
            me.append((ddf[f'{name}_ours'] - ddf[f'{name}_GEDI']).mean())
        
        df = pd.DataFrame({'RMSE': rmse, 'MAE': mae, 'ME': me}, index=['UMD', 'ETH', 'UM', 'META', 'OURS(RH95)', 'OURS(RH98)', 'OURS(RH100)'])
        comp_dfs.append(df)

    comp_dfs = pd.concat(comp_dfs, axis=1)
    comp_dfs.to_csv(f'output/evaluation/comparison_metrics_{run_id}{suffix}.csv')



class Evaluation:
    """
    Evaluation class for comparing the performance of the model with the sota maps.
    Parameters:
        sota_chm_df_fp: Path to the sota maps dataframe
        run_id: Run id of the model
        corrected: Whether the model is corrected
    """
    def __init__(self, sota_chm_df_fp, run_id, corrected):
        self.sota_chm_df_fp = Path(sota_chm_df_fp).expanduser()
        self.run_id = run_id
        self.corrected = corrected
        self.sota_chm_df = pd.read_parquet(self.sota_chm_df_fp)
        self.ddf_ours = pd.read_parquet(f'output/canopy_height_predictions_{self.run_id}{self.corrected}.parquet')
        self.ddf_ours = self.ddf_ours.rename(columns=lambda x: x+'_ours' if x.startswith('RH') and '_' not in x else x)

    def extract_pred_from_big_tile(self, tile_id):
        """
        Extract the predictions from the big tile prediction.
        Parameters:
            tile_id: Tile id of the big tile
        """
        sota_chm_df = gpd.read_parquet(self.sota_chm_dir / f'{tile_id}.parquet')
        lon = sota_chm_df.geometry.x.values
        lat = sota_chm_df.geometry.y.values
        preds = []
        for rh_idx in [95, 98, 100]:
            pred_fp = Path(f'~/data/gvs/deploy/predictions_2020/{tile_id}_cog/RH{rh_idx}_Q1.cog.tif').expanduser()
            with rasterio.open(pred_fp) as src:
                xs, ys = transform('EPSG:4326', src.crs, lon, lat)
                coords = list(zip(xs, ys))
                rhs = list(rasterio.sample.sample_gen(src, coords))
                preds.append(np.concatenate(rhs, axis=0)[:, None])
                nodata = src.nodata
        preds = np.concatenate(preds, axis=1) # (n_points, 3)
        
        # add uncorrected predictions to sota_chm_df
        sota_chm_df[['RH95_ours_raw', 'RH98_ours_raw', 'RH100_ours_raw']] = preds
        
        # add corrected predictions to sota_chm_df


@dataclass
class MyConfig:
    sota_chm_df_fp: str = '~/data/gvs/evaluation/sota_chm_val_with_gedi_biome.parquet'
    run_id: str = ''
    data_name: str = 'test'
    output_dir: str = 'output'
    task: str = 'aggregate_gedi_by_biome'
    corrected: bool = False

cs = ConfigStore.instance()
cs.store(name="my_config", node=MyConfig)


@hydra.main(config_name="my_config", version_base="1.2")
def main(cfg):
    if cfg.task == 'compare_result_precision':
        compare_result_precision(cfg.run_id, cfg.corrected)
    elif cfg.task == 'add_biome':
        add_biome(val_df_fp=f'~/data/gvs/train_subsets/{cfg.data_name}_filtered_v1.parquet', cal_pred_fp=cal_pred_fp)
    elif cfg.task == 'compare_with_sota_maps':
        compare_with_sota_maps(cfg.sota_chm_df_fp, cfg.run_id, cfg.corrected)
    elif cfg.task == 'compare_with_sota_maps_eu':
        compare_with_sota_maps_eu(cfg.sota_chm_df_fp, cfg.pred_df_fp, cfg.countries_file)

    # add biome for sota maps
    # add_biome(val_df_fp='~/data/gvs/train_subsets/val_filtered_v1.parquet', sota_chm_df_dir='~/data/gvs/evaluation/existing_canopy_height_pred_val_with_gedi')
    # add biome for cal/test predictions
    cal_pred_fp = f'~/data/gvs/uncertainty/canopy_height_predictions_{cfg.run_id}_corrected.parquet'
    add_biome(val_df_fp=f'~/data/gvs/train_subsets/{cfg.data_name}_filtered_v1.parquet', cal_pred_fp=cal_pred_fp)

if __name__ == '__main__':
    # from dask.distributed import Client, LocalCluster
    # cluster = LocalCluster(n_workers=8, dashboard_address=':38787')
    # client = Client(cluster)

    index_dir = '~/data/gvs/split_test0.1_cal0.1_val0.1_seed42_v1/index_table_test'
    save_dir = '~/data/gvs/evaluation/existing_canopy_height_pred_test'
    canopy_height_eth = ee.Image('users/nlang/ETH_GlobalCanopyHeight_2020_10m_v1').rename('RH98_ETH')
    canopy_height_umd = ee.ImageCollection("users/potapovpeter/GEDI_V27")
    canopy_height_meta = ee.ImageCollection("projects/meta-forest-monitoring-okw37/assets/CanopyHeight")
    canopy_height_um = ee.ImageCollection('projects/worldwidemap/assets/canopyheight2020')

    # sample_canopy_height_maps(index_dir, save_dir)
    compare_with_sota_maps_eu(save_dir, '~/data/gvs/uncertainty/rh_predictions_test.parquet', '~/data/gvs/deploy/EU_results/countries_list.txt')
    h5_dir = '~/data/gvs/split_test0.1_cal0.1_val0.1_seed42_v1/h5_partitions_val'
    # get_gedi_rhs(h5_dir, save_dir)
    # for file in Path('~/data/gvs/evaluation/existing_canopy_height_pred_val_with_gedi').expanduser().glob('*.parquet'):
    #     df = pd.read_parquet(file)
    #     zone = file.stem
    #     if 'RH95_UMD' not in df.columns:
    #         try:
    #             file2 = Path(f'~/data/gvs/evaluation/existing_canopy_height_pred_val/{zone}.parquet').expanduser()
    #             df_ = pd.read_parquet(file2)
    #             df.loc[:, 'RH95_UMD'] = df_['RH95_UMD']
    #             df.to_parquet(file)
    #         except:
    #             print(f'Failed to update {zone}')
    #             os.remove(file2)
    #             os.remove(file)
        
    # main()

    
