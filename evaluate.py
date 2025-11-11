import wandb
import os
import ee
import re
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
    test = gpd.sjoin_nearest(sota_chm_df, pred_df, how='left', distance_col='dist')
    
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
    """
    def __init__(self,prediction_dir: str, sota_chm_dir: str, gedi_ref_dir: str, **kwargs):
        self.prediction_dir = Path(prediction_dir).expanduser()
        self.sota_chm_dir = Path(sota_chm_dir).expanduser()
        self.gedi_ref_dir = Path(gedi_ref_dir).expanduser()
        self.year = self.prediction_dir.stem.split('_')[-1]

    def compare_with_sota_chms(self, sota_chm_and_ours_fp, countries_file: str=None):
        '''
        When we have our predictions, sota maps and GEDI referencesaved in the same dataframe.
        Compare the performance of the model with the sota maps.
        Args:
            sota_chm_and_ours_fp: Path to the sota maps dataframe
            countries_file (optional): Path to the countries file
        '''
        sota_chm_and_ours_df = dgp.read_parquet(sota_chm_and_ours_fp, gather_spatial_partitions=False)
        sota_chm_and_ours_df = sota_chm_and_ours_df.compute()
        sota_chm_and_ours_df = sota_chm_and_ours_df.dropna(subset=['RH95_UMD', 'RH95_META', 'RH98_ETH', 'RH100_UM'])
        if countries_file is not None:
            countries_df = get_geom_for_countries(countries_file)
            sota_chm_and_ours_df = sota_chm_and_ours_df[sota_chm_and_ours_df.intersects(countries_df.union_all())]
            postfix = '_eu'
        else:
            postfix = ''
        
        rmse, mae, me = [], [], []
        products = [('RH95_UMD', 'rh95', 1), ('RH95_META', 'rh95', 1), ('RH98_ETH', 'rh98', 1), ('RH100_UM', 'rh100', 1),
                    ('RH95_raw', 'rh95', 10), ('RH98_raw', 'rh98', 10), ('RH100_raw', 'rh100', 10),
                    ('RH95_linear_corrected', 'rh95', 10), ('RH98_linear_corrected', 'rh98', 10), ('RH100_linear_corrected', 'rh100', 10),
                    ('RH95_bias_corrected', 'rh95', 10), ('RH98_bias_corrected', 'rh98', 10), ('RH100_bias_corrected', 'rh100', 10)]
        for product, name, scale in products:
            rmse.append(((sota_chm_and_ours_df[product]/scale - sota_chm_and_ours_df[name])**2).mean()**0.5)
            mae.append((sota_chm_and_ours_df[product]/scale - sota_chm_and_ours_df[name]).abs().mean())
            me.append((sota_chm_and_ours_df[product]/scale - sota_chm_and_ours_df[name]).mean())
        df = pd.DataFrame({'RMSE': rmse, 'MAE': mae, 'ME': me}, index=[name for name, _, _ in products])
        df.to_csv(f'{Path(sota_chm_and_ours_fp).parent.parent}/correction_performance_2020_compared_with_sota_chm{postfix}.csv')

    def evaluate_eu_test_tiles(self, countries_file: str, s2_grid_file: str, test_tiles_file: str):
        countries_df = get_geom_for_countries(countries_file)
        s2_grid = gpd.read_parquet(s2_grid_file)
        tiles = s2_grid[s2_grid.intersects(countries_df.union_all())]['Name'].unique()
        test_tiles = pd.read_csv(test_tiles_file, header=None, sep=' ')
        test_tiles = test_tiles.loc[test_tiles[0].isin(tiles)][0].tolist()
        suffix = self.gedi_ref_dir.stem.split('_')[-1]
        self.save_dir = self.gedi_ref_dir.parent / f'extracted_raw_predictions_{self.year}_{suffix}'
        self.save_dir.mkdir(exist_ok=True, parents=True)
        
        if len(list(self.save_dir.glob('*.parquet'))) == 0:
            tasks = []
            for tile in test_tiles:
                tasks.append(self.extract_from_large_tile_prediction(tile))
            with ProgressBar():
                dask.compute(tasks)
        common_tiles = list(self.save_dir.glob('*.parquet'))
        common_tiles = [tile.stem for tile in common_tiles]
        rh_cols = [f'RH{i}' for i in range(101)]
        rh_cols_ref = [f'rh{i}' for i in range(101)]
        
        sota_comparison_dfs = []
        residuals_vsm = []
        for tile in common_tiles:
            gedi_ref = gpd.read_parquet(self.gedi_ref_dir / f'{tile}.parquet')
            sota_chm_df = gpd.read_parquet(self.sota_chm_dir / f'{tile}.parquet')
            ours = gpd.read_parquet(self.save_dir / f'{tile}.parquet')
            assert ours.index.equals(sota_chm_df.index)
            sota_chm_df[['RH95_raw', 'RH98_raw', 'RH100_raw']] = ours[['RH95', 'RH98', 'RH100']].values
            sota_comparison_dfs.append(sota_chm_df)
            nan_rows = ours.isna().any(axis=1)
            ours = ours[~nan_rows]
            gedi_ref = gedi_ref[~nan_rows]
            assert ours.index.equals(gedi_ref.index)
            residuals_vsm_tile = ours[rh_cols].values/10 - gedi_ref[rh_cols_ref].values
            residuals_vsm.append(residuals_vsm_tile)
        residuals_vsm = np.concatenate(residuals_vsm, axis=0) #(n_points, 101)
        sota_comparison_dfs = pd.concat(sota_comparison_dfs, axis=0)
        

        
        #### compare with sota chm residuals distribution
        sota_comparison_dfs = sota_comparison_dfs.dropna(subset=['RH95_UMD', 'RH95_META', 'RH98_ETH', 'RH100_UM', 'RH95_raw', 'RH98_raw', 'RH100_raw'])
        
        rmse, mae, me = [], [], []
        products = [('RH95_UMD', 'rh95', 1), ('RH95_META', 'rh95', 1), ('RH98_ETH', 'rh98', 1), ('RH100_UM', 'rh100', 1),
                    ('RH95_raw', 'rh95', 10), ('RH98_raw', 'rh98', 10), ('RH100_raw', 'rh100', 10)]
        for product, name, scale in products:
            rmse.append(((sota_comparison_dfs[product]/scale - sota_comparison_dfs[name])**2).mean()**0.5)
            mae.append((sota_comparison_dfs[product]/scale - sota_comparison_dfs[name]).abs().mean())
            me.append((sota_comparison_dfs[product]/scale - sota_comparison_dfs[name]).mean())
        df = pd.DataFrame({'RMSE': rmse, 'MAE': mae, 'ME': me}, index=[name for name, _, _ in products])
        df.to_csv(f'{self.save_dir.parent}/test_performance_2020_compared_with_sota_chm_eu.csv')
        #### our vsm residuals distribution
        plt.figure(figsize=(12, 2.5))
        plt.axhline(y=0, color='black', linestyle='-', linewidth=1)
        box_plot = plt.boxplot(residuals_vsm, vert=True, patch_artist=True, showfliers=False)
        for patch in box_plot['boxes']:
            patch.set_facecolor('#ccebc5')
            patch.set_edgecolor('black')
        
        plt.xlabel('Relative Height (RH0-RH100)', fontsize=12)
        plt.ylabel('Residuals [m]', fontsize=12)
        plt.xticks(ticks=np.arange(1, 102, 10), labels=[f'{i}' for i in range(0, 101, 10)])
        plt.tight_layout()
        plt.grid(True, axis='y', linestyle='--', linewidth=0.5)
        plt.savefig(f'{self.save_dir.parent}/vsm_residuals_distribution_boxplot_eu.pdf')

    
    
    @dask.delayed
    def extract_from_large_tile_prediction(self, tile_id: str):
        file = self.gedi_ref_dir / f'{tile_id}.parquet'
        if not file.exists() or (self.save_dir / f'{tile_id}.parquet').exists():
            return
        sota_chm_df = gpd.read_parquet(file)
        lon = sota_chm_df.geometry.x.values
        lat = sota_chm_df.geometry.y.values
        preds = []
        for rh_idx in range(101):
            pred_fp = self.prediction_dir / f'{tile_id}_cog/RH{rh_idx}_Q1.cog.tif'
            with rasterio.open(pred_fp) as src:
                xs, ys = transform('EPSG:4326', src.crs, lon, lat)
                coords = list(zip(xs, ys))
                nodata = src.nodata
                pred = list(rasterio.sample.sample_gen(src, coords))
                pred = np.concatenate(pred, axis=0).reshape(-1, 1) # (n_points, 1)
                preds.append(pred)
        preds = np.concatenate(preds, axis=1)
        preds = np.where(preds == nodata, np.nan, preds)
        assert preds.shape[0] == sota_chm_df.shape[0]
        df = pd.DataFrame(preds, columns=[f'RH{i}' for i in range(101)])
        df['geometry'] = sota_chm_df.geometry
        df = gpd.GeoDataFrame(df, crs="EPSG:4326")
        df.to_parquet(self.save_dir / f'{tile_id}.parquet')
        print(f'{self.save_dir / f"{tile_id}.parquet"} saved')


@dataclass
class MyConfig:
    sota_chm_df_fp: str = '~/data/gvs/evaluation/sota_chm_val_with_gedi_biome.parquet'
    prediction_dir: str = '~/data/gvs/deploy/predictions_2020'
    gedi_ref_dir: str = '~/data/gvs/GEDI_for_correction/partitions_2020_v1'
    sota_chm_dir: str = '~/data/gvs/GEDI_for_correction/partitions_with_sota_chm_2020_v2'
    s2_grid_file: str = '~/data/gvs/s2_tiles_with_growing_months.parquet'
    test_tiles_file: str = '~/data/gvs/split_test0.1_cal0.1_val0.1_seed42_v1/tiles_test.csv'
    run_id: str = ''
    data_name: str = 'test'
    output_dir: str = 'output'
    task: str = 'aggregate_gedi_by_biome'
    corrected: bool = False


cs = ConfigStore.instance()
cs.store(name="my_config", node=MyConfig)


@hydra.main(config_name="my_config", version_base="1.2")
def main(cfg):
    evaluation = Evaluation(**cfg)
    
    # evaluation.compare_with_sota_chms(cfg.sota_chm_and_ours_fp, cfg.get('countries_file'))
    countries_file = '~/data/gvs/deploy/eu_results/countries_list.txt'
    evaluation.evaluate_eu_test_tiles(countries_file, cfg.s2_grid_file, cfg.test_tiles_file)
    # if cfg.task == 'compare_result_precision':
    #     compare_result_precision(cfg.run_id, cfg.corrected)
    # elif cfg.task == 'add_biome':
    #     add_biome(val_df_fp=f'~/data/gvs/train_subsets/{cfg.data_name}_filtered_v1.parquet', cal_pred_fp=cal_pred_fp)
    # elif cfg.task == 'compare_with_sota_maps':
    #     compare_with_sota_maps(cfg.sota_chm_df_fp, cfg.run_id, cfg.corrected)
    # elif cfg.task == 'compare_with_sota_maps_eu':
    #     compare_with_sota_maps_eu(cfg.sota_chm_df_fp, cfg.pred_df_fp, cfg.countries_file)

    # # add biome for sota maps
    # # add_biome(val_df_fp='~/data/gvs/train_subsets/val_filtered_v1.parquet', sota_chm_df_dir='~/data/gvs/evaluation/existing_canopy_height_pred_val_with_gedi')
    # # add biome for cal/test predictions
    # cal_pred_fp = f'~/data/gvs/uncertainty/canopy_height_predictions_{cfg.run_id}_corrected.parquet'
    # add_biome(val_df_fp=f'~/data/gvs/train_subsets/{cfg.data_name}_filtered_v1.parquet', cal_pred_fp=cal_pred_fp)

if __name__ == '__main__':
    # from dask.distributed import Client, LocalCluster
    # cluster = LocalCluster(n_workers=8, dashboard_address=':38787')
    # client = Client(cluster)
    main()

    
