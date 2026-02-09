import os
from typing import List
from pathlib import Path
from dataclasses import dataclass, field
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig
import hydra
import matplotlib.pyplot as plt
import pandas as pd
import dask_geopandas as dgp
import seaborn as sns
import numpy as np
import geopandas as gpd
import dask.dataframe as dd
import torch
import dask
from tqdm import tqdm
import glob
from ffcv.loader import Loader, OrderOption
from const import BIOMES



def visualize_splitted_points(parquet_dir, data_version, train_tile=None, val_tile=None, cal_tile=None, test_tile=None, split_dir='~/data/gvs/split_test0.1_cal0.1_val0.1_seed42_v1'):
    import dask.dataframe as dd
    import matplotlib.pyplot as plt
    if train_tile is None:
        train_tiles = pd.read_csv(split_dir + '/tiles_train.csv', header=None)
        train_tiles = train_tiles[0].sample(1).values[0]
    if val_tile is None:
        val_tiles = pd.read_csv(split_dir + '/tiles_val.csv', header=None)
        val_tiles = val_tiles[0].sample(1).values[0]
    if cal_tile is None:
        cal_tiles = pd.read_csv(split_dir + '/tiles_cal.csv', header=None)
        cal_tiles = cal_tiles[0].sample(1).values[0]
    if test_tile is None:
        test_tiles = pd.read_csv(split_dir + '/tiles_test.csv', header=None)
        test_tiles = test_tiles[0].sample(1).values[0]
    colors = pd.DataFrame({
        'Name': [train_tile, val_tile, cal_tile, test_tile],
        'color': ['#666666', '#0000ff', '#ff0000', '#ffff00']
    })
    s2_grid = gpd.read_file('~/data/GEDI/Sentinel-2_tilling_shp/sentinel_2_index_shapefile.shp')
    tiles = s2_grid[s2_grid['Name'].isin([train_tile, val_tile, cal_tile, test_tile])]
    tiles = pd.merge(tiles, colors, on='Name')
    base = tiles.plot(edgecolor=tiles['color'], color='#ffffff00', legend=True)
    for name, tile_name in zip(['train', 'val', 'cal', 'test'], [train_tile, val_tile, cal_tile, test_tile]):
        tile = s2_grid[s2_grid['Name']==tile_name]
        index_table_ = dd.read_parquet(f'{parquet_dir}/{name}*_{data_version}.parquet', columns=['lat', 'lon']).compute()
        index_table_ = gpd.GeoDataFrame(index_table_, geometry=gpd.points_from_xy(index_table_.lon, index_table_.lat), crs='EPSG:4326')
        points_in_tile = index_table_.sjoin(tile, how='inner')
        color = colors[colors['Name'] == tile_name]['color']
        print(tile_name)
        points_in_tile.plot(ax=base, marker='o', color=color.item(), markersize=1, alpha=0.6)
    
    plt.savefig('output/split_distribution/example_split.png')


def visualize_quantile_distribution_per_rh(parquet_fps):
    """
    Visualize the quantile distribution per RH
    """
    parquet_fps = Path(parquet_fps).expanduser()
    parquet_fps = parquet_fps.parent.glob(parquet_fps.name)
    parquet_fps = list(parquet_fps)
    ddf = dd.read_parquet(parquet_fps)
    # percentiles = [p/100 for p in range(0, 101)]
    percentiles = [0, 0.00001, 0.0001, 0.001] + [p/100 for p in range(1, 100)] + [0.999, 0.9999, 0.99999, 0.999999, 0.9999999, 1]
    quantiles = []
    for i in range(0, 101):
        quantiles.append(ddf[f'rh{i}'].quantile(percentiles))
    quantiles = dask.compute(*quantiles)  

    for i in range(0, 101):
        plt.figure()
        plt.plot(percentiles, quantiles[i])
        plt.xlabel('Percentile (1-100)')
        plt.ylabel('Value')
        plt.grid(True)
        plt.savefig(f'output/data_stats/quantile_distribution_rh{i}.png')

    quantiles = pd.DataFrame(quantiles, columns=percentiles)
    quantiles.to_csv('output/data_stats/quantile_distribution.csv')

@dataclass
class MyConfig:
    beton_fps: str = '~/data/GEDI/train_subsets/train*_filtered_v1.beton'
    output_dir: str = 'output'
    task: str = 'aggregate_gedi_by_biome'


cs = ConfigStore.instance()
cs.store(name="my_config", node=MyConfig)


@hydra.main(config_name="my_config", version_base="1.2")
def main(cfg: DictConfig) -> None:
    print(cfg)
    import time
    t0 = time.time()
    task = cfg.task
    # print('Find the correlation between sensitivity and beam coverage')
    # find_sensitivity_beam_cor(glob.glob(cfg.parquet_fp))
    # find_sensitivity_biome_cor(glob.glob(cfg.parquet_fp))
    print(task)
    # get_actual_downloaded_gedi('~/data/GEDI/GEDI_with_s2_candidates_and_best', '~/data/gvs/train_subsets')
    if task == 'visualize_splitted_points':
        visualize_splitted_points('~/data/GEDI/train_subsets', 'v3')
    elif task == 'visualize_quantile_distribution_per_rh':
        visualize_quantile_distribution_per_rh('~/data/gvs/train_subsets/train*.parquet')
    # # model.run_land_cover_effect_size_analysis()

    print(f'time taken for running {cfg.task}: {time.time() - t0}')


if __name__ == '__main__':
    main()