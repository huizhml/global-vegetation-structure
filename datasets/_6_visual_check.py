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


def aggregate_gedi_by_biome(beton_fps: List[str], group_by_biome=False):
    '''
    Aggregate GEDI data (RHs + GEDI attributes) {train*}.beton to a single parquet file
    '''
    beton_fps = glob.glob(str(Path(beton_fps).expanduser()))
    print(beton_fps)
    cols = [f'rh{i}' for i in range(101)] + ['wc', 'slope', 'lon', 'lat']
    data_dir = Path(beton_fps[0]).parent.parent
    ecoregions = gpd.read_file( '~/data/GEDI/ecoregions/wwf_terr_ecos.shp')
    batch_size = 100 if 'debug' in beton_fps[0] else 4096
    for fp in beton_fps:
        file = Path(fp).with_suffix('.parquet')
        # if file.exists():
        #     print('file exists', file)
        #     continue
        loader = Loader(fp, batch_size=batch_size, num_workers=1,
                        distributed=False, batches_ahead=3,
                        order=OrderOption.SEQUENTIAL, os_cache=False, drop_last=False)

        data = []
        for batch in tqdm(loader):
            _, rhs, wc, slope, lon, lat = batch
            batch_ = np.concatenate([rhs, wc[...,7:8,7], slope[...,7:8,7], lon[:,7:8], lat[:, 7:8]], axis=1)
            data.append(batch_)
        data = np.concatenate(data, axis=0)
        print(data.shape)
        df = pd.DataFrame(data, columns=cols)
        df = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat))
        df = df.set_crs(epsg=4326)
        df = df.to_crs(epsg=3857)
        df = gpd.sjoin(df, ecoregions, how='left', predicate='within')
        df = df[cols + ['BIOME']]
        df.to_parquet(file)
        print('saved to', file)
    print('Grouping by biome')
    df_fp = glob.glob(str(file.parent / '*.parquet'))
    
    if group_by_biome:
        group_df_by_biome(df_fp)


def group_df_by_biome(df_fps: List[str] = None):
    """
    Group the dataframe by biome, and save parquets for each biome
    """
    import dask.dataframe as dd
    ddf = dd.read_parquet(df_fps)
    data_dir = Path(df_fps[0]).parent
    df = ddf.compute()
    print(df.BIOME.unique())

    def save_parquet(x):
        print(x)
        if x.name is not None and int(x.name) < 15:
            name = f'rhs_attrs_{BIOMES[int(x.name)-1].replace(" ", "_").replace("&", "and")}.parquet'
            x.to_parquet(data_dir / name)
        else:
            x.to_parquet(data_dir / f'rhs_attrs_biome_{x.name}.parquet')
    df.groupby('BIOME').apply(save_parquet)


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

def get_actual_downloaded_gedi(sampled_gedi_dir, downloaded_gedi_dir):
    """
    Get the actual downloaded GEDI data from the sampled GEDI data 
    # TODO: This function is not working since the shot_number is dropped in the beton file
    """
    sampled_gedi_dir = Path(sampled_gedi_dir).expanduser()
    downloaded_gedi_dir = Path(downloaded_gedi_dir).expanduser()
    zones = set()
    for year in range(2019, 2023):
        z = os.listdir(sampled_gedi_dir / str(year))
        zones.update(z)
    downloaded_gedi = dd.read_parquet(downloaded_gedi_dir/'train*.parquet', columns=['BIOME'])
    for zone in zones:
        print('Processing', zone)
        sampled_gedi_fps = glob.glob(str(sampled_gedi_dir / '*' / zone / '*.parquet'))
        print(f'{len(sampled_gedi_fps)} partitions found in {zone}')
        ddf = dgp.read_parquet(sampled_gedi_fps)
        ddf = ddf.drop(columns=['s2_candidates'])
        # ddf = ddf[ddf['shot_number'].isin(downloaded_gedi['shot_number'])]
        ddf.compute().to_parquet(downloaded_gedi_dir.parent / f'GEDI_valid/{zone}.parquet')

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
    if task == 'aggregate_gedi_by_biome':
        aggregate_gedi_by_biome(cfg.beton_fps)
    elif task == 'visualize_splitted_points':
        visualize_splitted_points('~/data/GEDI/train_subsets', 'v3')
    elif task == 'visualize_quantile_distribution_per_rh':
        visualize_quantile_distribution_per_rh('~/data/gvs/train_subsets/train*.parquet')
    # # model.run_land_cover_effect_size_analysis()

    print(f'time taken for running {cfg.task}: {time.time() - t0}')


if __name__ == '__main__':
    main()