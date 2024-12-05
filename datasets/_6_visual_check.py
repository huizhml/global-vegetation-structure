from typing import List
from pathlib import Path
from dataclasses import dataclass, field
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig
import hydra
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import geopandas as gpd
import torch
from tqdm import tqdm
import glob
from ffcv.loader import Loader, OrderOption
from const import BIOMES


def aggregate_gedi_by_biome(beton_fps: List[str], group_by_biome=False):
    '''
    Aggregate GEDI data (RHs + GEDI attributes) {train*}.beton to a single parquet file
    '''
    beton_fps = glob.glob(str(Path(beton_fps).expanduser()))
    cols = [f'rh{i}' for i in range(101)] + ['wc', 'slope', 'lat', 'lon', 'sensitivity']
    data_dir = Path(beton_fps[0]).parent.parent
    ecoregions = gpd.read_file( '~/data/GEDI/ecoregions/wwf_terr_ecos.shp')
    batch_size = 100 if 'debug' in beton_fps[0] else 4096
    for fp in beton_fps:
        file = Path(fp).with_suffix('.parquet')
        if file.exists():
            print('file exists', file)
            continue
        loader = Loader(fp, batch_size=batch_size, num_workers=2,
                        distributed=False, batches_ahead=3,
                        order=OrderOption.SEQUENTIAL, os_cache=False)

        data = []
        shot_numbers = []
        for batch in tqdm(loader):
            _, rhs, wc, slope, latlon, sensitivity, shot_number = batch
            shot_numbers.append(shot_number.numpy().copy())
            batch_ = np.concatenate([rhs, wc[...,7:8,7], slope[...,7:8,7], latlon, sensitivity], axis=1)
            data.append(batch_)
        data = np.concatenate(data, axis=0)
        df = pd.DataFrame(data, columns=cols)
        df['shot_number'] = np.concatenate(shot_numbers)
        df = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat))
        df = df.set_crs(epsg=4326)
        df = df.to_crs(epsg=3857)
        df = gpd.sjoin(df, ecoregions, how='left', predicate='within')
        df = df[cols + ['BIOME', 'shot_number']]
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


def visualize_split(parquet_dir, data_version, train_tile='31TGN', val_tile='31TGM', cal_tile='32TLS', test_tile='32TLT'):
    import dask.dataframe as dd
    import matplotlib.pyplot as plt
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
        index_table_ = dd.read_parquet(f'{parquet_dir}/{name}*_{data_version}.parquet', columns=['lat', 'lon', 'shot_number']).compute()
        index_table_ = gpd.GeoDataFrame(index_table_, geometry=gpd.points_from_xy(index_table_.lon, index_table_.lat), crs='EPSG:4326')
        points_in_tile = index_table_.sjoin(tile, how='inner')
        color = colors[colors['Name'] == tile_name]['color']
        print(tile_name)
        points_in_tile.plot(ax=base, marker='o', color=color.item(), markersize=1, alpha=0.6)
    
    plt.savefig('output/split_distribution/example_split.png')


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
    if task == 'aggregate_gedi_by_biome':
        aggregate_gedi_by_biome(cfg.beton_fps)
    elif task == 'visualize_split':
        visualize_split('~/data/GEDI/train_subsets', 'v3')
    # model.run_land_cover_effect_size_analysis()

    print(f'time taken for running {cfg.task}: {time.time() - t0}')


if __name__ == '__main__':
    main()