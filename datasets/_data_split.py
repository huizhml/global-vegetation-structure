import time
import hydra
from pathlib import Path
import dask.dataframe as dd
import geopandas as gpd
import dask_geopandas as dgp
from dataclasses import dataclass
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig

def train_cal_val_test_split(index_dir, test_ratio:float=0.1, val_ratio: float=0.1, cal_ratio:float=0.1, random_state:int=42):
    save_dir = index_dir.parent / 'splited_tiles'
    save_dir.mkdir(exist_ok=True)
    geo_index_df = dd.read_parquet(f'{index_dir}/*.parquet')
    unique_tiles = geo_index_df['s2_tile'].unique().compute()

    n_test = int(len(unique_tiles) * test_ratio)
    test_tiles = unique_tiles.sample(n=n_test, random_state=random_state)
    test_tiles.to_csv(save_dir / 'test_tiles.csv', header=None, index=None, sep=' ', mode='w')
    
    unique_tiles = unique_tiles.drop(test_tiles.index)
    n_val = int(len(unique_tiles) * val_ratio)
    val_tiles = unique_tiles.sample(n=n_val, random_state=random_state)
    val_tiles.to_csv(save_dir / 'val_tiles.csv', header=None, index=None, sep=' ', mode='w')

    unique_tiles = unique_tiles.drop(val_tiles.index)
    n_cal = int(len(unique_tiles) * cal_ratio)
    cal_tiles = unique_tiles.sample(n=n_cal, random_state=random_state)
    cal_tiles.to_csv(save_dir / 'cal_tiles.csv', header=None, index=None, sep=' ', mode='w')

    train_tiles = unique_tiles.drop(cal_tiles.index)
    train_tiles.to_csv(save_dir / 'train_tiles.csv', header=None, index=None, sep=' ', mode='w')
    return train_tiles.to_frame(), cal_tiles.to_frame(), val_tiles.to_frame(), test_tiles.to_frame()

def get_exclusive_patches(tiles):
    zone = tiles.name
    s2_grid = gpd.read_file('~/scratch/Sentinel-2_tilling_shp/sentinel_2_index_shapefile.shp')

    pass

def main_(cfg):
    train_tiles, cal_tiles, val_tiles, test_tiles = train_cal_val_test_split(cfg.index_dir, cfg.test_ratio, cfg.val_ratio, cfg.cal_ratio, cfg.random_state)
    s2_grid = gpd.read_file(cfg.s2_grid_file)
    s2_grid = s2_grid.set_index('Name')
    test_tiles = s2_grid.loc[test_tiles['s2_tile']]
    mgrs_df = gpd.read_parquet(cfg.mgrs_file, columns=['geometry', 'MGRS_UTM'])
    intersected_zones = mgrs_df.sjoin(test_tiles, how='left')
    intersected_zones = intersected_zones['MGRS_UTM'].tolist()
    intersected_index_fp = [f'{cfg.index_dir}/{zone}.parquet' for zone in intersected_zones]
    index_df = dd.read_parquet(intersected_index_fp, gather_spatial_partitions=False)
    index_df = index_df.compute()
    test_index_table = index_df.sjoin(test_tiles, how='left')

    test_index_df = test_tiles.sjoin(index_df)

    test_tiles['MGRS'] = test_tiles['s2_tile'].str[:3]
    test_tiles.groupby('MGRS').apply()

    n_test = int(len(unique_tiles) * cfg.test_ratio)
    test_tiles = unique_tiles.sample(n=n_test, random_state=cfg.random_state)
    unique_tiles = unique_tiles.drop(test_tiles.index)
    n_val = int(len(unique_tiles) * cfg.val_ratio)
    val_tiles = unique_tiles.sample(n=n_val, random_state=cfg.random_state)
    unique_tiles = unique_tiles.drop(val_tiles.index)
    n_cal = int(len(unique_tiles) * cfg.cal_ratio)
    cal_tiles = unique_tiles.sample(n=n_cal, random_state=cfg.random_state)
    train_tiles = unique_tiles.drop(cal_tiles.index)
    test_tiles = unique_tiles.sample(frac=0.1)


@dataclass
class MyConfig:
    index_dir: Path = Path('~/scratch/data/geo_index_table').expanduser()
    mgrs_file: Path = Path('~/scratch/mgrs_with_nbest.parquet').expanduser()
    s2_grid_file: Path = Path('~/scratch/Sentinel-2_tilling_shp/sentinel_2_index_shapefile.shp').expanduser()
    test_ratio: float = 0.1
    val_ratio: float = 0.1
    cal_ratio: float = 0.1
    random_state: int = 42

cs = ConfigStore.instance()
cs.store(name="my_config", node=MyConfig)

@hydra.main(config_name="my_config")
def main(cfg: DictConfig) -> None:
    from dask.distributed import Client, LocalCluster, performance_report
    from distributed.diagnostics import MemorySampler
    from dask import config
    cluster = LocalCluster()
    client = Client(cluster)

    print(client)

    t0 = time.time()
    geo_index_df = dd.read_parquet(f'{cfg.index_dir}/*.parquet')
    unique_tiles = geo_index_df['s2_tile'].unique().compute()
    unique_tiles = unique_tiles.tolist()

    index_df = dgp.read_parquet('/users/zhanghui/data/geo_index_table/*.parquet', columns=['geometry', 'path', 'in_partition_idx', 's2_tile'], gather_spatial_partitions=False)
    index_df = index_df.compute()
    # index_df.to_parquet('/users/zhanghui/data/geo_index_table/01G_test.parquet')
    s2_grid = gpd.read_file('/users/zhanghui/scratch/Sentinel-2_tilling_shp/sentinel_2_index_shapefile.shp')
    s2_grid = s2_grid.set_index('Name')
    s2_grid = s2_grid.loc[unique_tiles]
    print(s2_grid)
    # s2_grid = s2_grid.compute()
    # index_df.get_partition(0).compute()
    res = s2_grid.sjoin(index_df)
    print(f'time taken: {time.time() - t0}')
    client.close()

if __name__ == "__main__":
    main()