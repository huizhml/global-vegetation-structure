import os
import time
import hydra
from pathlib import Path
import dask
import hvplot.dask
import dask.dataframe as dd
from dask.utils import natural_sort_key
import geopandas as gpd
import dask_geopandas as dgp
from dataclasses import dataclass
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig


class DataSplitter:
    def __init__(self, index_dir, test_ratio:float=0.1, val_ratio: float=0.1, cal_ratio:float=0.1, random_state:int=42,
                s2_grid_file: Path=None, mgrs_file: Path=None):
        self.index_dir = index_dir
        self.test_ratio = test_ratio
        self.val_ratio = val_ratio
        self.cal_ratio = cal_ratio
        self.random_state = random_state
        self.train_cal_val_test_split(s2_grid_file)
        self.mgrs_df = gpd.read_parquet(mgrs_file, columns=['MGRS_UTM', 'geometry'])
        self.mgrs_df = self.mgrs_df.set_index('MGRS_UTM')
        self.mgrs_df.crs = 'EPSG:4326'
        self.splits = ['test', 'cal', 'val']
        for name in self.splits + ['train']:
            directory = self.index_dir.parent / f'{name}_index_table'
            directory.mkdir(exist_ok=True)
            self.__setattr__(f'{name}_index_table', directory)



    def train_cal_val_test_split(self, s2_grid_file:Path=None):
        save_dir = self.index_dir.parent / 'splited_tiles'
        save_dir.mkdir(exist_ok=True)
        geo_index_df = dd.read_parquet(f'{self.index_dir}/*.parquet')
        unique_tiles = geo_index_df['s2_tile'].unique().compute()
        s2_grid = gpd.read_file(s2_grid_file)
        s2_grid = s2_grid.set_index('Name')
        self.s2_grid = s2_grid.loc[unique_tiles]

        n_test = int(len(unique_tiles) * self.test_ratio)
        n_val = int(len(unique_tiles) * self.val_ratio)
        n_cal = int(len(unique_tiles) * self.cal_ratio)

        test_tiles = unique_tiles.sample(n=n_test, random_state=self.random_state)
        test_tiles.to_csv(save_dir / 'test_tiles.csv', header=None, index=None, sep=' ', mode='w')
        self.test_tiles = self.s2_grid.loc[test_tiles]
        
        unique_tiles = unique_tiles.drop(test_tiles.index)
        val_tiles = unique_tiles.sample(n=n_val, random_state=self.random_state)
        val_tiles.to_csv(save_dir / 'val_tiles.csv', header=None, index=None, sep=' ', mode='w')
        self.val_tiles = self.s2_grid.loc[val_tiles]

        unique_tiles = unique_tiles.drop(val_tiles.index)
        cal_tiles = unique_tiles.sample(n=n_cal, random_state=self.random_state)
        cal_tiles.to_csv(save_dir / 'cal_tiles.csv', header=None, index=None, sep=' ', mode='w')
        self.cal_tiles = self.s2_grid.loc[cal_tiles]

        # train_tiles = unique_tiles.drop(cal_tiles.index)
        # train_tiles.to_csv(save_dir / 'train_tiles.csv', header=None, index=None, sep=' ', mode='w')
        # self.train_tiles = self.s2_grid.loc[train_tiles]
        # print()


    def split_zone(self, index_df, partition_info):
        partition_idx = partition_info["number"] # the index of the partition in the dataframe
        zone = os.path.basename(self.index_table_fps[partition_idx])[:3]
        index_df = index_df.reset_index(drop=True)
        for name in self.splits:
            overlapped_tiles = getattr(self, f'{name}_tiles').sjoin(self.mgrs_df.loc[[zone]], how='left')
            overlapped_tiles = overlapped_tiles.dropna(subset=['index_right'])
            if not overlapped_tiles.empty:
                overlapped_tiles = overlapped_tiles.drop(columns=['index_right'])
                split_index_df = index_df.sjoin(overlapped_tiles, how='left')
                split_index_df = split_index_df.dropna(subset='index_right')
                split_index_df = split_index_df.rename(columns={'index_right': 'assigned_tile'})
                split_index_df = split_index_df[~split_index_df.index.duplicated(keep='first')]
                split_index_df.to_parquet(getattr(self, f'{name}_index_table') / f'{zone}.parquet')
                index_df = index_df.drop(split_index_df.index)
        
        index_df.to_parquet(self.train_index_table / f'{zone}.parquet')
        print(f'{zone} done')
    

    def run_split(self):
        index_table_fps = [f'{self.index_dir}/{zone}' for zone in os.listdir(self.index_dir)]
        self.index_table_fps = sorted(index_table_fps, key=natural_sort_key)
        index_df = dgp.read_parquet(self.index_table_fps , gather_spatial_partitions=False, columns=['path', 's2_tile', 'in_partition_idx', 'geometry'])
        index_df.map_partitions(self.split_zone, meta=index_df._meta).compute()

    def visualize_split(self):
        
        pass


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
    from dask.distributed import Client, LocalCluster
    from dask import config
    cluster = LocalCluster()
    client = Client(cluster)

    print(client)

    t0 = time.time()
    splitter = DataSplitter(cfg.index_dir, cfg.test_ratio, cfg.val_ratio, cfg.cal_ratio, cfg.random_state, cfg.s2_grid_file, cfg.mgrs_file)
    splitter.run_split()
    print(f'time taken: {time.time() - t0}')
    client.close()

if __name__ == "__main__":
    main()