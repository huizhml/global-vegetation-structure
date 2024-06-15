import os
import time
import hydra
from pathlib import Path
import dask
import hvplot.dask
import pandas as pd
import dask.dataframe as dd
from dask.utils import natural_sort_key
import geopandas as gpd
import dask_geopandas as dgp
from dataclasses import dataclass
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


class DataSplitter:
    def __init__(self, index_dir, test_ratio:float=0.1, val_ratio: float=0.1, cal_ratio:float=0.1, random_state:int=42,
                s2_grid_file: Path=None, mgrs_file: Path=None):
        self.index_dir = index_dir
        self.test_ratio = test_ratio
        self.val_ratio = val_ratio
        self.cal_ratio = cal_ratio
        self.random_state = random_state
        self.s2_grid_file = s2_grid_file
        self.mgrs_file = mgrs_file
        self.splits = ['test', 'cal', 'val']
        self.save_dir = self.index_dir.parent / f'split_test{self.test_ratio}_cal{self.cal_ratio}_val{self.val_ratio}_seed{self.random_state}'
        self.save_dir.mkdir(exist_ok=True)

    def train_cal_val_test_split(self):
        s2_grid = gpd.read_file(self.s2_grid_file)
        self.s2_grid = s2_grid.set_index('Name')
        exist = len(list(self.save_dir.glob('*_tiles.csv'))) == 4
        if exist:
            print('Splitting has been done.')
            for name in self.splits +['train']:
                tiles = pd.read_csv(self.save_dir / f'{name}_tiles.csv', header=None, sep=' ')
                tiles = self.s2_grid.loc[tiles[0]]
                setattr(self, f'{name}_tiles', tiles)
        else:  
            geo_index_df = dd.read_parquet(f'{self.index_dir}/*.parquet')
            unique_tiles = geo_index_df['s2_tile'].unique().compute()
            self.s2_grid = self.s2_grid.loc[unique_tiles]

            n_test = int(len(unique_tiles) * self.test_ratio)
            n_val = int(len(unique_tiles) * self.val_ratio)
            n_cal = int(len(unique_tiles) * self.cal_ratio)

            test_tiles = unique_tiles.sample(n=n_test, random_state=self.random_state)
            test_tiles.to_csv(self.save_dir / 'test_tiles.csv', header=None, index=None, sep=' ', mode='w')
            self.test_tiles = self.s2_grid.loc[test_tiles]
            
            unique_tiles = unique_tiles.drop(test_tiles.index)
            val_tiles = unique_tiles.sample(n=n_val, random_state=self.random_state)
            val_tiles.to_csv(self.save_dir / 'val_tiles.csv', header=None, index=None, sep=' ', mode='w')
            self.val_tiles = self.s2_grid.loc[val_tiles]

            unique_tiles = unique_tiles.drop(val_tiles.index)
            cal_tiles = unique_tiles.sample(n=n_cal, random_state=self.random_state)
            cal_tiles.to_csv(self.save_dir / 'cal_tiles.csv', header=None, index=None, sep=' ', mode='w')
            self.cal_tiles = self.s2_grid.loc[cal_tiles]

            self.train_tiles = unique_tiles.drop(cal_tiles.index)
            self.train_tiles.to_csv(self.save_dir / 'train_tiles.csv', header=None, index=None, sep=' ', mode='w')
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
        exist = len(list(self.save_dir.glob('*_index_table'))) == 4 #NOTE: didn't check files under the directory
        if exist:
            print('Splitting has been done.')
            return
        self.train_cal_val_test_split()
        self.mgrs_df = gpd.read_parquet(self.mgrs_file, columns=['MGRS_UTM', 'geometry'])
        self.mgrs_df.crs = 'EPSG:4326'
        for name in self.splits + ['train']:
            directory = self.save_dir / f'{name}_index_table'
            directory.mkdir(exist_ok=True)
            self.__setattr__(f'{name}_index_table', directory)

        index_table_fps = [f'{self.index_dir}/{zone}' for zone in os.listdir(self.index_dir)]
        self.index_table_fps = sorted(index_table_fps, key=natural_sort_key)
        index_df = dgp.read_parquet(self.index_table_fps , gather_spatial_partitions=False, columns=['path', 's2_tile', 'in_partition_idx', 'geometry'])
        index_df.map_partitions(self.split_zone, meta=index_df._meta).compute()

    def count_for_split_per_zone(self):
        '''
        Count the number of samples for each split in each MGRS zone
        Data splitting has been done.
        '''
        if (self.save_dir / 'mgrs_stats.parquet').exists() and (self.save_dir / 'split_stats.txt').exists():
            print('Counting has been done.')
            return
        self.mgrs_df = gpd.read_parquet(self.mgrs_file)
        self.mgrs_df['total_downloaded'] = self.mgrs_df[[f'downloaded_{year}' for year in range(2019, 2023)]].sum(axis=1)
        splits = self.splits + ['train']
        # for each split, e.g., test, reading index_df for each zone from test_index_table
        ntiles_per_split = {}
        for name in splits:
            data_dir = self.save_dir / f'{name}_index_table'
            split_index_df_fps = [f'{data_dir}/{zone}' for zone in os.listdir(data_dir)]
            split_index_df_fps = sorted(split_index_df_fps, key=natural_sort_key)
            split_index_df = dd.read_parquet(split_index_df_fps)
            count = split_index_df.map_partitions(len).compute()
            ntiles = split_index_df['s2_tile'].nunique().compute()
            ntiles_per_split[name] = ntiles
            count = count.to_frame()
            count.loc[:, 'MGRS_UTM'] = [os.path.basename(f)[:3] for f in split_index_df_fps]
            count = count.set_index('MGRS_UTM', drop=True)
            self.mgrs_df.loc[:, f'count_{name}'] = count[0]
            self.mgrs_df[f'ratio_{name}'] = self.mgrs_df[f'count_{name}'] / self.mgrs_df['total_downloaded']
            self.mgrs_df[f'ratio_{name}'] = self.mgrs_df[f'ratio_{name}'].replace(0, pd.NA)
        # self.mgrs_df = self.mgrs_df.fillna(value={f'count_{name}': 0 for name in splits})
        # sum_split = self.mgrs_df[[f'count_{name}' for name in splits]].sum(axis=1)
        self.mgrs_df.to_parquet(self.save_dir / 'mgrs_stats.parquet')

        # generate a summary
        stats = []
        for name in splits:
            n = self.mgrs_df[f'count_{name}'].sum()
            tiles = pd.read_csv(self.save_dir / f'{name}_tiles.csv', header=None, sep=' ')
            stats.append([n, len(tiles)])

        stats = pd.DataFrame(stats, index=splits, columns=['count', 'n_s2_cells'])
        stats['ratio'] = stats['count'] / stats['count'].sum()
        stats['n_image_tiles'] = pd.Series(ntiles_per_split)
        stats.to_csv(self.save_dir / 'split_stats.txt', sep=' ')
    
    def visualize_split(self):
        import matplotlib.pyplot as plt
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        # Visualize splitted tiles        
        s2_grid = gpd.read_file(self.s2_grid_file)
        s2_grid = s2_grid.set_index('Name')
        splits = self.splits + ['train']
        for name in splits:
            tiles = pd.read_csv(self.save_dir / f'{name}_tiles.csv', header=None, sep=' ')
            tiles = tiles[0].tolist()
            s2_grid.loc[tiles, 'split'] = name
        s2_grid = s2_grid.dropna(subset=['split'])
        s2_grid.plot(column='split', legend=True)
        s2_grid.to_parquet(self.save_dir / 's2_grid_with_split.parquet')
        # plot
        colors = ['red', 'yellow', 'gray', 'blue']
        cmap = ListedColormap(colors)
        fig = plt.figure()
        ax = s2_grid.plot(column='split', legend=True, cmap=cmap, figsize=(24, 10))
        plt.tight_layout()
        overlay_country_boundaries(ax)
        plt.savefig(self.save_dir / 'splitted_s2_tiles.png')

        # Visualize the number of samples in each zone for each split
        df = gpd.read_parquet(self.save_dir / 'mgrs_stats.parquet')
        df = df[df['total_downloaded']>0]
        df.crs = 'EPSG:4326'
        cmap = 'coolwarm'
        for name in splits:
            fig, ax = plt.subplots(1, 1, figsize=(30, 12))
            # Plotting
            df.plot(column=f'ratio_{name}', 
                    cmap=cmap, 
                    ax=ax, 
                    legend=False, 
                    missing_kwds={
                        "color": "lightgrey",
                        "edgecolor": "red",
                        "hatch": "///",
                        "label": "Missing values",
                    })
            overlay_country_boundaries(ax)
            # Create a divider for the existing axes instance
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("bottom", size="5%", pad=0.3)  # Adjust pad to 

            # Add colorbar 
            # TODO: adjust the colorbar length
            # Tried size, aspect, fraction, shrink, but none of them worked
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
            cbar = fig.colorbar(sm, cax=cax, orientation='horizontal') 
            cbar.set_label('Percentage', fontsize=15)
            cbar.ax.tick_params(labelsize=15)
            # Adjust the aspect ratio to change the length of the colorbar
            # cax.set_aspect(0.005)  # Adjust this value to change the length of the colorbar
            plt.tight_layout()
            plt.savefig(self.save_dir / f'split_ratio_map_{name}.png')


def overlay_country_boundaries(ax):
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    countries_flt = world[world.geometry.apply(lambda x: x.bounds[1] > -60)]
    countries_flt.plot(ax=ax, color='none', edgecolor='black', linewidth=0.5)

@dataclass
class MyConfig:
    index_dir: Path = Path('~/scratch/data/geo_index_table').expanduser()
    mgrs_file: Path = Path('~/scratch/data/download_stats.parquet').expanduser()
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
    print('Run split...')
    splitter.run_split()
    print('Count the number of train/val/cal/test samples per zone...')
    splitter.count_for_split_per_zone()
    print('Visualize...')
    splitter.visualize_split()

    print(f'time taken: {time.time() - t0}')
    client.close()

if __name__ == "__main__":
    main()  