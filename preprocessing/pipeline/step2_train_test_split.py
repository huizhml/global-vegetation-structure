import os
import time
import hydra
from pathlib import Path
import dask
import dask.bag as db
import h5py
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

from datasets._3_merge_h5s import dst_conf


class DataSplitter:
    '''
    Split the index table into train, cal, val, and test sets.
    Splitting is based on the Sentinel-2 grid. No spatial overlap between the splits.
    '''
    def __init__(self, index_dir, test_ratio:float=0.1, val_ratio: float=0.1, cal_ratio:float=0.1, random_state:int=42,
                s2_grid_file: Path=None, mgrs_file: Path=None, version: str='v1'):
        self.index_dir = index_dir
        self.test_ratio = test_ratio
        self.val_ratio = val_ratio
        self.cal_ratio = cal_ratio
        self.random_state = random_state
        self.s2_grid_file = s2_grid_file
        self.mgrs_file = mgrs_file
        self.splits = ['test', 'cal', 'val']
        self.save_dir = self.index_dir.parent / f'split_test{self.test_ratio}_cal{self.cal_ratio}_val{self.val_ratio}_seed{self.random_state}_{version}'
        self.save_dir.mkdir(exist_ok=True)

    def _train_cal_val_test_split(self):
        s2_grid = gpd.read_file(self.s2_grid_file)
        self.s2_grid = s2_grid.set_index('Name')
        exist = len(list(self.save_dir.glob('tiles_*.csv'))) == 4
        if exist:
            print('Splitting has been done.')
            for name in self.splits +['train']:
                tiles = pd.read_csv(self.save_dir / f'tiles_{name}.csv', header=None, sep=' ')
                tiles = self.s2_grid.loc[tiles[0]]
                setattr(self, f'tiles_{name}', tiles)
        else:  
            geo_index_df = dd.read_parquet(f'{self.index_dir}/*.parquet')
            geo_index_df = geo_index_df[geo_index_df['sensitivity'] >= 0.95] # !NOTE: filter out low sensitivity samples
            unique_tiles = geo_index_df['s2_tile'].unique().compute()
            unique_tiles = unique_tiles.reset_index(drop=True)
            self.s2_grid = self.s2_grid.loc[unique_tiles]

            n_test = int(len(unique_tiles) * self.test_ratio)
            n_val = int(len(unique_tiles) * self.val_ratio)
            n_cal = int(len(unique_tiles) * self.cal_ratio)

            tiles_test = unique_tiles.sample(n=n_test, random_state=self.random_state)
            tiles_test.to_csv(self.save_dir / 'tiles_test.csv', header=None, index=None, sep=' ', mode='w')
            self.tiles_test = self.s2_grid.loc[tiles_test]
            
            unique_tiles = unique_tiles.drop(tiles_test.index)
            tiles_val = unique_tiles.sample(n=n_val, random_state=self.random_state)
            tiles_val.to_csv(self.save_dir / 'tiles_val.csv', header=None, index=None, sep=' ', mode='w')
            self.tiles_val = self.s2_grid.loc[tiles_val]

            unique_tiles = unique_tiles.drop(tiles_val.index)
            tiles_cal = unique_tiles.sample(n=n_cal, random_state=self.random_state)
            tiles_cal.to_csv(self.save_dir / 'tiles_cal.csv', header=None, index=None, sep=' ', mode='w')
            self.tiles_cal = self.s2_grid.loc[tiles_cal]

            self.tiles_train = unique_tiles.drop(tiles_cal.index)
            self.tiles_train.to_csv(self.save_dir / 'tiles_train.csv', header=None, index=None, sep=' ', mode='w')
            # self.tiles_train = self.s2_grid.loc[tiles_train]
            # print()


    def _split_zone_by_spatial_query(self, index_df, partition_info):
        '''
        Split the index table by the spatial query.
        Parameters:
            index_df: the index table of one zone
            partition_info: stores the partition number
        '''
        partition_idx = partition_info["number"] # the index of the partition in the dataframe
        zone = os.path.basename(self.index_table_fps[partition_idx])[:3]
        index_df = index_df[index_df['sensitivity'] >= 0.95] # !NOTE: filter out low sensitivity samples
        index_df = index_df.reset_index(drop=True)
        for name in self.splits:
            overlapped_tiles = getattr(self, f'tiles_{name}').sjoin(self.mgrs_df.loc[[zone]], how='left')
            overlapped_tiles = overlapped_tiles.dropna(subset=['index_right'])
            if not overlapped_tiles.empty:
                overlapped_tiles = overlapped_tiles.drop(columns=['index_right'])
                split_index_df = index_df.sjoin(overlapped_tiles, how='left')
                split_index_df = split_index_df.dropna(subset='index_right')
                split_index_df = split_index_df.rename(columns={'index_right': 'assigned_tile'})
                split_index_df = split_index_df[~split_index_df.index.duplicated(keep='first')]
                split_index_df.to_parquet(getattr(self, f'index_table_{name}') / f'{zone}.parquet')
                index_df = index_df.drop(split_index_df.index)
        
        index_df.to_parquet(self.index_table_train / f'{zone}.parquet')
        print(f'{zone} done')
    

    def run_split(self):
        exist = len(list(self.save_dir.glob('index_table_*'))) >= 4 #NOTE: didn't check files under the directory
        if exist:
            print('Splitting has been done.')
            return
        self._train_cal_val_test_split()
        self.mgrs_df = gpd.read_parquet(self.mgrs_file, columns=['MGRS_UTM', 'geometry'])
        self.mgrs_df.crs = 'EPSG:4326'
        for name in self.splits + ['train']:
            directory = self.save_dir / f'index_table_{name}'
            directory.mkdir(exist_ok=True)
            self.__setattr__(f'index_table_{name}', directory)

        index_table_fps = [f'{self.index_dir}/{zone}' for zone in os.listdir(self.index_dir)]
        self.index_table_fps = sorted(index_table_fps, key=natural_sort_key)
        index_df = dgp.read_parquet(self.index_table_fps , gather_spatial_partitions=False, columns=['path', 's2_tile', 'in_partition_idx', 'geometry', 'shot_number', 'sensitivity'])
        index_df.map_partitions(self._split_zone_by_spatial_query, meta=index_df._meta).compute()

    def count_for_split_per_zone(self):
        '''
        Count the number of samples for each split in each MGRS zone
        Data splitting has been done.
        '''
        if (self.save_dir / 'mgrs_stats.parquet').exists() and (self.save_dir / 'split_stats.txt').exists():
            print('Counting has been done.')
            return
        self.mgrs_df = gpd.read_parquet(self.mgrs_file)
        # self.mgrs_df['total_downloaded'] = self.mgrs_df[[f'downloaded_{year}' for year in range(2019, 2023)]].sum(axis=1) # 
        self.mgrs_df['total_downloaded'] = self.mgrs_df[[f'count_high_sens_{year}' for year in range(2019, 2023)]].sum(axis=1) # TODO: tmp for old data, when new sup data ready, use above line
        splits = self.splits + ['train']
        # for each split, e.g., test, reading index_df for each zone from test_index_table
        ntiles_per_split = {}
        for name in splits:
            data_dir = self.save_dir / f'index_table_{name}'
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
        self.mgrs_df['count_high_sensitivity'] = self.mgrs_df['count_test'] + self.mgrs_df['count_cal'] + self.mgrs_df['count_val'] + self.mgrs_df['count_train']
        for name in splits:
            self.mgrs_df[f'ratio_{name}'] = self.mgrs_df[f'count_{name}'] / self.mgrs_df['count_high_sensitivity']
            self.mgrs_df[f'ratio_{name}'] = self.mgrs_df[f'ratio_{name}'].replace(0, pd.NA)
        # self.mgrs_df = self.mgrs_df.fillna(value={f'count_{name}': 0 for name in splits})
        # sum_split = self.mgrs_df[[f'count_{name}' for name in splits]].sum(axis=1)
        self.mgrs_df.to_parquet(self.save_dir / 'mgrs_stats.parquet')

        # generate a summary
        stats = []
        for name in splits:
            n = self.mgrs_df[f'count_{name}'].sum()
            tiles = pd.read_csv(self.save_dir / f'tiles_{name}.csv', header=None, sep=' ')
            stats.append([n, len(tiles)])

        stats = pd.DataFrame(stats, index=splits, columns=['count', 'n_s2_cells'])
        stats['ratio'] = stats['count'] / stats['count'].sum()
        stats['n_image_tiles'] = pd.Series(ntiles_per_split)
        stats.to_csv(self.save_dir / 'split_stats.txt', sep=' ')
    
    def visualize_split(self, update=False):
        exists = len(list(self.save_dir.glob(f'split_ratio_map_*.png'))) == 4
        if exists and not update:
            print('Visualizing has been done.')
            return
        import matplotlib.pyplot as plt
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        # Visualize splitted tiles        
        s2_grid = gpd.read_file(self.s2_grid_file)
        s2_grid = s2_grid.set_index('Name')
        splits = self.splits + ['train']
        for name in splits:
            tiles = pd.read_csv(self.save_dir / f'tiles_{name}.csv', header=None, sep=' ')
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


    def sample_subset(self, frac=0.2):
        save_dir = self.save_dir / 'index_table_subset'
        save_dir.mkdir(exist_ok=True)
        self.train_index_fps = [f'{self.save_dir}/index_table_train/{zone}' for zone in os.listdir(f'{self.save_dir}/index_table_train')]
        self.train_index_fps = sorted(self.train_index_fps, key=natural_sort_key)
        train_index_table = dgp.read_parquet(self.train_index_fps, gather_spatial_partitions=False)
        subset_index_table = train_index_table.sample(frac=frac, random_state=self.random_state).dropna()
   
        def save_zone(x, partition_info=None):
            zone = os.path.basename(self.train_index_fps[partition_info['number']])[:3]
            x.to_parquet(self.save_dir / 'index_table_subset' / f'{zone}.parquet')
            return len(x), x['s2_tile'].nunique()
        
        stats = subset_index_table.map_partitions(save_zone, meta=('i8', 'i8'))
        stats = stats.compute()
        size = sum([s[0] for s in stats])
        n_tiles = sum([s[1] for s in stats])
        with open(self.save_dir / 'stats_subset.txt', 'w') as f:
            f.write(f'Number of samples: {size}\n')
            f.write(f'Number of tiles: {n_tiles}\n')

def split_h5(h5_dir, index_dir, save_dir):
    
    index_dir = Path(index_dir).expanduser()
    h5_dir = Path(h5_dir).expanduser()
    save_dir = Path(save_dir).expanduser()

    for name in ['train', 'cal', 'val', 'test']:
        (save_dir / f'{name}_h5s').mkdir(exist_ok=True)
    h5_files = h5_dir.glob('*.h5')
    bg = db.from_sequence(h5_files, npartitions=16)
    bg.map(_split_h5_per_zone, save_dir, index_dir).compute()

def _split_h5_per_zone(h5_in_fp, save_dir, index_dir):
    zone = h5_in_fp.stem
    print('>>> Processing', h5_in_fp)
    with h5py.File(h5_in_fp, 'r') as f:
        for name in ['train', 'val', 'cal', 'test']:
            h5_file = save_dir / f'{name}_h5s/{zone}.h5'
            index_fp = index_dir / f'index_table_{name}' / f'{zone}.parquet'
            print(h5_file, index_fp)
            print(index_fp.exists(), h5_file.exists())
            if h5_file.exists():
                data = h5py.File(h5_file, 'r')
                if len(data.keys()) < 4: # check if the file is corrupted
                    data.close()
                    os.remove(h5_file)
                data.close()
            if h5_file.exists() or not index_fp.exists(): # should be true, false
                continue
            df = gpd.read_parquet(index_fp)
            with h5py.File(h5_file, 'w') as f_out:
                for path in df['path'].unique():
                    idx = df[df['path']==path]['in_partition_idx'].unique()
                    for name, config in dst_conf.items():
                        data = f[f'{path[4:]}/{name}'][idx]
                        f_out.create_dataset(f'{path[4:]}/{name}', 
                                                shape=data.shape, 
                                                chunks=(1,) +config['shape'], 
                                                dtype=config['dtype'],
                                                compression="gzip", #lzf
                                                compression_opts=7, 
                                                data=data)


def overlay_country_boundaries(ax):
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    countries_flt = world[world.geometry.apply(lambda x: x.bounds[1] > -60)]
    countries_flt.plot(ax=ax, color='none', edgecolor='black', linewidth=0.5)

@dataclass
class MyConfig:
    index_dir: Path = Path('~/data/GEDI/geo_index_table_with_sensitivity').expanduser()
    mgrs_file: Path = Path('~/data/GEDI/mgrs_stats.parquet').expanduser()
    s2_grid_file: Path = Path('~/data/GEDI/Sentinel-2_tilling_shp/sentinel_2_index_shapefile.shp').expanduser()
    test_ratio: float = 0.1
    val_ratio: float = 0.1
    cal_ratio: float = 0.1
    random_state: int = 42
    subset_frac: float = 0.2 # the fraction of the training data to sample
    version: str = 'v1'

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
    splitter = DataSplitter(cfg.index_dir, cfg.test_ratio, cfg.val_ratio, cfg.cal_ratio, cfg.random_state, cfg.s2_grid_file, cfg.mgrs_file, cfg.version)
    print('Run split...')
    splitter.run_split()
    print('Count the number of train/val/cal/test samples per zone...')
    splitter.count_for_split_per_zone()
    print('Visualize...')
    splitter.visualize_split()
    print('Split h5 files...') 
    # TODO: filter out low sensitivity samples before generating {split}.h5 to reduce time to convert to beton
    h5_dir = cfg.get('h5_dir', '~/data/GEDI/GEDI_S2_h5s_original')
    index_dir = cfg.get('save_dir', splitter.save_dir)
    save_dir = cfg.get('save_dir', splitter.save_dir)
    split_h5(h5_dir, index_dir, save_dir)
    print('Sample subset...')
    splitter.sample_subset(frac=cfg.subset_frac)

    print(f'time taken: {time.time() - t0}')
    client.close()

if __name__ == "__main__":
    main()  