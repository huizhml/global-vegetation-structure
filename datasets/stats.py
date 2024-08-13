from pathlib import Path
import hydra
import h5py
import json
from typing import Iterable
from dataclasses import dataclass, field
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import ScalarFormatter, MaxNLocator
import matplotlib.colors as colors
from torch.utils.data import DataLoader
import seaborn as sns
import dask.bag as db
import dask.array as da
from dask.utils import natural_sort_key
import dask.dataframe as dd
from datatree.io import _iter_nc_groups
from h5netcdf.legacyapi import Dataset as h5Dataset
from const import ESA_WC
import os
import json
os.environ['BOKEH_ALLOW_WS_ORIGIN'] = 'www.lumi.csc.fi'

HIST_PARAMS = {
    # keys from the h5 data
    'rhs':             {'bins': np.arange(0, 100, 2).tolist(), 'name': 'RH98', 'slices': (98,)},
    'delta_day':       {'bins': np.arange(0, 375, 10).tolist(), 'name': 'Delta Days', 'slices': ()},
    'defective_cover': {'bins': np.arange(0, 0.9, 0.05).tolist(), 'name': 'Defective Cover', 'slices': ()},
    'image':           {'bins': [0]+list(ESA_WC.values())+[110], 'name': 'ESA World Cover', 'slices': (13, 7, 7)},
    'slope':           {'bins': np.arange(0, 92, 2).tolist(), 'name': 'Slope [COPERNICUS GLO-30 & SRTM]', 'slices': (7, 7)}
}

class AverageMeter:
    def __init__(self, minv=0, maxv=100, step=10, bins=None, name:str=None):
        self.name = name
        if bins is not None:
            self.bins = bins
        else:
            self.bins = np.arange(minv, maxv, step)
        self.bincounts = np.zeros(len(self.bins)-1)
    
    def reset(self):
        self.bincounts = np.zeros(len(self.bins)-1)
    
    def update(self, y):
        self.bincounts += np.histogram(y, bins=self.bins)[0]


class Stats:
    def __init__(self, h5_dir:str='~/data/GEDI', save_dir:str='~/scratch/data/split_test0.1_cal0.1_val0.1_seed42'):
        self.h5_dir = Path(h5_dir).expanduser()
        # self.index_dir = Path(index_dir).expanduser()
        self.save_dir = Path(save_dir).expanduser()
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def plot_sample_map(self, mgrs_file:str='~/scratch/sample_stats.parquet'):
        if os.path.exists(mgrs_file):
            mgrs_df = self.sample_stats(mgrs_file=mgrs_df)
        else:
            mgrs_df = gpd.read_parquet(mgrs_file)
        mgrs_df.crs = 'EPSG:4326'

        countries = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
        countries_flt = countries[countries.geometry.apply(lambda x: x.bounds[1] > -60)]#.to_crs('+proj=robin')

        count_cols = [f'count_{year}' for year in range(2019, 2023)]
        sampled_cols = [f'sampled_{year}' for year in range(2019, 2023)]
        diff_cols = [f'diff_{year}' for year in range(2019, 2023)]
        perc_cols = [f'percentage_{year}' for year in range(2019, 2023)]
        mgrs_df = mgrs_df.astype({'nwant': int})
        for year in range(2019, 2023):
            mgrs_df[f'diff_{year}'] = mgrs_df[f'nwant'] - mgrs_df[f'downloaded_{year}']
        
        vmin = {}
        vmax = {}
        vmin['count'] = int(np.percentile(mgrs_df[count_cols], 2))
        vmax['count'] = int(np.percentile(mgrs_df[count_cols], 98))
        vmin['sampled'] =int(np.percentile(mgrs_df[sampled_cols], 2))
        vmax['sampled'] =int(np.percentile(mgrs_df[sampled_cols], 98))
        vmin['downloaded'] = vmin['sampled']
        vmax['downloaded'] = vmax['sampled']
        vmin['diff'] = mgrs_df[diff_cols].min().min()
        vmax['diff'] = mgrs_df[diff_cols].max().max()
        vmin['percentage'] = 0
        vmax['percentage'] = 1

        years = [2019, 2020, 2021, 2022]
        cols = ['sampled', 'count', 'downloaded', 'diff', 'percentage']
        # Create a color map
        cmap = 'viridis'
        titles = ['#Sampled', '#Total', '#Downloaded', '#Want - #Downloaded', '#Downloaded.div#Want']
        fig_width, fig_height = 10, 6  # Set the figure size
        for r, col in enumerate(cols):
            norm = colors.Normalize(vmin=vmin[col], vmax=vmax[col])
            if col == 'percentage':
                cmap = colors.ListedColormap(['gray', 'purple','orange', 'yellow', 'cyan', 'green'])
                bounds = [0, 0.6, 0.7, 0.8, 0.9, 1.0]
                norm = colors.BoundaryNorm(bounds, cmap.N)
            for c, year in enumerate(years):
                fig, ax = plt.subplots(figsize=(fig_width, fig_height))
                # Create a divider for the existing axes instance
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.1)
                mgrs_df.plot(column=f'{col}_{year}', vmin=vmin[col], vmax=vmax[col], cmap=cmap, norm=norm, ax=ax)
                countries_flt.plot(ax=ax, color='none', edgecolor='black', linewidth=0.5)
                ax.set_xlim(-180, 180)
                ax.set_title(f'{titles[r]} footprints in {year}', fontsize=16)
                # Create a scalar mappable for the colorbar
                sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
                sm.set_array([])  # You can also pass in some data here if needed
                
                # Create colorbar in the appended axes
                cbar = plt.colorbar(sm, cax=cax)
                cbar.ax.set_ylabel(col)
                # Set the colorbar to use a reasonable number of ticks
                cbar.locator = MaxNLocator(nbins=5, integer=False)
                cbar.update_ticks()

                # Set the colorbar formatter to scientific notation
                formatter = ScalarFormatter(useMathText=True)
                formatter.set_scientific(True)
                formatter.set_powerlimits((0, 0))  # Adjust limits if necessary
                cbar.ax.yaxis.set_major_formatter(formatter)
                plt.tight_layout()
                plt.savefig(f'outputs/{titles[r]}_{year}.png', dpi=300, bbox_inches='tight', transparent=False)

    def _check_s2_values(self, h5_file):
        print('checking file: ', h5_file)
        with h5Dataset(h5_file, mode='r') as ncds:
            with h5py.File(h5_file) as data:
                for group in _iter_nc_groups(ncds):
                    if len(group.split('/')) == 3:
                        img = data[f'{group}/image'][:]
                        if (img<0).any():
                            print(group)
                            print(img.min())

    def check_s2_values(self, h5_dir:str='~/data/GEDI'):
        h5_dir = Path(h5_dir).expanduser()
        h5_files = h5_dir.glob('*.h5')
        h5_db = db.from_sequence(h5_files, npartitions=16)
        h5_db.map(self._check_s2_values).compute()

    
    def sample_stats(self, index_table:str='~/scratch/data/index_table', mgrs_file:str='~/scratch/mgrs_with_nbest_v2.parquet'):
        from dask.distributed import Client, LocalCluster
        cluster = LocalCluster()
        client = Client(cluster)
        print(client)
        index_df = dd.read_parquet(f'{index_table}/*.parquet')
        splits = index_df.path.str.split('/')
        index_df['zone'] = splits.str[6:9]
        index_df['year'] = splits.str[13:17]
        downloaded_count = index_df.groupby(['zone', 'year']).size()
        downloaded_count = downloaded_count.compute()
        mgrs_df = gpd.read_parquet(mgrs_file)
        mgrs_df = mgrs_df.drop(columns=['tracks'] + [f'tracks_{year}' for year in range(2019, 2023)])
        mgrs_df = mgrs_df.set_index('MGRS_UTM')
        for zone in downloaded_count.index.get_level_values('zone'):
            for year in range(2019,2023):
                mgrs_df.loc[zone, f'downloaded_{year}'] =  downloaded_count.loc[zone, str(year)] if str(year) in downloaded_count.loc[zone] else 0
                mgrs_df.loc[zone, f'percentage_{year}'] = mgrs_df.loc[zone, f'downloaded_{year}'] / mgrs_df.loc[zone, f'sampled_{year}']
        withna_cols = [f'downloaded_{year}' for year in range(2019, 2023)] + [f'percentage_{year}' for year in range(2019, 2023)]
        mgrs_df[withna_cols] = mgrs_df[withna_cols].fillna(0)
        mgrs_df.to_parquet('~/scratch/sample_stats.parquet')
        return mgrs_df
        
    def _get_rh(self, group, rh_id):
        group.sort_values('in_partition_idx', inplace=True)
        idx = group['in_partition_idx'].to_list()
        zone = group.name[1:4]
        with h5py.File(self.h5_dir/f'{zone}.h5') as h5_file:
            data = h5_file[f'{group.name[4:]}/rhs'][idx, rh_id]
        data = pd.DataFrame(data, columns=[f'rh_{rh_id}'], index=group.index)
        group = pd.concat([group, data], axis=1)
        return group
    
    def _agg_rhs_per_zone(self, index_df, rh_id, partition_info:dict=None):
        part_idx = partition_info['number']
        # h5_file = h5py.File(self.h5_dir/f'{zone}.h5')
        index_df = index_df.reset_index(drop=True)
        # ddf = dd.from_pandas(index_df, npartitions=8)
        
        meta = {'path': str, 'in_partition_idx': int, f'rh_{rh_id}': float}
        ddf = index_df.groupby('path').apply(self._get_rh, rh_id, include_groups=False)
        # ddf = ddf.groupby('path').apply(self._get_rh, rh_id, meta=meta).compute()
        print('finish partition: ', part_idx)
        # print(ddf)
        ddf = ddf.reset_index(1).set_index('level_1')
        ddf = ddf.drop(columns=['in_partition_idx'])
        index_df = pd.concat([index_df, ddf], axis=1)

        # h5_file.close()
        return index_df

    def plot_boxplots(self, splits: Iterable=None, boxplot_dir: str=None, **kwargs):
        """
        Using bootstrap method to plot the boxplot of the relative heights.
        Extracts the relative heights from the h5 files and saves them as parquet files.
        Then plots the boxplot/violin plot of the relative heights.

        """
        boxplot_dir = Path(boxplot_dir).expanduser()
        import time
        from dask.distributed import Client, LocalCluster
        cluster = LocalCluster()
        client = Client(cluster)
        print(client)
        for split in splits:
            print(split)          
            if exists := os.path.exists(f'{str(self.save_dir)}/boxplot_stats_rhs_{split}.json'):
                print('loading RHs from json')
                with open(f'boxplot_stats_rhs_{split}.json', 'r') as f:
                    stats = json.load(f)
            else:
                data_dir = f'{str(self.save_dir)}/index_table_{split}'
                index_df_files = [f"{data_dir}/{f}" for f in os.listdir(data_dir)]
                self.index_df_files = sorted(index_df_files, key=natural_sort_key)
                index_df = dd.read_parquet(self.index_df_files, columns=['path', 'in_partition_idx'], aggregate_files=False)

                stats = []
                for i in range(101):
                    meta = {'path': str, 'in_partition_idx': int, f'rh_{i}': float}
                    rhs = index_df.map_partitions(self._agg_rhs_per_zone, rh_id=i, meta=meta).compute()
                    rhs = rhs.dropna()
                    rhs = rhs.drop(columns=['path', 'in_partition_idx'])
                    stats.extend(cbook.boxplot_stats(rhs, labels=[f'rh_{i}']))
                    print(f'finish rh_{i}\n', stats)
                    file = boxplot_dir/f'boxplot_stats_rhs_{split}.json'
                    if file.exists():
                        with open(file, 'r+') as f:
                            old = json.load(f)
                            old.extend(stats)
                            # Move the file pointer to the beginning
                            f.seek(0)
                            # Write the new data, overwriting the old content
                            json.dump(old, f, indent=4, default=convert_to_serializable)
                            # Truncate the file to remove any leftover old content
                            f.truncate()
                    else:
                        with open(file, 'w') as f:
                            json.dump(stats, f, indent=4, default=convert_to_serializable)
            fig, ax = plt.subplots()
            ax.bxp(stats, patch_artist=True, boxprops={'facecolor': 'bisque'})
            plt.savefig(f'{boxplot_dir}/RHs_boxplot_{split}.png')

    def plot_violins(self, splits: Iterable=None, **kwargs):
        """
        Using bootstrap method to plot the boxplot of the relative heights.
        Extracts the relative heights from the h5 files and saves them as parquet files.
        Then plots the boxplot/violin plot of the relative heights.

        """
        import time
        from dask.distributed import Client, LocalCluster
        cluster = LocalCluster()
        client = Client(cluster)
        print(client)
        for split in splits:
            print(split)          
            if exists := os.path.exists(f'{str(self.save_dir)}/rhs_{split}.csv'):
                print('loading RHs from csv')
                rhs = pd.read_csv(f'{str(self.save_dir)}/rhs_{split}.csv')
                plt.figure(figsize=(24,6))
                ax = sns.violinplot(data=rhs)
            else:
                data_dir = f'{str(self.save_dir)}/index_table_{split}'
                index_df_files = [f"{data_dir}/{f}" for f in os.listdir(data_dir)]
                self.index_df_files = sorted(index_df_files, key=natural_sort_key)
                index_df = dd.read_parquet(self.index_df_files, columns=['path', 'in_partition_idx'])

                for i in range(101): # not enough RAM for aggragating 101 RHs all at once
                    meta = {'path': str, 'in_partition_idx': int, f'rh_{i}': float}
                    rhs = index_df.map_partitions(self._agg_rhs_per_zone, rh_id=i, meta=meta).compute()
                    rhs = rhs.dropna()
                    rhs = rhs.drop(columns=['path', 'in_partition_idx'])
                    fig, ax = plt.subplots()
                    sns.violinplot(data=rhs, ax=ax)

                    violin_data = get_violin_stats(ax)
                    violin_data.to_csv(f'{str(self.save_dir)}/violin_data_{split}_rh{i}.csv', index=False)
                    plt.savefig(f'{self.save_dir}/RHs_violin_{split}_rh{i}.png')  

    def agg_rhs(self, zone, save_dir:Path):
        rhs = []
        with h5Dataset(self.h5_dir/f'{zone}.h5', mode='r') as ncds:
            with h5py.File(self.h5_dir/f'{zone}.h5',) as data:
                for group in _iter_nc_groups(ncds):
                    if len(group.split('/')) == 3:
                        rhs.append(data[f'{group}/rhs'][:])
        res = np.concatenate(rhs, axis=0)
        df = pd.DataFrame(res, columns=[f'rh{i}' for i in range(101)])
        df.to_parquet(f'{save_dir}/{zone}.parquet')
        return
    
    def plot_histgram(self, splits: Iterable=None):
        for split in splits:
            data = self._get_hist_counts(self.save_dir /f'hist_data_{split}.json', self.save_dir /f'index_table_{split}')
            for k, m in data.items():
                name = HIST_PARAMS[k]['name']
                bins = m['bins']
                counts = m['counts']
                plt.figure()
                if name == 'ESA World Cover':
                    x = ['no-data']+list(ESA_WC.keys())
                    width = 0.8
                else:
                    x = bins[:-1]
                    width = 0.8*(bins[1] - bins[0])
                plt.bar(x, counts, width=width, edgecolor='black', log=True)
                plt.xlabel(name)
                plt.ylabel('Number of samples')
                if name == 'ESA World Cover':
                    plt.xticks(rotation=45, ha='right')
                    plt.tick_params(axis='x', labelsize=8)
                plt.tight_layout()
                plt.savefig(self.save_dir/ f'histogram_{name}_{split}.png')

    def _get_hist_counts(self, file:Path, index_dir:Path):
        if file.exists():
            with open(file, 'r') as f:
                data = json.load(f)
            return data
        

        index_df_files = [str(index_dir/f) for f in os.listdir(index_dir)]
        self.index_df_files = sorted(index_df_files, key=natural_sort_key)
        index_df = dd.read_parquet(self.index_df_files, columns=['path', 'in_partition_idx'])
        meta = ('rhs', 'object')
        data = index_df.map_partitions(self._agg_zone, HIST_PARAMS, meta=meta).compute()
        
        # Following was used for aggregating data for the whole dataset, updated one is able to handle subsets
        # h5_files = self.h5_dir.glob('*.h5')
        # zones = [f.stem for f in h5_files]
        # data = db.from_sequence(zones).map(self._agg_zone, HIST_PARAMS)
        # data = data.compute()

        # init result dict
        res = {}
        for name in HIST_PARAMS.keys():
            res[name] = {
                'bins': HIST_PARAMS[name]['bins'],
                'counts': np.zeros(len(HIST_PARAMS[name]['bins'])-1)
            }
        for m in data:
            for c, name in enumerate(HIST_PARAMS.keys()):
                res[name]['counts'] += m[c].bincounts
          
        for name, m in res.items():
            m['counts'] = m['counts'].tolist() # to be able to serialize to json

        json_data = json.dumps(res, indent=4)
        with open(file, 'w') as f:
            f.write(json_data)
        return res

    def _agg_zone(self, index_df, hist_params, partition_info:dict=None):
        """
        Walk through all groups in given zone.h5, 
        and aggregate data/cols specified by hist_params for histogram plot.
        """
        part_idx = partition_info['number']
        zone = os.path.basename(self.index_df_files[part_idx]).split('.')[0]
        metrics = []
        for name, params in hist_params.items():
            metric = AverageMeter(bins=params['bins'], name=name)
            metrics.append(metric)
        with h5py.File(self.h5_dir/f'{zone}.h5',) as data:
            for name, group in index_df.groupby('path'):
                idx = group['in_partition_idx'].to_list()
                idx = sorted(idx)
                for m in metrics:
                    slic = hist_params[m.name]['slices']
                    m.update(data[f'{name[4:]}/{m.name}'][(idx, *slic)])   
        
        # Following was used for aggregating data for the whole dataset, updated one is able to handle subsets
        # with h5Dataset(self.h5_dir/f'{zone}.h5', mode='r') as ncds:
        #     with h5py.File(self.h5_dir/f'{zone}.h5',) as data:
        #         for group in _iter_nc_groups(ncds):
        #             if len(group.split('/')) == 3:
        #                 for m in metrics:
        #                     slic = hist_params[m.name]['slices']
        #                     m.update(data[f'{group}/{m.name}'][slic])                                                
        return metrics

def convert_to_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()  # Convert the numpy array to a list
    if isinstance(obj, np.float32):
        return float(obj)  # Convert float32 to Python float
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable") 

def get_violin_stats(ax):
    # Extract data from the violin plot
    violin_data = []
    for i, artist in enumerate(ax.findobj(lambda x: hasattr(x, 'get_paths'))):
        for path in artist.get_paths():
            vertices = path.vertices
            for vert in vertices:
                violin_data.append([i, vert[0], vert[1]])

    # Convert the data to a DataFrame
    df = pd.DataFrame(violin_data, columns=['violin', 'x', 'y'])
    return df

def plot_from_violin_stats(df):
    fig, ax = plt.subplots()

    # Plot each violin body using the saved data
    for i in df['violin'].unique():
        subset = df[df['violin'] == i]
        ax.fill(subset['x'], subset['y'], alpha=0.8, color=sns.color_palette("deep")[0], edgecolor="black", linewidth=1)

    # Match the Seaborn axis and grid style
    sns.despine(left=True)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color='gray', linestyle='dashed', linewidth=0.5)
    ax.xaxis.grid(False)

    return ax

@dataclass
class MyConfig:
    # index_dir: str = '~/scratch/data/split_test0.1_cal0.1_val0.1_seed42/test_index_table'#
    h5_dir: str = '~/data/GEDI'
    save_dir: str = '~/scratch/data/split_test0.1_cal0.1_val0.1_seed42'
    splits: list = field(default_factory=lambda: ['test', 'cal', 'val', 'train'])
    task: str = 'histogram'

cs = ConfigStore.instance()
cs.store(name="my_config", node=MyConfig)

@hydra.main(config_name="my_config", version_base="1.2")
def main(cfg: DictConfig) -> None:
    print(cfg)
    import time
    t0 = time.time()
    stats = Stats(cfg.h5_dir, cfg.save_dir)
    # stats.check_s2_values(cfg.h5_dir)
    if hasattr(stats, cfg.task):
        getattr(stats, cfg.task)(**cfg)


    print(f'time taken: {time.time() - t0}')
    # client.close()


if __name__ == '__main__':
    main()
    