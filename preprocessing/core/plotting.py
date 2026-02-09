from pathlib import Path
import hydra
from typing import Iterable
from dataclasses import dataclass, field
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig
import h5py
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import ScalarFormatter, MaxNLocator
from matplotlib.ticker import FormatStrFormatter
import matplotlib.colors as colors
import dask.dataframe as dd
from dask.utils import natural_sort_key
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
    def __init__(self, rh_table_fps:str=None):
        self.rh_table_fps = Path(rh_table_fps).expanduser()

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
    
    def plot_rh_distribution(self, rh_idx:int=None):
        name = self.rh_table_fps.name.split('_')[0]
        df = dd.read_parquet(self.rh_table_fps, columns=[f'rh{rh_idx}'])
        df = df.compute()
        rh = df[f'rh{rh_idx}'].values
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        bins = np.arange(0, 55, 5)
        counts, bin_edges = np.histogram(rh, bins=bins)
        counts_normalized = counts / counts.sum()
        ax.hist(bin_edges[:-1], bins=bins, weights=counts_normalized, alpha=0.5, label=f'RH{rh_idx}')

        # Normalize the counts so that they sum to 1
        

        # Display the normalized counts for each bin
        for count_norm, bin_edge in zip(counts_normalized, bin_edges[:-1]):
            # Position the text at the center of the bin and slightly above the bar
            x_pos = bin_edge + (bins[1] - bins[0]) / 2  # Center of the bin
            y_pos = count_norm + 0.01  # Slightly above the bar
            ax.text(x_pos, y_pos, f'{count_norm:.3f}', ha='center', va='bottom', fontsize=9)

        # Customize x and y ticks
        ax.set_xticks(bins)  # Set x-ticks at bin edges
        ax.set_yticks(np.arange(0, max(counts_normalized)+0.1, 0.1))

        # Add labels and title
        ax.set_xlabel('Relative height', fontsize=12)
        ax.set_ylabel('Normalized Counts', fontsize=12)
        ax.set_title('RH98 distribution', fontsize=14)
        ax.legend()
        plt.savefig(f'output/rh{rh_idx}_distribution_{name}.png', dpi=300, bbox_inches='tight', transparent=False)


    def plot_split_distribution(self, splits: Iterable=None, split_dir:str=None):
        split_dir = Path(split_dir).expanduser()
        for split in splits:
            self.h5_dir = split_dir / f'{split}_h5s'
            data = self._get_hist_counts(split_dir /f'{split}_distribution_hist_data.json', split_dir /f'index_table_{split}')
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
                plt.savefig(split_dir/ f'histogram_{name}_{split}.png')

    def _get_hist_counts(self, file:Path, index_dir:Path):
        if file.exists():
            with open(file, 'r') as f:
                data = json.load(f)
            return data
        

        index_df_files = [str(index_dir/f) for f in os.listdir(index_dir)]
        self.index_df_files = sorted(index_df_files, key=natural_sort_key)
        # self.index_df_files = [str(f) for f in self.index_df_files if '04L' in f]
        index_df = dd.read_parquet(self.index_df_files, columns=['path', 'in_partition_idx'])
        meta = ('rhs', 'object')
        # index_df = index_df.get_partition(0).compute()
        # index_df = index_df.compute()
        # test = self._agg_zone(index_df, HIST_PARAMS, partition_info={'number': 0})
        # import ipdb; ipdb.set_trace()
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
        if index_df.empty:
            return []
        
        index_df = index_df.sort_values(['path', 'in_partition_idx'])
        index_df['in_partition_idx'] = index_df.groupby('path').cumcount()
        metrics = []
        for name, params in hist_params.items():
            metric = AverageMeter(bins=params['bins'], name=name)
            metrics.append(metric)
        for name, group in index_df.groupby('path'):
            zone = name[1:4]
            with h5py.File(self.h5_dir/f'{zone}.h5',) as data:
                idx = group['in_partition_idx'].to_list()
                idx = sorted(idx)
                for m in metrics:
                    slic = hist_params[m.name]['slices']
                    try:
                        m.update(data[f'{name[4:]}/{m.name}'][(idx, *slic)]) 
                    except KeyError:
                        # import ipdb; ipdb.set_trace()
                        print(f'{name}/{m.name} not found', self.h5_dir/f'{zone}.h5')
                        raise ValueError
        
        # Following was used for aggregating data for the whole dataset, updated one is able to handle subsets
        # with h5Dataset(self.h5_dir/f'{zone}.h5', mode='r') as ncds:
        #     with h5py.File(self.h5_dir/f'{zone}.h5',) as data:
        #         for group in _iter_nc_groups(ncds):
        #             if len(group.split('/')) == 3:
        #                 for m in metrics:
        #                     slic = hist_params[m.name]['slices']
        #                     m.update(data[f'{group}/{m.name}'][slic])                                                
        return metrics


@dataclass
class MyConfig:
    # index_dir: str = '~/scratch/data/split_test0.1_cal0.1_val0.1_seed42/test_index_table'#
    rh_table_fps: str = '~/data/train_subsets/train*_filtered_v1.parquet'
    task: str = 'plot_rh_distribution'

cs = ConfigStore.instance()
cs.store(name="my_config", node=MyConfig)

@hydra.main(config_name="my_config", version_base="1.2")
def main(cfg: DictConfig) -> None:
    print(cfg)
    import time
    t0 = time.time()
    stats = Stats(cfg.rh_table_fps)
    # stats.check_s2_values(cfg.h5_dir)
    # if hasattr(stats, cfg.task):
    #     getattr(stats, cfg.task)(rh_idx = 98)
    split_dir = '~/data/gvs/split_test0.1_cal0.1_val0.1_seed42_v1'
    stats.plot_split_distribution(splits=['train', 'cal', 'val', 'test'], split_dir=split_dir)


    print(f'time taken: {time.time() - t0}')
    # client.close()


if __name__ == '__main__':
    main()
    