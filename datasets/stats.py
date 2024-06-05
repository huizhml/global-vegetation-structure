from pathlib import Path
import hydra
import h5py
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import dask
import seaborn as sns
import dask.bag as db
import dask.array as da
import xarray as xr
import dask.dataframe as dd
from datatree.io import _iter_nc_groups
from h5netcdf.legacyapi import Dataset as h5Dataset
from const import ESA_WC
import os
import json
os.environ['BOKEH_ALLOW_WS_ORIGIN'] = 'www.lumi.csc.fi'

HIST_PARAMS = {
    # keys from the h5 data
    'rhs':             {'bins': np.arange(0, 100, 2).tolist(), 'name': 'RH98', 'slices': (slice(None), 98)},
    'delta_day':       {'bins': np.arange(0, 375, 10).tolist(), 'name': 'Delta Days', 'slices': (slice(None))},
    'defective_cover': {'bins': np.arange(0, 0.9, 0.05).tolist(), 'name': 'Defective Cover', 'slices': (slice(None))},
    'image':           {'bins': [0]+list(ESA_WC.values())+[110], 'name': 'ESA World Cover', 'slices': (slice(None), 13, 7, 7)},
    'slope':           {'bins': np.arange(0, 92, 2).tolist(), 'name': 'Slope [COPERNICUS GLO-30 & SRTM]', 'slices': (slice(None), 7, 7)}
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
    def __init__(self, h5_dir:str='~/data/GEDI', mgrs_file:str='~/data/mgrs_zones.json'):
        self.h5_dir = Path(h5_dir).expanduser()

    def plot_map(self):
        pass

    def read_h5(self, index_df):

        if not hasattr(self, 'h5_file'):
            self.h5_file = h5py.File('/users/zhanghui/flash/data/GVS.h5', 'r')
        try:
            index_df.name in self.h5_file
        except:
            print(index_df)
        # #  TMP
        # if not index_df.name in self.h5_file:
        #     return

        idx = index_df['in_partition_idx'].to_list()
        idx = sorted(idx)
        rhs = self.h5_file[f'{index_df.name}/rhs'][idx]
        return rhs.tolist()



    def boxplot(self, subset_size:int=1e4, repeat:int=10, seed:int=0):
        """
        Using bootstrap method to plot the boxplot of the relative heights.
        Extracts the relative heights from the h5 files and saves them as parquet files.
        Then plots the boxplot/violin plot of the relative heights.

        """
        import time
        from dask.distributed import Client, LocalCluster
        cluster = LocalCluster(n_workers=8)
        client = Client(cluster)
        print(client)

        index_df = dd.read_parquet(f"{str(self.h5_dir.parent)}/index_table/*.parquet")
        index_df = index_df.compute()
        index_df = index_df.sample(frac=0.01, random_state=seed)
        # index_df = index_df.groupby('path')
        index_df = dd.from_pandas(index_df, npartitions=32)
        rhs = index_df.groupby('path').apply(self.read_h5,  meta=('rh', object)).compute()
        # rhs = index_df.map_partitions(self.read_h5, meta=('rh', 'object')).compute()
        rhs = rhs.dropna()
        rhs = [json.loads(rh) for rh in rhs]
        rhs = np.concatenate(rhs, axis=0)
        rhs_df = pd.DataFrame(rhs, columns=[f'rh{i}' for i in range(101)])
        rhs_df.to_csv(f'/users/zhanghui/scratch/data/rhs_sample0.01_{seed}.csv')
        plt.figure(figsize=(24,6))
        ax = sns.boxenplot(data=rhs_df)
        plt.savefig(f'outputs/boxenplot_sample0.01_{seed}.png')
              

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
    
    def plot_histgram(self):
        data = self.get_hist_counts(self.h5_dir.parent /'hist_data.json')
        plt.figure()
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
            plt.savefig(f'outputs/histogram_{name}.png')

    def get_hist_counts(self, file:Path):
        if file.exists():
            with open(file, 'r') as f:
                data = json.load(f)
            return data
        
        import time
        from dask.distributed import Client, LocalCluster
        cluster = LocalCluster(n_workers=32)
        client = Client(cluster)
        print(client)

        t0 = time.time()
        h5_files = self.h5_dir.glob('*.h5')
        zones = [f.stem for f in h5_files]
        data = db.from_sequence(zones).map(self.agg_zone, HIST_PARAMS)
        data = data.compute()
        print(f'time taken: {time.time() - t0}')
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

    def agg_zone(self, zone, hist_params):
        """
        Walk through all groups in given zone.h5, 
        and aggregate data/cols specified by hist_params for histogram plot.
        """
        metrics = []
        for name, params in hist_params.items():
            metric = AverageMeter(bins=params['bins'], name=name)
            metrics.append(metric)
        with h5Dataset(self.h5_dir/f'{zone}.h5', mode='r') as ncds:
            with h5py.File(self.h5_dir/f'{zone}.h5',) as data:
                for group in _iter_nc_groups(ncds):
                    if len(group.split('/')) == 3:
                        for m in metrics:
                            slic = hist_params[m.name]['slices']
                            m.update(data[f'{group}/{m.name}'][slic])                                                
        return metrics



if __name__ == '__main__':
    stats = Stats()
    stats.boxplot(seed=3)
    