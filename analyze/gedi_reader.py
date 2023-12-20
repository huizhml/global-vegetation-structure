from multiprocessing import freeze_support
import os
import json
from time import time
import pystac_client
import planetary_computer
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

import torch
import getpass
import itertools as it
import requests
import shutil
import tempfile
import os.path
from pprint import pprint
from urllib.parse import urljoin

# from pystac_client import Client
from pystac import ExtensionNotImplemented
from pystac.extensions.scientific import ScientificExtension

from dask.distributed import Client

from dask_jobqueue import SLURMCluster
import pandas as pd
import dask_geopandas as dgpd
import dask.dataframe as dd
from shapely.geometry import shape
import hvplot.pandas
import hvplot.dask
import numpy as np


class GEDIReader:
    def __init__(self, dataFolder):
        self.dataFolder = Path.home() / dataFolder
        self.year = int(dataFolder[-4:])
        self.keyRhs = ['rh10', 'rh25', 'rh50', 'rh75', 'rh90', 'rh98']

    def removeEmptyCSV(self, addParentFolder=False):
        if addParentFolder:
            for folder in os.listdir(self.dataFolder):
                if os.path.isdir(self.dataFolder / folder):
                    sampled = pd.read_json(self.dataFolder / folder / f'sampled{self.year}.json')
                    sampled['filename'] = sampled['filename'].apply(lambda x: f'{folder}/{x}')
                    sampled.to_json(self.dataFolder / folder / f'sampled{self.year}.json', orient='columns') # orient='records'

        sampleAssetTable = dd.read_json(self.dataFolder / f'*/sampled{self.year}.json', orient='columns') # orient='values'
        emptyFiles = sampleAssetTable[sampleAssetTable['sampled'] == 0]
        emptyFiles = emptyFiles.compute()
        emptyFiles['filename'] = emptyFiles['filename'].apply(lambda x: self.dataFolder / f'{x}.csv')
        # emptyFiles['filename'].apply(lambda x: os.remove(x))
        if len(emptyFiles) == 0:
            print(f'No empty files in {self.dataFolder}')
            return
        for i, file in enumerate(emptyFiles['filename']):
            try:
                pd.read_csv(file)
            except:
                print(f'{i}, Cannot read {file}, is empty or corrupted. Now removing...')
                os.remove(file)
            
    def readCSV(self):
        print('reading all csv files using dask dataframe...')
        t0 = time()
        ddf = dd.read_csv(self.dataFolder / f'21M/*.csv', usecols=['delta_time', 'pft_class', '.geo'], blocksize=64e6)
        ddf['geometry'] = ddf['.geo'].apply(lambda x: shape(json.loads(x)), meta=('geometry', 'object'))
        geodf = dgpd.from_dask_dataframe(ddf)
        # geodf = geodf.compute()
        plot = geodf.compute().hvplot(global_extent=True, frame_height=450, tiles=True, rasterize=True, cmap='plasma')
        hvplot.save(plot, 'sampledmap21M.html')
        print('Time to read all csv files: ', time() - t0)
        return 'done'
    

    def histFromValueCounts(self, valueCountFile=None):
        for rh in self.keyRhs:
            with open(f'output/value_counts_{rh}.csv/0.part', 'r') as f:
                valueCounts = f.readlines()
            valueCounts = [{'value': int(x.split(',')[0]), 'count': int(x.split(',')[1])} for x in valueCounts[1:]]
            valueCounts = pd.DataFrame(valueCounts)
            valueCounts[rh].hist(bins=np.arange(0, 100))

    def plotCountMap(self):
        sampledTable = pd.read_json(self.dataFolder / f'SampledTable{self.year}.json')

    def plotHistogram(self, ):
        keyRhs = ['rh10', 'rh25', 'rh50', 'rh75', 'rh90', 'rh98']
        dataFolder = Path('output')
        for file in dataFolder.glob(f'value_counts_*.csv/0.part'):
            with open(file, 'r') as f:
                counts = f.readlines()[1:]
                counts = [x.strip().split(',') for x in counts]
                counts = np.array(counts, dtype=float)

            number_of_bins = 100
            min_value = -10
            max_value = 90
            bins = np.linspace(min_value, max_value, number_of_bins)

            indices = np.digitize(counts[:, 0], bins)

            # Now we sum up the counts for each bin
            binned_counts = np.zeros(len(bins))

            for i, count in enumerate(counts[:, 1]):
                bin_index = indices[
                    i] - 1  # -1 because `digitize` bins are 1-indexed
                binned_counts[bin_index] += count

            # Plotting the histogram with the binned counts
            # plt.figure()
            plt.bar(bins,
                    binned_counts,
                    width=np.diff(bins)[0],
                    align='edge',
                    alpha=0.7,
                    label=f'RH{file.parent.name[-6:-4]}')
        plt.xlabel('Height')
        plt.ylabel(f'Number of Points (total: {counts[:,1].sum():.2e})')
        plt.title(f'RH{file.parent.name[-6:-4]}')
        plt.legend()
        plt.savefig(file.parent.parent / 'histogram.png')

    def checkObitUniqueness(self):
        '''if each track has one unique orbit number, the number of unique orbit numbers should be the same as the number of tracks (csv files)'''
        cvsFiles = len(list(self.dataFolder / f'GEDI02**/*.csv'))
        print(f'number of csv files: {cvsFiles}')
        ddf = dd.read_csv(self.dataFolder / f'**/*.csv', usecols=['orbit_number'], blocksize=64e6)
        obrbitNumbers = ddf['orbit_number'].value_counts().compute()
        obrbitNumbers.to_csv(self.dataFolder / 'orbitNumbers.csv')
        


if __name__ == '__main__':
    # from dask_cuda import LocalCUDACluster
    from dask.distributed import LocalCluster
    cluster = LocalCluster(n_workers=16)
    client = Client(cluster)
    print(f"/proxy/{client.scheduler_info()['services']['dashboard']}/status")

    import azure.storage.blob
    from pathlib import Path
    import segmentation_models_pytorch

    import warnings

    # ignore SyntaxWarning in pretrainedmodels
    warnings.filterwarnings("ignore", category=SyntaxWarning)

    gediReader = GEDIReader('GEDI2019')
    gediReader.readCSV()