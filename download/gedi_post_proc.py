#%%
import re
import json
from datetime import datetime, timedelta
from pathlib import Path
import ipdb

import pandas as pd
import dask
import dask.dataframe as dd
import geopandas as gpd
import dask_geopandas as dgd
from shapely.geometry import shape

from const import dtypes

#%%
GEDI_START = pd.Timestamp('2019-01-01')

def getDate(year, doy):
    start_of_year = datetime(int(year), 1, 1)
    calendar_date = start_of_year + timedelta(days=int(doy) - 1)
    return calendar_date.strftime('%Y-%m-%d')


def addTrackNumberForFile(csvFile):
    search = re.search(r'(\d{4})(\d{3})', csvFile.stem)
    df = dd.read_csv(csvFile, dtype=dtypes, usecols=list(dtypes.keys()))
    df['date'] = getDate(search.group(1), search.group(2))
    df['track_id'] = re.search(r'(\d{13})_O(\d{5})_.*_T(\d{5})',
                               csvFile.stem)[0]

    df['x'] = df['.geo'].apply(lambda x: json.loads(x)['coordinates'][0],
                                      meta=('geometry', 'float'))
    df['y'] = df['.geo'].apply(lambda x: json.loads(x)['coordinates'][1],
                                      meta=('geometry', 'float'))
    df = df.drop('.geo', axis=1)
    df.to_parquet(csvFile.parent, name_function=lambda x: csvFile.stem + '.parquet')
    return


def addTrackNumber(zone=None):
    zone = zone or '**'
    dataFolder = Path.home() / 'GEDI2019'
    csvFiles = dataFolder.glob(f'{zone}/GEDI02*.csv')
    res = []
    for file in csvFiles:
        res.append(dask.delayed(addTrackNumberForFile)(file))
    dask.compute(res)

def addTrackNumberAndRepartition(zone=None, partition_size='64K'):
    zone = zone or '**'
    dataFolder = Path.home() / 'GEDI2019' / zone
    df = dd.read_csv(dataFolder / '*.csv', dtype=dtypes, usecols=list(dtypes.keys()))
    df['x'] = df['.geo'].apply(lambda x: json.loads(x)['coordinates'][0],
                                      meta=('geometry', 'float'))
    df['y'] = df['.geo'].apply(lambda x: json.loads(x)['coordinates'][1],
                                      meta=('geometry', 'float'))
    df = df.drop('.geo', axis=1)
    df['date'] = dd.to_timedelta(df['delta_time'], unit='S') + GEDI_START
    df['date'] = df['date'].dt.strftime('%Y-%m-%d')#.astype(str)
    df = df.repartition(partition_size=partition_size)
    df.to_parquet(dataFolder, name_function=lambda x: f'partition_{x}.parquet')

#%%
if __name__ == '__main__':
    from dask.distributed import Client, LocalCluster
    cluster = LocalCluster()
    client = Client(cluster, asynchronous=True)
    # futures = client.submit(addTrackNumber, '56H')
    addTrackNumberAndRepartition('20M', partition_size='4M')
    # client.gather(futures)