#%%
import re
import json
from datetime import datetime, timedelta
from pathlib import Path
import ipdb
import hydra

import pandas as pd
import dask
import dask.dataframe as dd

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

def addTrackNumberAndRepartition(dataFolder=None, partition_size='64K'):
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
    print(f'finish zone {dataFolder.stem}')

@hydra.main(config_path="../config", config_name="s2_download", version_base="1.2")
def main(cfg):
    from dask.distributed import Client, LocalCluster
    cluster = LocalCluster()
    client = Client(cluster, asynchronous=True)
    if cfg.zone is not None:
        addTrackNumberAndRepartition(Path.home() / f'GEDI{cfg.year}/{cfg.zone}', cfg.partition_size)
    else:
        zone_paths = [f for f in (Path.home() / f'GEDI{cfg.year}').iterdir() if f.is_dir()]
        for zone in zone_paths:
            addTrackNumberAndRepartition(zone, cfg.partition_size)

#%%
if __name__ == '__main__':
    main()