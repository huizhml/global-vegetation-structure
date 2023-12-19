#%%
import re
import json
from datetime import datetime, timedelta
from pathlib import Path
import ipdb

import dask
import dask.dataframe as dd
import geopandas as gpd
import dask_geopandas as dgd
from shapely.geometry import shape

from const import dtypes

#%%

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
    df.to_parquet(csvFile.with_suffix('.parquet'))
    return


def addTrackNumber():
    dataFolder = Path.home() / 'GEDI2019'
    csvFiles = dataFolder.glob('**/GEDI02*.csv')
    res = []
    for file in csvFiles:
        res.append(dask.delayed(addTrackNumberForFile)(file))
    # dask.compute(res)
#%%
if __name__ == '__main__':
    addTrackNumber()   