from pathlib import Path
import dask
import dask.dataframe as dd
import geopandas as gpd
import re
from datetime import datetime, timedelta
import json
from shapely.geometry import shape
import dask_geopandas as dgd
import ipdb

def getDate(year, doy):
    start_of_year = datetime(int(year), 1, 1)
    calendar_date = start_of_year + timedelta(days=int(doy) - 1)
    return calendar_date.strftime('%Y-%m-%d')


def addTrackNumberForFile(csvFile):
    search = re.search(r'(\d{4})(\d{3})', csvFile.stem)
    df = dd.read_csv(csvFile, dtype={'system:index': 'object'})
    df['date'] = getDate(search.group(1), search.group(2))
    df['track_id'] = re.search(r'(\d{13})_O(\d{5})_.*_T(\d{5})',
                               csvFile.stem)[0]

    df['geometry'] = df['.geo'].apply(lambda x: shape(json.loads(x)),
                                      meta=('geometry', object))
    df = df.drop('.geo', axis=1)
    ddf = dgd.from_dask_dataframe(df).compute()
    ddf.to_parquet(csvFile.with_suffix('.parquet'))
    return


def addTrackNumber():
    dataFolder = Path.home() / 'GEDI2019'
    csvFiles = dataFolder.glob('**/GEDI02*.csv')
    res = []
    for file in csvFiles:
        res.append(dask.delayed(addTrackNumberForFile)(file))
    dask.compute(res)