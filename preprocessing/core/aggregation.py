
from typing import List
from pathlib import Path
import pandas as pd
import numpy as np
import geopandas as gpd
import dask.dataframe as dd
from tqdm import tqdm
import glob
from ffcv.loader import Loader, OrderOption
from const import BIOMES


def aggregate_gedi_by_biome(beton_fps: List[str], group_by_biome=False):
    '''
    Aggregate GEDI data (RHs + GEDI attributes) {train*}.beton to a single parquet file
    '''
    beton_fps = glob.glob(str(Path(beton_fps).expanduser()))
    print(beton_fps)
    cols = [f'rh{i}' for i in range(101)] + ['wc', 'slope', 'lon', 'lat']
    data_dir = Path(beton_fps[0]).parent.parent
    ecoregions = gpd.read_file( '~/data/GEDI/ecoregions/wwf_terr_ecos.shp')
    batch_size = 100 if 'debug' in beton_fps[0] else 4096
    for fp in beton_fps:
        file = Path(fp).with_suffix('.parquet')
        # if file.exists():
        #     print('file exists', file)
        #     continue
        loader = Loader(fp, batch_size=batch_size, num_workers=1,
                        distributed=False, batches_ahead=3,
                        order=OrderOption.SEQUENTIAL, os_cache=False, drop_last=False)

        data = []
        for batch in tqdm(loader):
            _, rhs, wc, slope, lon, lat = batch
            batch_ = np.concatenate([rhs, wc[...,7:8,7], slope[...,7:8,7], lon[:,7:8], lat[:, 7:8]], axis=1)
            data.append(batch_)
        data = np.concatenate(data, axis=0)
        print(data.shape)
        df = pd.DataFrame(data, columns=cols)
        df = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat))
        df = df.set_crs(epsg=4326)
        df = df.to_crs(epsg=3857)
        df = gpd.sjoin(df, ecoregions, how='left', predicate='within')
        df = df[cols + ['BIOME']]
        df.to_parquet(file)
        print('saved to', file)
    print('Grouping by biome')
    df_fp = glob.glob(str(file.parent / '*.parquet'))
    
    if group_by_biome:
        group_df_by_biome(df_fp)


def group_df_by_biome(df_fps: List[str] = None):
    """
    Group the dataframe by biome, and save parquets for each biome
    """
    import dask.dataframe as dd
    ddf = dd.read_parquet(df_fps)
    data_dir = Path(df_fps[0]).parent
    df = ddf.compute()
    print(df.BIOME.unique())

    def save_parquet(x):
        print(x)
        if x.name is not None and int(x.name) < 15:
            name = f'rhs_attrs_{BIOMES[int(x.name)-1].replace(" ", "_").replace("&", "and")}.parquet'
            x.to_parquet(data_dir / name)
        else:
            x.to_parquet(data_dir / f'rhs_attrs_biome_{x.name}.parquet')
    df.groupby('BIOME').apply(save_parquet)

