import dask
import pandas as pd
from pathlib import Path
import dask_geopandas as dgp
import geopandas as gpd

def repartition_index_table(parquet_dir: str=None, save_dir: str=None, based_on_col: str=None):
    parquet_dir = Path(parquet_dir).expanduser()
    save_dir = Path(save_dir).expanduser()
    save_dir.mkdir(parents=True, exist_ok=True)
    parquet_files = list(parquet_dir.glob('*.parquet'))

    ddf = dgp.read_parquet(parquet_files, gather_spatial_partitions=False)
    
    def reindex_in_partition_idx(x):
        x['in_partition_idx'] = x.groupby('path').cumcount()
        return x
    
    ddf = ddf.map_partitions(reindex_in_partition_idx, meta=ddf._meta.copy())
    
    def save_partition(x):
        name = x.name
        x = gpd.GeoDataFrame(x)
        x.to_parquet(save_dir / f'{name}.parquet')
    
    ddf.groupby(based_on_col).apply(save_partition, meta=('object', None)).compute()