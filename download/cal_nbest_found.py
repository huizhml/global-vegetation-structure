from pathlib import Path
import numpy as np
import geopandas as gpd
import dask
import dask_geopandas as dgp
import dask.array as da

ASSET_TABLE_DT = np.dtype('U28')
data_dir = Path('~/scratch/GEDI_with_s2_candidates_and_best').expanduser()
mgrs_df = gpd.read_parquet('~/GEDI/mgrs_with_count_orbits_and_sampled.parquet')

years = np.arange(2019, 2023)
zone_table = np.full((len(mgrs_df), len(years)), None, dtype=ASSET_TABLE_DT)
for i, zone in enumerate(mgrs_df['MGRS_UTM']):
    for j, year in enumerate(years):
        if (data_dir/f"{year}/{zone}").exists() and any((data_dir/f"{year}/{zone}").iterdir()):
            zone_table[i, j] = f"{year}/{zone}/partition_*.parquet"
            

zone_table_dask = da.from_array(
    zone_table,
    chunks=(1,1),
    # inline_array=True,
    name="zone-table-" + dask.base.tokenize(zone_table),
)

def count_nbest(path):
    if path == 'None' or path.size==0 or len(path[0,0])<28:
        return np.array([[-1]])
    df = dgp.read_parquet(data_dir/path[0][0], gather_spatial_partitions=False, columns=['best_s2'])
    df = df.dropna(subset=['best_s2'])
    sampled = df.map_partitions(len).compute()
    sampled = sampled.sum().item()
    print(path, sampled)
    return np.array([[sampled]])

count_arr = zone_table_dask.map_blocks(count_nbest).compute()
for i, year in enumerate(years):
    mgrs_df[f'nbest_{year}'] = count_arr[:, i]
    mgrs_df[f'enough_{year}'] = mgrs_df[f'nbest_{year}'] >= mgrs_df['nwant']
    mgrs_df[f'percentage_{year}'] = mgrs_df[f'nbest_{year}'] / mgrs_df['nwant']

mgrs_df.to_parquet('~/GEDI/mgrs_with_nbest.parquet')

