import geopandas as gpd
import dask.dataframe as dd
from dask.distributed import Client, LocalCluster


def sampling_stats(index_table:str='~/scratch/data/index_table', mgrs_file:str='~/scratch/mgrs_with_nbest_v2.parquet'):
    
    cluster = LocalCluster()
    client = Client(cluster)
    print(client)
    index_df = dd.read_parquet(f'{index_table}/*.parquet')
    splits = index_df.path.str.split('/')
    index_df['zone'] = splits.str[6:9]
    index_df['year'] = splits.str[13:17]
    downloaded_count = index_df.groupby(['zone', 'year']).size()
    downloaded_count = downloaded_count.compute()
    mgrs_df = gpd.read_parquet(mgrs_file)
    mgrs_df = mgrs_df.drop(columns=['tracks'] + [f'tracks_{year}' for year in range(2019, 2023)])
    mgrs_df = mgrs_df.set_index('MGRS_UTM')
    for zone in downloaded_count.index.get_level_values('zone'):
        for year in range(2019,2023):
            mgrs_df.loc[zone, f'downloaded_{year}'] =  downloaded_count.loc[zone, str(year)] if str(year) in downloaded_count.loc[zone] else 0
            mgrs_df.loc[zone, f'percentage_{year}'] = mgrs_df.loc[zone, f'downloaded_{year}'] / mgrs_df.loc[zone, f'sampled_{year}']
    withna_cols = [f'downloaded_{year}' for year in range(2019, 2023)] + [f'percentage_{year}' for year in range(2019, 2023)]
    mgrs_df[withna_cols] = mgrs_df[withna_cols].fillna(0)
    mgrs_df.to_parquet('~/scratch/sample_stats.parquet')
    return mgrs_df
        