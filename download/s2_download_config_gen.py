import os
import math
import logging
from pathlib import Path
import pandas as pd
import pyarrow.parquet as pq
import geopandas as gpd
import dask_geopandas as dgp
import hydra
import dask
logger = logging.getLogger(__name__)

def repartition(year_folder, zone, partition_size='128K'):
    ddf = dgp.read_parquet(year_folder / zone / 'GEDI*.parquet')
    ddf = ddf.repartition(partition_size=partition_size)
    ddf.to_parquet(year_folder / f'{zone}', name_function=lambda x: f'partition_{x}.parquet')

def aggregate_zones(group):
    if group.index[0] <= 16:
        zones = ','.join(group['zone'])
        res = group.iloc[0, :]
        # logger.info(res)
        res['zone'] = zones
        return res.to_frame().T
    else:
        return group

def gen_config_for_zone(zone_folder, year):
    npartitions = len(list(zone_folder.glob('partition*.parquet')))
    if npartitions >=500:
        n_cores = 64
    else:
        n_cores = math.floor(math.log(npartitions, 2)) // 2
        n_cores = max(n_cores, 1)
        n_cores= min(2**n_cores, 128) # maximum #cores we can get
    n_parallel = n_cores * 2

    return n_cores, npartitions, n_parallel


@hydra.main(config_path='../config', config_name='s2_download', version_base='1.2')
def main(cfg):
    from dask.distributed import LocalCluster, Client, wait
    cluster = LocalCluster()
    client = Client(cluster)
    if isinstance(cfg.year, int):
        years = [cfg.year]
    else:
        years = cfg.year

    for year in years:
        data_folder = Path.home() / cfg.data_dir/ str(year)
        config = []
        for zone in os.listdir(data_folder):
            if not os.path.isdir(data_folder / zone) or (data_folder/f'{zone}.parquet').exists():
                continue
            logger.info(f'repartition {zone}...')
            repartition(data_folder, zone, cfg.partition_size)

            s2_data_dir = Path.home() / cfg.save_dir
            if (s2_data_dir / f'{zone}_{year}_done').exists():
                continue
            logger.info(f'generate download config for {zone} {year}...')
            line = gen_config_for_zone(data_folder/zone, year)
            config.append(line+[zone])
    df = pd.DataFrame(config, columns=['ncores', 'npartitions', 'nparallel', 'zone'])
    df = df.set_index('ncores').groupby('ncores').apply(aggregate_zones)
    df = df.reset_index().drop(columns='level_1')
    df.to_csv(data_folder /'download_config.csv', index=False, sep=';')
    client.close()

if __name__ == '__main__':
    main()