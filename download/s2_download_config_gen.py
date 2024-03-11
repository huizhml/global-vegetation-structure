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
import ipdb
logger = logging.getLogger(__name__)

def _repartition(year_folder, zone, partition_size='128K'):
    ddf = dgp.read_parquet(year_folder / zone / 'GEDI*.parquet')
    ddf = ddf.repartition(partition_size=partition_size)
    ddf.to_parquet(year_folder / f'{zone}', name_function=lambda x: f'partition_{x}.parquet')

def aggregate_zones(group):
    if group.index[0] <= 16:
        zones = ','.join(group['MGRS_UTM'])
        res = group.iloc[0, :]
        # logger.info(res)
        res['MGRS_UTM'] = zones
        return res.to_frame().T
    else:
        return group

def gen_config_for_zone(zone_folder, year):
    npartitions = len(list(zone_folder.glob('partition*.parquet')))
    if npartitions >=500:
        n_cores = 64
    elif npartitions >= 300:
        n_cores = 32
    else:
        n_cores = math.floor(math.log(npartitions, 2)) // 2
        n_cores = max(n_cores, 1)
        n_cores= min(2**n_cores, 128) # maximum #cores we can get
    n_parallel = n_cores * 2

    return [n_cores, npartitions, n_parallel]


def repartition(cfg):
    from dask.distributed import LocalCluster, Client
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
            s2_data_dir = Path.home() / cfg.save_dir
            if not os.path.isdir(data_folder / zone):
                continue
            if cfg.rewrite:
                logger.info(f'remove partitions in {zone}...')
                os.system(f'rm -rf {data_folder / zone}/partition*')
            if not (data_folder / zone / 'partition_0.parquet').exists():
                logger.info(f'repartition {zone}...')
                _repartition(data_folder, zone, cfg.partition_size)

            s2_data_dir = Path.home() / cfg.save_dir
            if (s2_data_dir / f'{zone}_{year}_done').exists():
                continue
            logger.info(f'generate download config for {zone} {year}...')
            line = gen_config_for_zone(data_folder/zone, year)
            config.append(line+[zone])
        df = pd.DataFrame(config, columns=['ncores', 'npartitions', 'nparallel', 'MGRS_UTM'])
        mgrs_df = pd.read_csv(Path.home() / 'GEDI/mgrs_sampled.csv')
        pd.merge(df, mgrs_df, on='MGRS_UTM', how='left').to_csv(data_folder /'download_config_count.csv', index=False, sep=';')
        df = df.set_index('ncores').groupby('ncores').apply(aggregate_zones)
        df = df.reset_index()
        df.to_csv(data_folder /'download_config.csv', index=False, sep=';')
    client.close()


def queue_zones_for_slurm_job(max_time, df, name='hendrix', years=['2019', '2020', '2021', '2022']):
    # queue zones for each slurm task until the task capacity(7 days for hendrix & 3 days for LUMI) is reached
    years = ','.join(years)
    task_cap = max_time * 24 * 3600 * 10 / 0.703 # the number of locations (divide by 0.703 to be comparable to landmass) that one slurm task can download within the time limit
    acc_cap = 0
    acc_zones = []
    jobId = 1
    for i, (idx, row) in enumerate(df.iterrows()):
        if acc_cap >= task_cap:
            with open(Path.home()/f'GEDI/download_job_{name}_{jobId}.txt', 'w') as f:
                f.write(f'{years} {",".join(acc_zones)}')
            acc_cap = 0
            acc_zones = []
            jobId += 1
        else:
            acc_cap += row['landmass'] * 4 # cap needed for 4 years
            acc_zones.append(row['MGRS_UTM'])
        if i == len(df) - 1 and len(acc_zones) > 0:
            with open(Path.home()/f'GEDI/download_job_{name}_{jobId}.txt', 'w') as f:
                f.write(f'{years} {",".join(acc_zones)}')


def split_zones():
    years = ['2019', '2020', '2021', '2022']
    years = ','.join(years)
    filepath = Path.home() / 'GEDI/mgrs_sampled.csv'
    sampled_df = pd.read_csv(filepath)
    # schedule zones with less than 200K locations on hendrix
    hendrix = sampled_df[sampled_df['landmass']<= 200_000/0.703]
    merged_zones = hendrix[hendrix['landmass']<= 16_000/0.703]['MGRS_UTM'].values
    with open(Path.home()/'GEDI/download_job_hendrix_0.txt', 'w') as f:
        f.write(f'{years} {",".join(merged_zones)} True')
    hendrix = hendrix[hendrix['landmass']> 16_000/0.703]
    # queue zones for each slurm task until the task capacity(7 days) is reached
    queue_zones_for_slurm_job(7, hendrix, 'hendrix')
    
    # schedule zones with more than 200K locations on LUMI
    lumi = sampled_df[sampled_df['landmass']> 200_000/0.703]
    queue_zones_for_slurm_job(3, lumi, 'lumi')
    

@hydra.main(config_path='../config', config_name='s2_download', version_base='1.2')
def main(cfg):
    # repartition(cfg)
    split_zones()

if __name__ == '__main__':
    main()