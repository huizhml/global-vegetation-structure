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


class GEDIPostProcessor:
    def __init__(self, year, data_dir, save_dir, root_dir:str=None, partition_size:str=None, rewrite:bool=False):
        if isinstance(year, int):
            self.years = [year]
        else:
            self.years = year

        root_dir = Path(root_dir) if root_dir else Path.home()
        self.data_dir = root_dir / data_dir / str(year)
        self.save_dir = root_dir / save_dir / str(year)
        self.partition_size = partition_size
        self.rewrite = rewrite

    def __call__(self):
        pass

    def repartition(self, zone, year, root_dir, data_dir, partition_size):
        for year in self.years:
            config = []
            for zone in os.listdir(self.data_dir):
                if not os.path.isdir(self.data_dir / zone):
                    continue
                (self.save_dir / zone).mkdir(parents=True, exist_ok=True)
                if self.rewrite:
                    logger.info(f'remove partitions in {zone}...')
                    os.system(f'rm -rf {self.save_dir / zone}/*')
                if not (self.save_dir / zone / 'partition_0.parquet').exists():
                    logger.info(f'repartition {zone}...')
                    _repartition(self.data_dir, zone, self.partition_size, self.partition_name)

                logger.info(f'generate download config for {zone} {year}...')
                line = gen_config_for_zone(self.data_dir/zone, year)
                config.append(line+[zone])
            df = pd.DataFrame(config, columns=['ncores', 'npartitions', 'nparallel', 'MGRS_UTM'])
            mgrs_df = pd.read_csv(Path.home() / 'GEDI/mgrs_sampled.csv')
            pd.merge(df, mgrs_df, on='MGRS_UTM', how='left').to_csv(self.data_dir /'download_config_count.csv', index=False, sep=';')
            df = df.set_index('ncores').groupby('ncores').apply(aggregate_zones)
            df = df.reset_index()
            df.to_csv(self.data_dir /'download_config.csv', index=False, sep=';')

def _repartition(year_folder, zone, partition_size='128K', partition_name='partition_'):
    ddf = dgp.read_parquet(year_folder / zone / 'GEDI*.parquet')
    ddf = ddf.repartition(partition_size=partition_size)
    ddf.to_parquet(year_folder / f'{zone}', name_function=lambda x: f'{partition_name}{x}.parquet')

def aggregate_zones(group):
    if group.index[0] <= 16:
        zones = ','.join(group['MGRS_UTM'])
        res = group.iloc[0, :]
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
    if cfg.dask_cluster:
        import dask_gateway
        gateway = dask_gateway.Gateway()
        options = gateway.cluster_options()
        options['worker_cores'] = 8
        options['worker_memory'] = 64
        cluster = gateway.new_cluster(options)
        cluster.scale(13)
    else:

        from dask.distributed import LocalCluster, Client
        cluster = LocalCluster()
    
    client = Client(cluster)
    
    processor = GEDIPostProcessor(**cfg)
    
    if cfg.repartition:
        processor.repartition(cfg)
    else:
        processor.split_zones()
    client.close()

if __name__ == '__main__':
    main()