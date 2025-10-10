import xarray as xr
from typing import Any
import logging
import dask
import zarr
import time
import warnings
import pandas as pd
import geopandas as gpd
from pathlib import Path
from dataclasses import dataclass
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig
from omegaconf import ListConfig
import hydra
from dask.distributed import Lock

from download._dask_downloader import DaskDownloader

warnings.filterwarnings("ignore", 
                        category=UserWarning,
                        module="zarr.codecs.vlen_utf8",
                        message=".*vlen.*")
logger = logging.getLogger(__name__)

class ConvertToZarr(DaskDownloader):
    def __init__(self, h5_dir, save_dir, index_dir, n_parallel:int=10, flag_dir:str=None, **kwargs):
        super().__init__(n_parallel=n_parallel)
        self.h5_dir = Path(h5_dir).expanduser()
        self.save_dir = Path(save_dir).expanduser()
        self.save_dir.mkdir(exist_ok=True, parents=True)
        self.index_dir = Path(index_dir).expanduser()
        self.flag_dir = Path(flag_dir).expanduser()
        self.flag_dir.mkdir(exist_ok=True, parents=True)
        self.patch_size = 15
        shard_size = 1000
        compressor = zarr.codecs.BloscCodec(cname='blosclz', clevel=9)
        self.encoding = {
             'image': {
                "compressors": compressor,
                 "shards": (shard_size, 14, self.patch_size, self.patch_size),
                "chunks": (1, 14, self.patch_size, self.patch_size)
            },
            'rhs': {
                "compressors": compressor,
                "shards": (shard_size, 101),
                "chunks": (1, 101)
            },
            'gedi_attrs': {
                "compressors": compressor,
                "shards": (shard_size, 30),
                "chunks": (1, 30)
            },
            'slope': {
                "compressors": compressor,
                "shards": (shard_size, self.patch_size, self.patch_size),
                "chunks": (1, self.patch_size, self.patch_size)
            },
            'latlon': {
                "compressors": compressor,
                "shards": (shard_size, 2),
                "chunks": (1, 2)
            },
        }
        

    def convert_to_zarr(self):
        pass

    def convert_zone(self, zone):
        self.index_df = pd.read_parquet(self.index_dir / f'{zone}.parquet', columns=['path'])
        self.index_df = self.index_df.drop_duplicates()
        # self.convert_partition(self.index_df[self.index_df['path'].str.contains(f'{2019}')])
        years = [year for year in range(2019, 2023) if not (self.flag_dir / (f'{zone}_{year}_done')).exists()]
        if len(years) == 0:
            print(f'{zone} already done')
            return
        # tasks = [self.convert_partition(self.index_df[self.index_df['path'].str.contains(f'{year}')]) for year in years]
        # dask.compute(*tasks)

        tasks = [self.convert_partition(path) for i, path in self.index_df.iterrows()]
        self.schedule_tasks(delayed_tasks=tasks) 
        (self.flag_dir / f'{zone}_done').touch()
        

    @dask.delayed
    def convert_partition(self,  paths):
        _, zone, year, _ = paths.iloc[0].split('/') #['path']
        file = self.save_dir / f'year_{year}.zarr'
        # if (file/zone).exists():
        #     with xr.open_zarr(file, group=zone) as store:
        #         shot_number_snapshot = xr.open_zarr(file, group=zone).shot_number.data
        # else:
        #     shot_number_snapshot = []
        # for i, p in paths.iterrows():
        partition = paths['path'].split('/')[-1]
        h5_fp = self.h5_dir/f'{zone}.h5'
        ds = xr.open_dataset(h5_fp, group=f'{year}/{partition}', engine='h5netcdf')
        idx = ds.gedi_attrs.sel(attr='sensitivity') >= 0.95
        ds = ds.isel(shot_number=idx.data)
        # exists = ds.shot_number.isin(shot_number_snapshot)
        # if exists.all():
        #     print(f'{zone} {year} {partition} already exists')
        #     print(exists.sum()/len(exists))
        #     return
        # elif exists.any():
        #     print(f'{zone} {year} {partition} partially exists')
        #     idx_to_keep = ~exists
        #     ds = ds.isel(shot_number=idx_to_keep)

        file = self.save_dir / f'year_{year}.zarr'
        # ds = ds.chunk({'shot_number': 1, 'band': 14, 'y':15, 'x':15, 'rh': 101, 'attr':30, 'xy':2})
        with Lock('zarr'):
            if  (file/zone).exists():
                ds.to_zarr(file, mode='a-', group=zone, append_dim='shot_number')
            else:
                ds.to_zarr(file, mode='w', group=zone, encoding=self.encoding, zarr_version=3)
        



@dataclass
class MyConfig:
    index_dir: str = '~/data/gvs/geo_index_table_with_sensitivity'
    h5_dir: str = '~/data/gvs/GEDI_S2_h5'
    save_dir: str = '~/data/gvs/GEDI_S2_zarr'
    flag_dir: str = '~/data/gvs/flag_to_zarr'
    n_parallel: int=32
    zone: Any=None
    version: str = 'v1'

cs = ConfigStore.instance()
cs.store(name="my_config", node=MyConfig)

@hydra.main(config_name="my_config", version_base='1.2')
def main(cfg: DictConfig) -> None:
    from dask.distributed import Client, LocalCluster
    from dask import config
    cluster = LocalCluster()
    client = Client(cluster)
    t0 = time.time()
    config.set({'distributed.scheduler.allowed-failures': 10})
    converter = ConvertToZarr(**cfg)
    if isinstance(cfg.zone, (list, ListConfig)):
        zones = cfg.zone
    elif len(cfg.zone) > 3:
        zones = cfg.zone.split(',')
    else:
        zones = [cfg.zone]
    print(zones)
    for zone in zones:
        t0 = time.time()
        logger.info(f'processing zone: {zone}')
        converter.convert_zone(zone)
        logger.info(f'time taken for {zone}: {time.time() - t0}')


if __name__ == '__main__':
    main()



