
# %%
from typing import Any
from dotenv import load_dotenv
import hydra
import ipdb
import pystac.item_collection
from stackstac.raster_spec import RasterSpec
from shapely.geometry import box, shape

import pyproj
import dask.bag as db
import dask.dataframe as dd
import geopandas as gpd
import pandas as pd
from dask.utils import natural_sort_key
from dask.distributed import Lock, get_client
import dask_geopandas as dgp
import dask
import os
import gc
import time
import datetime
import logging
from pathlib import Path
import pickle
from omegaconf import ListConfig, OmegaConf
from hydra.core.config_store import ConfigStore
from dataclasses import dataclass, field

import numpy as np
import xarray as xr
import h5py

import pystac
import pystac_client
import planetary_computer
from urllib3 import Retry
from pystac_client.stac_api_io import StacApiIO

from ._dask_downloader import DaskDownloader
from ._const import gedi_attr_dtype, rh_dtype, latlon_dtype, STAC_ITEM_KEYS, S2_ITEM_PROPS
from ._utils import trim_memory, row_to_stac_item, buffer_and_snap_bounds, get_total_bounds, get_patch, get_tile_by_id
from ._slope import slope


# %%
# disable cuda before importing numba (CudaAPIError(3, 'Call to cuCtxGetCurrent results in CUDA_ERROR_NOT_INITIALIZED'))
# numba is used to calcluate slope
os.environ['NUMBA_DISABLE_CUDA'] = '1'

load_dotenv('.planetarycomputer/settings.env')
os.environ["GDAL_HTTP_MAX_RETRY"] = "3"
# g0, g1, g2 = gc.get_count()
# gc.set_threshold(g0*5, g1*5, g2 * 5)

retry = Retry(
    # too many retries cause worker sleep too long when backoff_factor is 1
    total=5, backoff_factor=1, status_forcelist=[502, 503, 504], allowed_methods=None
)
stac_api_io = StacApiIO(max_retries=retry)
stac_endpoint = 'https://planetarycomputer.microsoft.com/api/stac/v1'
api = pystac_client.Client.open(stac_endpoint, modifier=planetary_computer.sign_inplace, stac_io=stac_api_io)

defective_SCL = [0, 1, 8, 9, 10, 11]  # keep cloud shadows, model should learn to be invariant to cloud shadows

logger = logging.getLogger(__name__)
cfg = {
    "sentinel-2-l2a": {
        "assets": {
            "*": {"data_type": "uint16", "nodata": 0},
            "WVP": {"data_type": "uint16", "nodata": 0},
            "B04": {"data_type": "uint16", "nodata": 0},
            "B03": {"data_type": "uint16", "nodata": 0},
            "B02": {"data_type": "uint16", "nodata": 0},
            "B08": {"data_type": "uint16", "nodata": 0},
            "SCL": {"data_type": "uint8", "nodata": 0},
            "visual": {"data_type": "uint16", "nodata": 0},
        },
    },
    "*": {"warnings": "ignore"},
}



class S2Downloader(DaskDownloader):

    def __init__(self,
                 rewrite: bool = False,
                 root_dir: str = None,
                 year: int = 2019,
                 n_parallel: int = 100,
                 gedi_dir='GEDI',
                 h5_dir: str = 'data/GEDI',
                 save_dir: str = 'data/GEDI',
                 flag_dir: str = 'scratch/Download_flags',
                 S2_meta_dir: str = 'scratch/S2_meta',
                 patch_size: int = 15,
                 esa_wc_year: int = 2021,
                 comp_level: int = 7,
                 out_res: int = 10,
                 **kwargs
                 ) -> None:
        super().__init__(n_parallel=n_parallel, max_retries=1, **kwargs)
        root_dir = Path(root_dir).expanduser() if root_dir else Path.home()
        self.h5_dir = Path(h5_dir).expanduser()
        self.gedi_dir = Path(gedi_dir).expanduser() 
        self.save_dir = Path(save_dir).expanduser()
        self.flag_dir = Path(flag_dir).expanduser()
        self.year = year
        self.rewrite = rewrite
        self.bands = [
            'B01', 'B04', 'B03', 'B02', 'B05', 'B06', 'B07', 'B08', 'B8A',
            'B09', 'B11', 'B12', 'SCL'
        ]
        self.patch_size = 15
        self.comp = {
            'image': {
                "zlib": True,
                "complevel": comp_level,
                "fletcher32": True,
                "chunksizes": (1, 14, self.patch_size, self.patch_size)
            },
            'rhs': {
                "zlib": True,
                "complevel": comp_level,
                "fletcher32": True,
                "chunksizes": (1, 101)
            },
            'gedi_attrs': {
                "zlib": True,
                "complevel": comp_level,
                "fletcher32": True,
                "chunksizes": (1, 30)
            },
            'slope': {
                "zlib": True,
                "complevel": comp_level,
                "fletcher32": True,
                "chunksizes": (1, self.patch_size, self.patch_size)
            }
        }


    def download_zone(self, zone: str = None):
        """
        Download patches for all GEDI locations in given MGRS zone.

        Parameters
        ----------
        * zone (str): The MGRS zone to download patches from.
        * from_file (str): The name of the file containing the GEDI data.
        * rewrite (bool): Whether to rewrite the existing files.
        """
        if isinstance(zone, str):
            files = list(self.gedi_dir.glob(f'{self.year}/{zone}/partition*.parquet'))
            # (self.save_dir / zone).mkdir(exist_ok=True, parents=True)
            zone_flag = self.flag_dir / f'{zone}_{self.year}_done'
            self.zone_flag_dir  = self.flag_dir / zone
        elif isinstance(zone, (list, ListConfig)):
            zone_flag = self.flag_dir / f'small_zones_{self.year}_done'
            self.zone_flag_dir  = self.flag_dir / 'small_zones'
            files = []
            for z in zone:
                files.extend(list(self.gedi_dir.glob(f'{self.year}/{z}/partition*.parquet')))
                # (self.save_dir / z).mkdir(exist_ok=True, parents=True)
        else:
            raise ValueError('zone must be a string or a list of strings.')
        self.zone_flag_dir.mkdir(exist_ok=True, parents=True)
        if self.rewrite:
            if zone_flag.exists():
                os.remove(zone_flag)
            for f in self.zone_flag_dir.glob(f'{self.year}*'):
                os.remove(f)
        if zone_flag.exists():
            logger.info(f'{zone} {self.year} has been processed.')
            return

        files = [str(p) for p in files]
        self.files = sorted(files, key=natural_sort_key)
        self.unfinished_files = []
        for i, fp in enumerate(self.files):
            flag = self.zone_flag_dir / f'{self.year}_partition_{i}_done'
            if not flag.exists():
                self.unfinished_files.append(fp)
        

        print('unfinished files', self.unfinished_files)
        gedi_df = dgp.read_parquet(self.unfinished_files, gather_spatial_partitions=False)
        logger.info(f'Processing {gedi_df.npartitions} partitions...')

        # number = 0
        # # test = gedi_df.get_partition(number).compute()
        # #  2022/35L/partition_167
        # # self.unfinished_files = self.files
        # test = gpd.read_parquet(self.unfinished_files[number])
        # df = self.correct_partition(test, partition_info={'number': number})
        # logger.info('test done')
        df = gedi_df.map_partitions(self.correct_partition, meta=(None, 'string'))
        nfailed = self.schedule_tasks(df)
        
        # if nfailed > 0:
        #     # one more try
        #     logger.info('Some partitions failed. Restarting the client...')
        #     client = dask.distributed.get_client()
        #     client.restart()
        #     nfailed = self.schedule_tasks(df)

        if nfailed <= 0:
            zone_flag.touch()


    def correct_partition(self, partition, partition_info: dict = None):
        """
        Query, filter, and stack S2 and ESA world cover patches for each GEDI partition.
        GEDI data is partitioned to cache a number of locations for the sake of memory efficiency.

        Parameters
        ----------
        * partition (pandas.DataFrame): The partition containing the data.
        * esa_wc_items (pystac.ItemCollection): The collection of ESA WC items.
        * partition_info (dict, optional): Information about the partition.
        """

        partition_number = partition_info["number"] # the index of the partition in the dataframe
        zone = self.unfinished_files[partition_number].split('/')[-2]
        partition_number_infile = self.files.index(self.unfinished_files[partition_number])

        flag = self.zone_flag_dir / f'{self.year}_partition_{partition_number_infile}_done'
        # if flag.exists() and not self.rewrite:
        #     logger.info(f'{zone} {self.year}_partition_{partition_number_infile} has been processed.')
        #     return

        partition = partition.dropna(subset='best_s2')
        partition = partition.drop_duplicates(subset=['shot_number'], keep=False)
        if partition.empty or not (self.h5_dir/f'{zone}.h5').exists():
            return
        ds = xr.open_dataset(self.h5_dir/f'{zone}.h5', group=f'{self.year}/{partition_number_infile}', engine='h5netcdf')
        ds = ds.drop_duplicates(['shot_number', 'xy'], keep=False)
        slope_da = ds.slope
        img = ds.image

        partition = partition[partition['shot_number'].isin(ds.shot_number.data)]
        partition = partition.set_index('shot_number')
        rh_da = partition[rh_dtype.keys()].to_xarray().to_dataarray('rh', 'rhs')
        gedi_attr_da = partition[gedi_attr_dtype.keys()].to_xarray().to_dataarray('attr', 'gedi_attrs')
        latlon_da = partition[latlon_dtype.keys()].to_xarray().to_dataarray('xy', 'latlon')
        drop_cols = list(rh_dtype.keys()) + list(gedi_attr_dtype.keys()) + list(latlon_dtype.keys())
        partition = partition.drop(columns=drop_cols)
        partition_bounds = partition.total_bounds
        ds = xr.merge([img, slope_da, rh_da.transpose(), gedi_attr_da.transpose(), latlon_da.transpose()], join="inner")
        ds = ds.assign_attrs(partition_bounds=partition_bounds)
        ds['band'] = ds['band'].astype('<U6')
        ds = ds.assign_coords(
                    delta_day=('shot_number', partition['delta_day'].loc[ds.shot_number.data]),
                    defective_cover=('shot_number', partition['defective_cover'].loc[ds.shot_number.data]),
            )
        # import ipdb; ipdb.set_trace()
        # if not (self.save_dir / f'{zone}/{self.year}').exists():
        #     ds.to_zarr(self.save_dir, mode='a-', group=f'/{zone}/{self.year}', consolidated=True)
        # else:
        #     ds.to_zarr(self.save_dir, mode='a-', append_dim='shot_number', group=f'/{zone}/{self.year}', consolidated=True)
        with Lock('netcdf_lock'):
            try:
                ds.to_netcdf(self.save_dir / f'{zone}.h5', group=f'{self.year}/{partition_number_infile}',
                         format='NETCDF4', engine='h5netcdf', encoding=self.comp, mode='a')
            except:
                with h5py.File(self.save_dir / f'{zone}.h5', 'a') as file:
                    del file[f'{self.year}/{partition_number_infile}']
                ds.to_netcdf(self.save_dir / f'{zone}.h5', group=f'{self.year}/{partition_number_infile}',
                         format='NETCDF4', engine='h5netcdf', encoding=self.comp, mode='a')
        flag.touch()


# %%
@dataclass
class MyConfig:
    zone: Any = '09U'
    year: int = 2019
    n_parallel: int = 32
    root_dir: str = '~/data/GEDI'
    rewrite: bool = False
    gedi_dir: str = '~/data/GEDI/GEDI_with_s2_candidates_and_best' # GEDI data dir
    h5_dir: str = '~/data/GEDI/GEDI_S2_h5s_original'
    save_dir: str = '~/data/gvs/GEDI_S2_h5'
    flag_dir: str = '~/data/gvs/Correct_order_flags'
    S2_meta_dir: str = 'S2_geoparquet_items'
    patch_size: int = 15
    debug: bool = False
    merge_zones: str = ''

cs = ConfigStore.instance()
cs.store(name="config", node=MyConfig)


@hydra.main(config_name='config', version_base='1.2')
def main(cfg):
    # if not (Path.home() / f'GEDI/{cfg.year}/{cfg.zone}').exists(): # some small zones might not have GEDI data in a certain year
    #     return
    logger.info(OmegaConf.to_yaml(cfg))
    if not cfg.debug:
        from dask.distributed import Client, LocalCluster
        from dask import config
        config.set({'distributed.scheduler.locks.lease-timeout': 60})
        # might fix the communication error caused by I/O. ref: https://github.com/dask/distributed/issues/3129#issuecomment-1684858307
        dask.config.set({"distributed.comm.retry.count": 10})
        dask.config.set({"distributed.comm.timeouts.connect": 30})
        dask.config.set({"distributed.scheduler.active-memory-manager.MALLOC_TRIM_THRESHOLD_": 0})
        cluster = LocalCluster()# n_workers=4, threads_per_worker=4
        client = Client(cluster)  # timeout
        print(client)

    s2downloader = S2Downloader(**cfg)
    t0 = time.time()
    small_zones = ['23J','25L','25S','26K','26P','26Q','26S','27P','27Q','27R','28H','28M','30K','30L','31M','32K','32L','33H','38M','60U','40K','40M','40P','41K','42M','43M','43N','46N','46P','48L','49J','49K','49L','50P','52P','52R','53N','54N','54P','54R','55N','55P','55Q','55T','55U','56K','56L','56N','56P','56R','56T','56U','57J','57K','57L','57M','57N','57U','58J','58K','58L','58M','58P','58Q','59H','59K','59L','59N','60K','60M','01K','01L','01R','02K','02L','03K','03L','04K','04L','04N','05K','05L','05M','06J','06K','06L','07K','07L','07M','08K','40L','58N','59P','60L','01J','01N','02M','04M','07J','29G','58G','60G','01G','02Q','02R','28N','22T','22U','28S','29U','01U','02U','04Q','05Q','52H','20S','54G','37G','58F','60F','09J','10J','11Q','12J','13J','15M','15N','16M','17L','18K','21F','20F','21P','14P','12Q','18R','20Q','21T']
    if cfg.zone == 'all':
        h5_dir = Path(cfg.h5_dir).expanduser()
        zones = h5_dir.glob('*.h5')
        zones = [z.stem for z in zones]
        zones = set(zones) - set(small_zones)
        for year in range(2019, 2023):
            logger.info(f'processing small zones for year: {year}')
            s2downloader.download_zone(small_zones)
            for zone in zones:
                logger.info(f'processing zone: {zone}, year: {year}')
                s2downloader.year = year
                s2downloader.download_zone(zone)
                logger.info(f'time taken for {zone} {year}: {time.time() - t0}')

    else:
        for year in range(2019, 2023):
            s2downloader.year = year
            if isinstance(cfg.zone, (list, ListConfig)):
                zones = cfg.zone
            elif len(cfg.zone) > 3:
                zones = cfg.zone.split(',')
            else:
                zones = [cfg.zone]
            print(zones)

            if cfg.merge_zones == 'True':
                logger.info(f'processing small zones for year: {year}')
                s2downloader.download_zone(small_zones)
            else:
                for zone in zones:
                    logger.info(f'processing zone: {zone}, year: {year}')
                    s2downloader.download_zone(zone)
                    logger.info(f'time taken for {zone} {year}: {time.time() - t0}')
    # client.close()


# %%
if __name__ == "__main__":
    # from omegaconf import DictConfig, OmegaConf
    # cfg = OmegaConf.load('config/s2_download.yaml')
    main()

# %%

# %%
