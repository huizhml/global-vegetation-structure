
# %%
from dotenv import load_dotenv
import hydra
import ipdb
from shapely.geometry import box, Point
from rasterio.errors import RasterioIOError
import pyproj
import dask.bag as db
import dask.dataframe as dd
import geopandas as gpd
import pandas as pd
from dask.utils import natural_sort_key
from dask.distributed import Lock, Semaphore, get_client, as_completed
import dask_geopandas as dgp
import dask
import os
import gc
import ctypes
import time
import logging
from pathlib import Path
from typing import Union, List
import pickle
from itertools import chain

import numpy as np
import xarray as xr

import pystac
import pystac_client
import planetary_computer
from urllib3 import Retry
from pystac_client.stac_api_io import StacApiIO

from download.dask_downloader import DaskDownloader
from ._const import dtypes, gedi_attr_dtype, rh_dtype, latlon_dtype, STAC_ITEM_KEYS, S2_ITEM_PROPS
from utils._stackstac import stack
from _utils import resign_items, row_to_stac_item, trim_memory, buffer_and_snap_bounds, get_total_bounds

# %%
load_dotenv('.planetarycomputer/settings.env')
os.environ["GDAL_HTTP_MAX_RETRY"] = "3"

# TODO: control from yaml
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

def backoff_hdlr(details):
    print("Backing off {wait:0.1f} seconds after {tries} tries "
          "calling function {target} with args {args} and kwargs "
          "{kwargs}".format(**details))

# TODO: add backoff
def get_patch(items,
              assets: Union[str, List[str]] = None,
              resolution: int = 10,
              fill_value: Union[int, float] = 0,
              band_coords: bool = False,
              properties: bool = False,
              dtype: str = 'uint16',
              xy_coords: bool = 'topleft',
              snap_bounds=False,
              **kwargs):
    default_args = dict(assets=assets,
                        resolution=resolution,
                        fill_value=fill_value,
                        band_coords=band_coords,
                        properties=properties,
                        dtype=dtype,
                        xy_coords=xy_coords,
                        snap_bounds=snap_bounds)
    try:
        patch = stack(items, **default_args, **kwargs)
    except:
        # TODO: rasterioerror still occurs sometimes, the url indeed didn't work, why?
        # TODO: this will fail the whole partition, how to catch such error and retry?
        # token might expire, sign again
        items = resign_items(items)
        patch = stack(items, **default_args, **kwargs)
    return patch


class S2Downloader(DaskDownloader):

    def __init__(self,
                 year: int = 2019,
                 n_parallel: int = 100,
                 root_dir: str = None,
                 data_dir='GEDI',
                 save_dir: str = 'data/GEDI',
                 patch_size: int = 15,
                 out_res: int = 10,
                 **kwargs
                 ) -> None:
        super().__init__(n_parallel=n_parallel, max_retries=3, **kwargs)
        root_dir = Path(root_dir) if root_dir else Path.home()
        self.gedi_dir = root_dir / f'{data_dir}/{year}'
        self.save_dir = root_dir / save_dir
        self.year = year
        self.n_parallel = n_parallel
        self.patch_size_in_meters = patch_size * out_res
        self.out_res = out_res
        self.buffer_size = patch_size // 2 * out_res  # in meters
        self.patch_size = (self.buffer_size * 2 + out_res) / out_res

    def find_best_s2(self, zone: str = None, from_file:str=None, to_file:str=None, rewrite: bool = False):
        """
        Find the best Sentinel-2 scene for each GEDI location in the given GEDI zone.
        Sentinel-2 candidates are pre-filtered based on scene level cloud cover and growing season (if the GEDI footprint is captured in the growing season).
        The best scene is selected based on the patch-level defective cover and relative days between the GEDI acquisition and the Sentinel-2 scene.

        Parameters
        ----------
        * zone (str): The MGRS zone to find the best S2 scene for.
        * from_file (str): The name of the file containing the GEDI data.

        Returns
        -------
        * str: A string indicating the status of the processing. Returns 'done' if the zone and year have already been processed.
        """
        if isinstance(zone, str):
            (self.save_dir/zone).mkdir(exist_ok=True, parents=True)
            flag = self.save_dir / f'{zone}_{self.year}_done'
            if rewrite:
                if flag.exists():
                    os.remove(flag)
                for f in (self.save_dir/zone).glob('*'):
                    os.remove(f)
            if flag.exists():
                logger.info(f'{zone} {self.year} has been processed.')
                return

            gedi_df = dgp.read_parquet(self.gedi_dir / f'{zone}/{from_file}.parquet')
        else:
            paths = []
            for z in zone:
                (self.save_dir/z).mkdir(exist_ok=True, parents=True)
                flag = self.save_dir / f'{z}_{self.year}_done'
                if rewrite:
                    if flag.exists():
                        os.remove(flag)
                    for f in (self.save_dir/z).glob('*'):
                        os.remove(f)
                if flag.exists():
                    logger.info(f'{z} {self.year} has been processed.')
                    continue
                paths.extend(list(self.gedi_dir.glob(f'{z}/{from_file}.parquet')))
            if len(paths) == 0:
                logger.info(f'All zones: {zone} in {self.year} have been processed.')
                return
            paths = [str(p) for p in paths]
            paths = sorted(paths, key=natural_sort_key)
            divisions = tuple(paths + [paths[-1]])
            gedi_df = dgp.read_parquet(paths, dropna=True)
            gedi_df.divisions = divisions

        logger.info(f'Processing {gedi_df.npartitions} partitions...')

        number = 13
        test = gedi_df.get_partition(number).compute()
        df = self.find_best_s2_for_partition(test, zone, rewrite=True, partition_info={'number': number, 'division': None})
        logger.info('test done')
        df = gedi_df.map_partitions(self.find_best_s2_for_partition, zone, to_file, rewrite, meta=(None, 'string'))
        self.schedule_tasks(df)

        # Create flags #TODO: use global Variable
        if isinstance(zone, str):
            flags = list(self.save_dir.glob(f'{zone}/{self.year}*'))
            if len(flags) == df.npartitions: # all partitions are done
                flag.touch()
        else:
            for p in paths:
                zone = Path(p).parent.name
                # check if if #partition parquet files == #zone/year_partition_done files
                flags = list(self.save_dir.glob(f'{zone}/{self.year}*'))
                if len(flags) == len(list((self.gedi_dir/zone).glob('partition*.parquet'))):
                    (self.save_dir/ f'{zone}_{self.year}_done').touch()
        return

    def find_best_s2_for_partition(self, partition, zone: str, to_file:str, rewrite: bool = False, partition_info: dict = None):
        """
        Find the best Sentinel-2 scene for each GEDI location in the given partition.
        A new column 'best_s2' is added to the partition(GeoDataFrame) and saved to the save_dir with name pattern specified by to_file.

        Parameters
        ----------
        * partition (pandas.DataFrame): The partition containing s2_candidates.
        * zone (str): The zone identifier.
        * to_file (str): The name pattern of the output file.
        * rewrite (bool): Whether to overwrite the existing file.
        """
        flag = 'best_s2' in partition.columns
        if flag and not rewrite:
            logger.info(f'{zone} {self.year}_partition_{partition_number} has been processed.')
            # TODO: update the list of done partitions using global Variable
            return

        if partition_info["division"] is not None:
            zone = partition_info["division"].split('/')[-2]
            partition_number = int(partition_info["division"].split('_')[1].split('.')[0])
        else:
            partition_number = partition_info['number']

        partition = partition.set_index('shot_number')
        op_part = partition[['date', 'geometry', 's2_candidates']]

        s2_candidates = set(chain.from_iterable(op_part['s2_candidates']))
        s2_meta_table = gpd.read_parquet(
            self.gedi_dir / f'{zone}/s2_items.parquet', filters=[('id', 'in', s2_candidates)])
        s2_meta_table = s2_meta_table.set_index('id')
        s2_meta_table['datetime'] = s2_meta_table['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S.%f')

        op_part = op_part.explode('s2_candidates')
        gedi_date = pd.to_datetime(op_part['date'])
        s2_date = op_part['s2_candidates'].str.extract('(\d{8})')
        s2_date = pd.to_datetime(s2_date[0], format='%Y%m%d')
        op_part['delta_day'] = (gedi_date - s2_date).dt.days.abs().astype('uint16')
        op_part = op_part.drop(columns=['date'])
        op_part = op_part.reset_index().set_index('s2_candidates')
        client = get_client()
        items = row_to_stac_item(s2_meta_table, S2_ITEM_PROPS)
        items_table = []
        for item in items:
            bounds = op_part.loc[[item.id]]
            items_table.append((item, bounds))

        items_table = db.from_sequence(items_table, npartitions=10)
        res = items_table.map(self.calculate_defective_cover).compute()
        
        client.run(trim_memory)
        df = pd.concat(res)
        df = df[df['defective_cover'] <= 0.8]
        df = df.sort_values(['defective_cover', 'delta_day']).groupby('shot_number').head(1)
        df = df.reset_index().set_index('shot_number').drop(columns='geometry')
        df = df.rename(columns={'s2_candidates': 'best_s2'})
        res = partition.merge(df, how='left', left_index=True, right_index=True)
        res = res.reset_index()
        to_file = to_file.replace('*', str(partition_number))
        res.to_parquet(self.save_dir / zone / f'{to_file}.parquet')
        #TODO: create a list of done partitions using global Vairable


    def calculate_defective_cover(self, entry):
        """
        Calculate patch-level defective cover for all GEDI locations on the same S2 scene.

        Parameters
        ----------
        * entry (tuple):
            * item (pystac.Item): The S2 STAC item.
            * locs (geopandas.DataFrame): GEDI locations.

        Returns
        -------
        * geopandas.DataFrame: locs with a new column defective cover added
        """
        
        item, locs = entry
        epsg = item.properties['proj:epsg']
        locs = locs.to_crs(epsg)
        bounds = buffer_and_snap_bounds(locs.geometry, self.buffer_size, self.out_res)
        total_bounds = get_total_bounds(bounds)

        image = get_patch(item, ['SCL'], resolution=self.out_res, bounds=total_bounds, epsg=epsg, dtype='uint8')
        defective_cover = []
        image = image.load() #NOTE: slicing on lazy object is inefficient
        for row in bounds.itertuples():
            xrange = range(row.minx, row.maxx, self.out_res) #slice(row.minx, row.maxx)
            yrange = range(row.maxy, row.miny, -self.out_res) #slice(row.maxy, row.miny)
            patch = image.sel(x=xrange, y=yrange).squeeze()
            dc = patch.isin(defective_SCL).sum(dim=['x', 'y']) / np.prod(patch.shape[-2:])
            defective_cover.append(dc.data)
        locs['defective_cover'] = defective_cover
        locs = locs.to_crs(4326)
        del image
        return locs.astype({'defective_cover': 'float32'})


# %%
@hydra.main(config_path="../config", config_name="s2_download", version_base="1.2")
def main(cfg):
    # if not (Path.home() / f'GEDI/{cfg.year}/{cfg.zone}').exists(): #TODO: some small zones might not have GEDI data in a certain year
    #     return
    from dask.distributed import Client, LocalCluster
    from dask import config
    # might fix the communication error caused by I/O. ref: https://github.com/dask/distributed/issues/3129#issuecomment-1684858307
    dask.config.set({"distributed.comm.retry.count": 10})
    dask.config.set({"distributed.comm.timeouts.connect": 30})
    dask.config.set({"distributed.scheduler.active-memory-manager.MALLOC_TRIM_THRESHOLD_": 0})
    cluster = LocalCluster()
    client = Client(cluster)
    print(client)

    s2downloader = S2Downloader(**cfg)
    t0 = time.time()

    logger.info(f'processing zone: {cfg.zone}')
    s2downloader.find_best_s2(cfg.zone, cfg.rewrite)
    logger.info(f'time taken for {cfg.zone} {cfg.year}: {time.time() - t0}')
    client.close()


# %%
if __name__ == "__main__":
    main()
