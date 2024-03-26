
# %%
from dotenv import load_dotenv
import hydra
import ipdb
from stackstac.raster_spec import RasterSpec
from shapely.geometry import box, Point
from rasterio.errors import RasterioIOError
from rasterio.enums import Resampling

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
import json
import random
import datetime
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

from .dask_downloader import DaskDownloader
from ._const import dtypes, gedi_attr_dtype, rh_dtype, latlon_dtype, STAC_ITEM_KEYS, S2_ITEM_PROPS
from ._utils import trim_memory, row_to_stac_item, resign_items, buffer_and_snap_bounds, get_total_bounds
from ._slope import slope
from utils._stackstac import stack

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

def harmonize_to_old(data):
    """
    Harmonize new Sentinel-2 data to the old baseline.

    Parameters
    ----------
    data: xarray.DataArray
        A DataArray with four dimensions: time, band, y, x

    Returns
    -------
    harmonized: xarray.DataArray
        A DataArray with all values harmonized to the old
        processing baseline.
    """
    cutoff = datetime.datetime(2022, 1, 25)
    offset = 1000
    bands = [
        "B01",
        "B02",
        "B03",
        "B04",
        "B05",
        "B06",
        "B07",
        "B08",
        "B8A",
        "B09",
        "B10",
        "B11",
        "B12",
    ]

    old = data.sel(time=slice(cutoff))

    to_process = list(set(bands) & set(data.band.data.tolist()))
    new = data.sel(time=slice(cutoff, None)).drop_sel(band=to_process)

    new_harmonized = data.sel(time=slice(cutoff, None), band=to_process).clip(offset)
    new_harmonized -= offset

    new = xr.concat([new, new_harmonized], "band").sel(band=data.band.data.tolist())
    return xr.concat([old, new], dim="time")


def backoff_hdlr(details):
    print("Backing off {wait:0.1f} seconds after {tries} tries "
          "calling function {target} with args {args} and kwargs "
          "{kwargs}".format(**details))


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
                 esa_wc_year: int = 2021,
                 comp_level: int = 7,
                 out_res: int = 10,
                 **kwargs
                 ) -> None:
        super().__init__(n_parallel=n_parallel, max_retries=3, **kwargs)
        root_dir = Path(root_dir) if root_dir else Path.home()
        self.gedi_dir = root_dir / f'{data_dir}/{year}'
        self.save_dir = root_dir / save_dir
        self.year = year
        self.esa_wc_time = pd.Timestamp(f'{esa_wc_year}-01-01', tz='UTC')
        self.n_parallel = n_parallel
        self.bands = [
            'B01', 'B04', 'B03', 'B02', 'B05', 'B06', 'B07', 'B08', 'B8A',
            'B09', 'B11', 'B12', 'SCL'
        ]
        self.patch_size_in_meters = patch_size * out_res
        self.out_res = out_res
        self.buffer_size = patch_size // 2 * out_res  # in meters
        self.dem_res = 30
        dem_buffer_size_in_pixel = self.patch_size_in_meters // self.dem_res // 2 + 2  # 2 pixels buffer for slope and upsampling
        self.dem_buffer_size = dem_buffer_size_in_pixel * self.dem_res
        self.patch_size = (self.buffer_size * 2 + out_res) / out_res
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

    def get_aux_df(self, collection_id, filters=None, time_col=None):
        """
        Retrieves auxiliary geodataframe for a specified collection (ESA world cover or DEM).

        Parameter
        ---------
        * collection_id (str): The ID of the collection to retrieve data from.
        * filters (dict, optional): Additional filters to apply to the data. Defaults to None.
        * time_col (str, optional): The name of the time column to include in the retrieved data. Defaults to None.

        Returns
        -------
        * pandas.DataFrame: The retrieved auxiliary data as a pandas DataFrame.
        """
        parquet_file = self.gedi_dir.parent / f'{collection_id}_items.parquet'
        if parquet_file.exists():
            logger.info(f'Loading STAC parquet files for {collection_id} from: {parquet_file}')
            df = gpd.read_parquet(parquet_file, columns=STAC_ITEM_KEYS+[time_col])
        else:
            logger.info(f'Downloading STAC parquet files for {collection_id}')
            asset = api.get_collection(collection_id).assets["geoparquet-items"]
            df = dgp.read_parquet(
                asset.href, storage_options=asset.extra_fields["table:storage_options"],
                gather_spatial_partitions=False, filters=filters, columns=STAC_ITEM_KEYS+[time_col])
            df = df.compute()
            df.to_parquet(parquet_file)
        df = df.set_index('id')
        return df

    def download_zone(self, zone: str = None, from_file: str=None, rewrite: bool = False):
        """
        Download patches for all GEDI locations in given MGRS zone.

        Parameters
        ----------
        * zone (str): The MGRS zone to download patches from.
        * from_file (str): The name of the file containing the GEDI data.
        * rewrite (bool): Whether to rewrite the existing files.
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

        s2_meta_table = gpd.read_parquet(self.gedi_dir / f'{zone}/s2_items.parquet')
        wc_df = self.get_aux_df(
            'esa-worldcover', filters=[('start_datetime', '>=', self.esa_wc_time)],
            time_col='start_datetime')
        dem_df = self.get_aux_df('cop-dem-glo-30', time_col='datetime')
        self.wc_df = wc_df[wc_df.geometry.intersects(box(*s2_meta_table.total_bounds))]
        self.dem_df = dem_df[dem_df.geometry.intersects(box(*s2_meta_table.total_bounds))]
        self.wc_df['datetime'] = self.wc_df['start_datetime'].dt.strftime('%Y-%m-%d %H:%M:%S.%f')
        self.dem_df['datetime'] = self.dem_df['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S.%f')

        logger.info(f'Processing {gedi_df.npartitions} partitions...')

        # number = 17
        # test = gedi_df.get_partition(number).compute()
        # df = self.download_patches_for_partition(test, zone, rewrite=True, partition_info={'number': number, 'division': None})
        # logger.info('test done')
        df = gedi_df.map_partitions(self.download_patches_for_partition, zone, rewrite, meta=(None, 'string'))
        self.schedule_tasks(df)

        # # Create flags
        # if isinstance(zone, str):
        #     flags = list(self.save_dir.glob(f'{zone}/{self.year}*'))
        #     if len(flags) == df.npartitions: # all partitions are done
        #         flag.touch()
        # else:
        #     for p in paths:
        #         zone = Path(p).parent.name
        #         # check if if #partition parquet files == #zone/year_partition_done files
        #         flags = list(self.save_dir.glob(f'{zone}/{self.year}*'))
        #         if len(flags) == len(list((self.gedi_dir/zone).glob('partition*.parquet'))):
        #             (self.save_dir/ f'{zone}_{self.year}_done').touch()
        # return

    def download_patches_for_partition(self, partition, zone: str, rewrite: bool = False, partition_info: dict = None):
        """
        Query, filter, and stack S2 and ESA world cover patches for each GEDI partition.
        GEDI data is partitioned to cache a number of locations for the sake of memory efficiency.

        Parameters
        ----------
        * partition (pandas.DataFrame): The partition containing the data.
        * zone (str): The zone identifier.
        * esa_wc_items (pystac.ItemCollection): The collection of ESA WC items.
        * partition_info (dict, optional): Information about the partition.
        """
        if partition_info["division"] is not None:
            zone = partition_info["division"].split('/')[-2]
            partition_number = int(partition_info["division"].split('_')[1].split('.')[0])
        else:
            partition_number = partition_info['number']

        flag = self.save_dir / zone / f'{self.year}_partition_{partition_number}_done'
        if flag.exists() and not rewrite:
            logger.info(f'{zone} {self.year}_partition_{partition_number} has been processed.')
            return

        partition = partition.set_index('shot_number')
        op_part = partition[['geometry', 'best_s2']]

        s2_ids = op_part['best_s2'].unique()
        s2_meta_table = gpd.read_parquet(
            self.gedi_dir / f'{zone}/s2_items.parquet', filters=[('id', 'in', s2_ids)])
        s2_meta_table = s2_meta_table.set_index('id')
        s2_meta_table['datetime'] = s2_meta_table['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S.%f')

        client = get_client()

        wc_df = self.wc_df[self.wc_df.geometry.intersects(box(*s2_meta_table.total_bounds))]
        dem_df = self.dem_df[self.dem_df.geometry.intersects(box(*s2_meta_table.total_bounds))]
        items = row_to_stac_item(s2_meta_table, S2_ITEM_PROPS)
        wc_items = row_to_stac_item(wc_df, ['datetime'])
        dem_items = row_to_stac_item(dem_df, ['datetime'])

        items_table = []
        for item in items:
            bounds = op_part.loc[[item.id]]
            items_table.append((item, bounds))

        items_table = db.from_sequence(items_table, npartitions=4)
        ds = items_table.map(self.extract_patches_from_tile, wc_items, dem_items).compute()
        # client.run(gc.collect)
        client.run(trim_memory)
        slope_da = xr.concat([s['dem_da'] for s in ds], dim='shot_number',
                             compat='override', coords='minimal', join='override')
        ds = xr.concat([s['da'] for s in ds], dim='shot_number')
        ds.name = 'image'
        slope_da.attrs['res'] = self.dem_res
        slope_da = slope(slope_da)
        w, h = slope_da.shape[-2:]
        # set xy coords to the center of the pixel (to match s2 xrr coords)
        slope_da = slope_da.assign_coords(x=range(1, 3*w, 3), y=range(1, 3*h, 3))
        slope_da = slope_da.interp(x=range(3*w), y=range(3*h))
        slope_da = slope_da.isel(x=slice(6, -6), y=slice(6, -6))  # remove nan
        slope_da = slope_da.assign_coords(x=ds.x, y=ds.y)  # set xy coords back to 0-14
        slope_da = slope_da.drop_vars(['x', 'y'])

        rh_da = partition[rh_dtype.keys()].to_xarray().to_dataarray('rh', 'rhs')
        gedi_attr_da = partition[gedi_attr_dtype.keys()].to_xarray().to_dataarray('attr', 'gedi_attrs')
        latlon_da = partition[latlon_dtype.keys()].to_xarray().to_dataarray('xy', 'latlon')
        drop_cols = list(rh_dtype.keys()) + list(gedi_attr_dtype.keys()) + list(latlon_dtype.keys())
        partition = partition.drop(columns=drop_cols)
        partition_bounds = partition.total_bounds
        ds = xr.merge([ds, slope_da, rh_da.transpose(), gedi_attr_da.transpose(), latlon_da.transpose()])
        ds = ds.assign_attrs(partition_bounds=partition_bounds)
        ds['band'] = ds['band'].astype('<U6')
        bands = ds.band.values
        bands[-1] = 'esa_wc'
        ds = ds.assign_coords(band=bands)
        with Lock('netcdf_lock'):
            ds.to_netcdf(self.save_dir / f'{zone}.h5', group=f'{self.year}/{partition_number}',
                         format='NETCDF4', engine='h5netcdf', encoding=self.comp, mode='a')
        flag.touch()

    def calculate_defective_cover(self, entry):
        '''
        Calculate defective cover (patch level) for GEDI locations on the same S2 tile.

        Args:
            group (pandas.DataFrame): DataFrame of GEDI locations on the same S2 tile

        Returns:
            pandas.DataFrame: DataFrame with defective cover for each GEDI location
        '''
        client = get_client()
        item, locs = entry
        epsg = item.properties['proj:epsg']
        locs = locs.to_crs(epsg)
        bounds = self.buffer_and_snap_bounds(locs.geometry, self.buffer_size, self.out_res)
        total_bounds = self.total_bounds(bounds)

        image = get_patch(item, ['SCL'], resolution=self.out_res, bounds=total_bounds, epsg=epsg, dtype='uint8')
        defective_cover = []
        image = image.load()
        for row in bounds.itertuples():
            xrange = range(row.minx, row.maxx, self.out_res) #slice(row.minx, row.maxx)
            yrange = range(row.maxy, row.miny, -self.out_res) #slice(row.maxy, row.miny)
            patch = image.sel(x=xrange, y=yrange).squeeze()
            dc = patch.isin(defective_SCL).sum(dim=['x', 'y']) / np.prod(patch.shape[-2:])
            defective_cover.append(dc.data)
        # defective_cover = dask.compute(*defective_cover)
        locs['defective_cover'] = defective_cover
        locs = locs.to_crs(4326)
        del image
        client.run(trim_memory)
        return locs.astype({'defective_cover': 'float32'})

    def extract_patches_from_tile(self, entry, wc_items, dem_items):
        client = get_client()
        item, locs = entry
        epsg = item.properties['proj:epsg']
        locs = locs.copy().to_crs(epsg)
        bounds = buffer_and_snap_bounds(locs.geometry, self.buffer_size, self.out_res)
        dem_bounds = buffer_and_snap_bounds(locs.geometry, self.dem_buffer_size, self.dem_res)
        total_bounds = get_total_bounds(bounds)
        total_bounds_dem = get_total_bounds(dem_bounds)
        
        s2_image = get_patch(item, self.bands, resolution=self.out_res, bounds=total_bounds, epsg=epsg, dtype='uint16')
        wc_image = get_patch(wc_items, ['map'], resolution=self.out_res,bounds=total_bounds, epsg=epsg, dtype='uint16')
        wc_image = wc_image.max(dim='time', skipna=True).squeeze()
        s2_image = harmonize_to_old(s2_image)
        image = xr.concat([s2_image.squeeze(), wc_image.squeeze()], dim='band',
                          coords='minimal', compat='override')  # only keep s2's time & id

        dem_image = get_patch(dem_items, ['data'], resolution=self.dem_res, bounds=total_bounds_dem, epsg=epsg, dtype='float32', fill_value=np.nan)
        dem_image = dem_image.max(dim='time', skipna=True).squeeze()

        patches = []
        specs = []
        image = image.load()
        for row in bounds.itertuples():
            xrange = range(row.minx, row.maxx, self.out_res)
            yrange = range(row.maxy, row.miny, -self.out_res)
            patch = image.sel(x=xrange, y=yrange).squeeze()
            patch = patch.drop_vars(['x', 'y'])
            patches.append(patch)
            spec = RasterSpec(
                epsg=epsg,
                bounds=(row.minx, row.miny, row.maxx, row.maxy),
                resolutions_xy=(self.out_res, self.out_res),
            )
            specs.append(pickle.dumps(spec))
        da = xr.concat(
            patches, dim=pd.Index(locs.shot_number.values, name='shot_number'),
            coords='all', combine_attrs='drop')
        da = da.assign_coords({
            'delta_day': ('shot_number', locs['delta_day']),
            'defective_cover': ('shot_number', locs['defective_cover']),
            'spec': ('shot_number', specs)
        })
        da['epsg'] = da.epsg.astype('uint16')
        
        client.run(trim_memory)

        patches = []
        dem_image = dem_image.load()
        for row in dem_bounds.itertuples():
            xrange = range(row.minx, row.maxx, self.dem_res)
            yrange = range(row.maxy, row.miny, -self.dem_res)
            patch = dem_image.sel(x=xrange, y=yrange)
            patch = patch.drop_vars(['x', 'y'])
            patches.append(patch)
        dem_da = xr.concat(patches, dim=pd.Index(locs.shot_number.values, name='shot_number'), combine_attrs='drop')
        da, dem_da = dask.compute(da, dem_da)
        del dem_image
        del image
        
        client.run(trim_memory)
        return {'da': da, 'dem_da': dem_da.drop_vars(['epsg'])}

# %%


@hydra.main(config_path="../config", config_name="s2_download", version_base="1.2")
def main(cfg):
    # if not (Path.home() / f'GEDI/{cfg.year}/{cfg.zone}').exists(): # some small zones might not have GEDI data in a certain year
    #     return
    from dask.distributed import Client, LocalCluster
    from dask import config
    config.set({'distributed.scheduler.locks.lease-timeout': 60})
    # might fix the communication error caused by I/O. ref: https://github.com/dask/distributed/issues/3129#issuecomment-1684858307
    dask.config.set({"distributed.comm.retry.count": 10})
    dask.config.set({"distributed.comm.timeouts.connect": 30})
    dask.config.set({"distributed.scheduler.active-memory-manager.MALLOC_TRIM_THRESHOLD_": 0})
    cluster = LocalCluster(n_workers=4, threads_per_worker=4)
    client = Client(cluster)  # timeout
    print(client)

    s2downloader = S2Downloader(**cfg)
    t0 = time.time()

    logger.info(f'processing zone: {cfg.zone}')
    res = s2downloader.download_zone(cfg.zone, cfg.rewrite)
    logger.info(f'time taken for {cfg.zone} {cfg.year}: {time.time() - t0}')
    # client.close()


# %%
if __name__ == "__main__":
    # from omegaconf import DictConfig, OmegaConf
    # cfg = OmegaConf.load('config/s2_download.yaml')
    main()

# %%

# %%
