
# %%
import warnings
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
from dask import delayed
import dask_geopandas as dgp
import dask
import os
import gc
import time
import datetime
import logging
from typing import Any
from pathlib import Path
import pickle
from omegaconf import OmegaConf, ListConfig
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

warnings.filterwarnings("ignore", 
                        category=UserWarning,
                        module="zarr.codecs.vlen_utf8",
                        message=".*vlen.*")

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


class S2Downloader(DaskDownloader):

    def __init__(self,
                 rewrite: bool = False,
                 year: int = 2019,
                 n_parallel: int = 100,
                 gedi_dir='GEDI',
                 save_dir: str = 'data/GEDI',
                 flag_dir: str = 'scratch/Download_flags',
                 S2_meta_dir: str = 'scratch/S2_meta',
                 wc_dem_meta_dir: str = 'scratch/WC_DEM_meta',
                 patch_size: int = 15,
                 esa_wc_year: int = 2021,
                 comp_level: int = 7,
                 out_res: int = 10,
                 **kwargs
                 ) -> None:
        super().__init__(n_parallel=n_parallel, max_retries=3, **kwargs)
        self.gedi_dir = Path(gedi_dir).expanduser()
        self.save_dir = Path(save_dir).expanduser()
        self.flag_dir = Path(flag_dir).expanduser()
        self.S2_meta_dir = Path(S2_meta_dir).expanduser()
        self.wc_dem_meta_dir = Path(wc_dem_meta_dir).expanduser() 
        self.year = year
        self.rewrite = rewrite
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
        parquet_file = self.wc_dem_meta_dir / f'{collection_id}_items.parquet'
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
            self.s2_table_file = self.S2_meta_dir / f'{self.year}/{zone}.parquet'
        else:
            zone_flag = self.flag_dir / f'small_zones_{self.year}_done'
            self.zone_flag_dir  = self.flag_dir / 'small_zones'
            files = []
            for z in zone:
                files.extend(list(self.gedi_dir.glob(f'{self.year}/{z}/partition*.parquet')))
                # (self.save_dir / z).mkdir(exist_ok=True, parents=True)
            self.s2_table_file = self.S2_meta_dir / f'{self.year}/small_zones.parquet'
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
        self.unfinished_files = self.files #TODO: remove
        gedi_df = dgp.read_parquet(self.unfinished_files, gather_spatial_partitions=False)

        s2_meta_table = gpd.read_parquet(self.s2_table_file)

        # update s2 meta table. TODO: to be put in find_best_s2_api
        s2_ids = gedi_df.map_partitions(lambda x: x['best_s2']).compute()
        s2_ids = s2_ids.reset_index().dropna().drop_duplicates(subset='best_s2')
        new_ids = s2_ids[~s2_ids['best_s2'].isin(s2_meta_table['id'])]
        if not new_ids.empty:
            df = []
            for i in new_ids['best_s2']:
                item = get_tile_by_id(i)
                assets = {k: v.to_dict() for k, v in item.assets.items()}
                df.append([item.id, item.bbox, assets, item.properties['datetime'], item.properties['proj:epsg'], shape(item.geometry)])
            df = gpd.GeoDataFrame(df, columns=['id', 'bbox', 'assets', 'datetime', 'proj:epsg', 'geometry'])
            df = df.astype({'proj:epsg': 'uint16'})
            df.crs='epsg:4326'
            df['datetime'] = pd.to_datetime(df['datetime']).dt.tz_localize(None)
            new_meta_table = pd.concat([s2_meta_table, df])
            new_meta_table = gpd.GeoDataFrame(new_meta_table)
            new_meta_table.to_parquet(self.s2_table_file)

        wc_df = self.get_aux_df(
            'esa-worldcover', filters=[('start_datetime', '>=', self.esa_wc_time)],
            time_col='start_datetime')
        dem_df = self.get_aux_df('cop-dem-glo-30', time_col='datetime')
        self.wc_df = wc_df[wc_df.geometry.intersects(box(*s2_meta_table.total_bounds))]
        self.dem_df = dem_df[dem_df.geometry.intersects(box(*s2_meta_table.total_bounds))]
        self.wc_df['datetime'] = self.wc_df['start_datetime'].dt.strftime('%Y-%m-%d %H:%M:%S.%f')
        self.dem_df['datetime'] = self.dem_df['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S.%f')

        logger.info(f'Processing {gedi_df.npartitions} partitions...')

        # number = 2
        # # # test = gedi_df.get_partition(number).compute()
        # #  2022/35L/partition_167
        # # self.unfinished_files = self.files
        # test = gpd.read_parquet(self.unfinished_files[number])
        # df = self.download_patches_for_partition(test, partition_info={'number': number})
        # logger.info('test done')
        df = gedi_df.map_partitions(self.download_patches_for_partition, meta=(None, 'string'))
        nfailed = self.schedule_tasks(df)
        
        # if nfailed > 0:
        #     # one more try
        #     logger.info('Some partitions failed. Restarting the client...')
        #     client = dask.distributed.get_client()
        #     client.restart()
        #     nfailed = self.schedule_tasks(df)

        if nfailed <= 0:
            zone_flag.touch()

    def process_batch(self, batch, wc_items):
        results = []
        for item in batch:
            result = self.extract_patches_from_tile(item, wc_items)
            if result is not None:
                results.append(result)
            # Explicitly free memory
            del result
        return results

    def download_patches_for_partition(self, partition, partition_info: dict = None):
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
        if flag.exists() and not self.rewrite:
            logger.info(f'{zone} {self.year}_partition_{partition_number_infile} has been processed.')
            return

        partition = partition.dropna(subset='best_s2')
        partition = partition.drop_duplicates(subset='shot_number', keep='last')
        if partition.empty:
            return

        op_part = partition[['shot_number', 'geometry', 'best_s2', 'defective_cover']].set_index('best_s2')
        s2_ids = op_part.index.unique()
        s2_meta_table = gpd.read_parquet(
            self.s2_table_file, filters=[('id', 'in', s2_ids)])

        s2_meta_table = s2_meta_table.set_index('id')
        s2_meta_table['datetime'] = s2_meta_table['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S.%f')

        client = get_client()
        bbox = box(*s2_meta_table.total_bounds)
        wc_df = self.wc_df[self.wc_df.geometry.intersects(bbox)]
        if wc_df.empty:
            print(s2_meta_table.total_bounds)
            logger.info('No world cover found.')
            return

        items = row_to_stac_item(s2_meta_table, S2_ITEM_PROPS)
        wc_items = row_to_stac_item(wc_df, ['datetime'])
        
        items_table = []
        for item in items:
            bounds = op_part.loc[[item.id]]
            items_table.append((item, bounds))

        # test_item = items_table[0]
        # test_id = test_item[0].id
        # print(op_part.loc[test_id])
        # self.extract_patches_from_tile(test_item, wc_items)
        batch_size = min(10, len(items_table))
        batches = [items_table[i:i + batch_size] for i in range(0, len(items_table), batch_size)]

        # Process each batch as a single task
        tasks = [delayed(self.process_batch)(batch, wc_items) for batch in batches]

        # Compute the results in parallel
        ds = dask.compute(*tasks)
        # items_table_db = db.from_sequence(items_table, npartitions=4)
        # ds = items_table_db.map(self.extract_patches_from_tile, wc_items).compute()
        # client.run(gc.collect)
        client.run(trim_memory)
        ds = [ss for s in ds for ss in s if ss is not None]
        if len(ds) == 0:
            return
        slope_da = xr.concat([s['dem_da'] for s in ds], dim='shot_number',
                             compat='override', coords='minimal', join='override')
        ds = xr.concat([s['da'] for s in ds], dim='shot_number')
        ds.name = 'image'
        # check if delta_day and defective cover match
        partition = partition[partition['shot_number'].isin(ds.shot_number.data)]
        partition = partition.set_index('shot_number')
        dc = partition['defective_cover'].loc[ds.shot_number.data]
        dd = partition['delta_day'].loc[ds.shot_number.data]
        dc_real = ds.sel(band='SCL').isin(defective_SCL).sum(dim=['x', 'y']) / np.prod(ds.shape[-2:])
        dc_real = dc_real.astype('float32')
        gedi_date = pd.to_datetime(partition['date'])
        s2_date = partition['best_s2'].str.extract(r'(\d{8})')
        s2_date = pd.to_datetime(s2_date[0], format='%Y%m%d')
        dd_real = (gedi_date - s2_date).dt.days.abs().astype('uint16')
        assert (dc == dc_real).all() and (dd == dd_real.loc[dd.index]).all(), 'delta_day or defective cover mismatch'

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
        ds = xr.merge([ds, slope_da, rh_da.transpose(), gedi_attr_da.transpose(), latlon_da.transpose()], join="inner")
        ds = ds.assign_attrs(partition_bounds=partition_bounds)
        ds['band'] = ds['band'].astype('<U6')
        bands = ds.band.values
        bands[-1] = 'esa_wc'
        ds = ds.assign_coords(
                    band=bands,
                    delta_day=('shot_number', dd),
                    defective_cover=('shot_number', dc),
            )
        # file  = self.save_dir / f'{zone}.zarr'
        # try:
        #     if file.exists():
        #         ds.to_zarr(file, group=f'{self.year}', mode='a-', append_dim='shot_number')
        #     else:
        #         ds.to_zarr(file, group=f'{self.year}', mode='w')
        # except Exception as e:
        #     logger.error(f'Failed to write to Zarr file: {e}')

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


    def extract_patches_from_tile(self, entry, wc_items):
        client = get_client()
        item, locs = entry
        locs = locs.set_index('shot_number')
        bbox = box(*locs.geometry.total_bounds)
        epsg = item.properties['proj:epsg']
        locs = locs.copy().to_crs(epsg)
        bounds = buffer_and_snap_bounds(locs.geometry, self.buffer_size, self.out_res)
        dem_bounds = buffer_and_snap_bounds(locs.geometry, self.dem_buffer_size, self.dem_res)
        total_bounds = get_total_bounds(bounds)
        total_bounds_dem = get_total_bounds(dem_bounds)
        s2_image = get_patch(item, self.bands, resolution=self.out_res, bounds=total_bounds, epsg=epsg, dtype='uint16', fill_value=np.uint16(0))
        wc_image = get_patch(wc_items, ['map'], resolution=self.out_res,bounds=total_bounds, epsg=epsg, dtype='uint16', fill_value=np.uint16(0))
        dem_df = self.dem_df[self.dem_df.geometry.intersects(bbox)]
        if dem_df.empty:
            dem_items = api.search(collections=['nasadem'], intersects=bbox).item_collection()
            dem_items.asset_name = 'elevation'
        else:
            dem_items = row_to_stac_item(dem_df, ['datetime'])
            dem_items = pystac.item_collection.ItemCollection(dem_items)
            dem_items.asset_name = 'data'
        dem_image = get_patch(dem_items.items, [dem_items.asset_name], resolution=self.dem_res, bounds=total_bounds_dem, epsg=epsg, dtype='float32', fill_value=np.float32(np.nan))
        if wc_image.shape[0] == 0 or dem_image.shape[0]==0: # there're areas without world cover or dem https://code.earthengine.google.com/72d2134898deffe65f81150ac7caeb73
            return
        
        dem_image = dem_image.max(dim='time', skipna=True).squeeze()
        wc_image = wc_image.max(dim='time', skipna=True).squeeze()
        s2_image = harmonize_to_old(s2_image)
        image = xr.concat([s2_image.squeeze(), wc_image.squeeze()], dim='band',
                          coords='minimal', compat='override')  # only keep s2's time & id


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
            patches, dim=pd.Index(bounds.index.values, name='shot_number'),
            coords='all', combine_attrs='drop')
        da = da.assign_coords({
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
        dem_da = xr.concat(patches, dim=pd.Index(dem_bounds.index.values, name='shot_number'), combine_attrs='drop')
        da, dem_da = dask.compute(da, dem_da)
        del dem_image, image, s2_image, wc_image
        
        return {'da': da, 'dem_da': dem_da.drop_vars(['epsg'])}

# %%
@dataclass
class MyConfig:
    zone: Any = '09U'
    year: int = 2019
    n_parallel: int = 8
    rewrite: bool = False
    gedi_dir: str = '~/data/GVS/GEDI_extra_with_s2_candidates_and_best' # GEDI data dir
    save_dir: str = '~/data/GVS/GEDI_S2_h5_extra'
    flag_dir: str = '~/data/GVS/Correct_order_flags/'
    S2_meta_dir: str = '~/data/GEDI/S2_geoparquet_items'
    wc_dem_meta_dir: str = '~/data/GEDI'
    patch_size: int = 15
    debug: bool = False

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
        client.cluster.adapt(minimum=1, maximum=8)
        print(client)

    s2downloader = S2Downloader(**cfg)
    t0 = time.time()
    if isinstance(cfg.zone, (list, ListConfig)):
        zones = cfg.zone
    elif len(cfg.zone) > 3:
        zones = cfg.zone.split(',')
    else:
        zones = [cfg.zone]
    print(zones)
    for zone in zones:
        for year in range(2019, 2023):
            s2downloader.year = year
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
