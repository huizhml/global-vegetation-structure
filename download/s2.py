
#%%
import os
import gc
import time
import json
import random
import datetime
import logging
from pathlib import Path
from typing import Union, List
import pickle

import numpy as np
import xarray as xr

import pystac
import pystac_client
import planetary_computer
from urllib3 import Retry
from pystac_client.stac_api_io import StacApiIO
# disable cuda before importing numba (CudaAPIError(3, 'Call to cuCtxGetCurrent results in CUDA_ERROR_NOT_INITIALIZED'))
# numba is used to calcluate slope
os.environ['NUMBA_DISABLE_CUDA'] = '1'

import dask
import dask_geopandas as dgp
from dask.distributed import Lock, Semaphore
from dask.utils import natural_sort_key
import pandas as pd
import geopandas as gpd
import dask.dataframe as dd
import pyproj
# from xrspatial import slope
from download._slope import slope
from rasterio.enums import Resampling
from rasterio.errors import RasterioIOError
import shapely

import ipdb
import hydra
from dotenv import load_dotenv

from utils._stackstac import stack
from const import dtypes, gedi_attr_dtype, rh_dtype, latlon_dtype
from download.dask_downloader import DaskDownloader
#%%
load_dotenv('.planetarycomputer/settings.env')
os.environ["GDAL_HTTP_MAX_RETRY"] = "3"
g0, g1, g2 = gc.get_count()
gc.set_threshold(g0*5, g1*5, g2 * 5)

retry = Retry(
    total=5, backoff_factor=1, status_forcelist=[502, 503, 504], allowed_methods=None # too many retries cause worker sleep too long when backoff_factor is 1
)
stac_api_io = StacApiIO(max_retries=retry)
stac_endpoint = 'https://planetarycomputer.microsoft.com/api/stac/v1'
api = pystac_client.Client.open(stac_endpoint, modifier=planetary_computer.sign_inplace, stac_io=stac_api_io)

gedi_attr_dtype.pop('shot_number')
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
    print ("Backing off {wait:0.1f} seconds after {tries} tries "
           "calling function {target} with args {args} and kwargs "
           "{kwargs}".format(**details))

def resign_items(items):
    """
    Resigns a list of items using the planetary_computer.sign() function.

    Args:
        items (list): A list of items to be resigned.

    Returns:
        list: A list of resigned items.
    """
    res = []
    for item in items:
        item = planetary_computer.sign(item)
        res.append(item)
    return res


def get_patch(items,
              assets: Union[str, List[str]] = None,
              resolution: int = 10,
              fill_value: Union[int, float] = 0,
              band_coords: bool = False,
              properties: bool = False,
              dtype: str = 'uint16',
              xy_coords: bool = False,
              **kwargs):
    default_args = dict(assets=assets,
                        resolution=resolution,
                        fill_value=fill_value,
                        band_coords=band_coords,
                        properties=properties,
                        dtype=dtype,
                        xy_coords=xy_coords)
    try:
        patch = stack(items, **default_args, **kwargs)
    except:
        #TODO: rasterioerror still occurs sometimes, the url indeed didn't work, why?
        # TODO: this will fail the whole partition, how to catch such error and retry?
        # token might expire, sign again
        items = resign_items(items)
        patch = stack(items, **default_args, **kwargs)
    return patch


def get_tile_by_id(tile_id):
    """
    Get the sentinel-2 tile by tile id.
    """
    url = f'{stac_endpoint}/collections/sentinel-2-l2a/items/{tile_id}'
    item = pystac.Item.from_file(url)
    return planetary_computer.sign_inplace(item) #TO CHCEK: the token generated seems to be only valid for 1 hour

def get_most_common_epsg(items):
    """
    Get the most common epsg code for a list of items.
    """
    epsgs = [item.properties['proj:epsg'] for item in items]
    return max(set(epsgs), key=epsgs.count)

def reproject_bounds(raster_spec, crs_to='EPSG:4326'):
    transformer = pyproj.Transformer.from_crs(f'EPSG:{raster_spec.epsg}', crs_to, always_xy=True)
    # Transform the bounds
    minx, miny = transformer.transform(raster_spec.bounds[0], raster_spec.bounds[1])
    maxx, maxy = transformer.transform(raster_spec.bounds[2], raster_spec.bounds[3])
    return [minx, miny, maxx, maxy]

class S2Downloader(DaskDownloader):

    def __init__(self,
                 year: int = 2019,
                 n_parallel: int = 100,
                 sem_max_release: int = 60,
                 root_dir: str = None,
                 data_dir = 'GEDI',
                 save_dir: str = 'data/GEDI',
                 patch_size: int = 15,
                 maxCloudCover: int = 50,
                 maxWaterPercentage: int = 100,
                 queryDaysRange: int = 90,
                 extendDays: int = 30,
                 esa_wc_year: int = 2021,
                 comp_level: int = 7,
                 out_res: int =10,
                 mgrs_file: str = 'mgrs_with_tracks_and_count.parquet',
                 **kwargs
                ) -> None:
        super().__init__(n_parallel=n_parallel, max_retries=3, **kwargs)
        # self.mgrs_df = gpd.read_parquet(Path.home() / f'GEDI/{mgrs_file}')
        root_dir = Path(root_dir) if root_dir else Path.home()
        self.gediFolder = root_dir / f'{data_dir}/{year}'
        self.save_dir = root_dir / save_dir
        self.year = year 
        self.esa_wc_year = esa_wc_year
        self.yearStart = pd.Timestamp(f'{self.year}-01-01', tz='UTC')
        self.queryDaysRange = pd.to_timedelta(queryDaysRange, unit='D')
        self.extendDays = pd.to_timedelta(extendDays, unit='D')
        self.maxCloudCover = maxCloudCover
        self.maxWaterPercentage = maxWaterPercentage
        self.n_parallel = n_parallel
        self.sem = Semaphore(sem_max_release, name='max_queries', register=True)
        self.bands = [
            'B01', 'B04', 'B03', 'B02', 'B05', 'B06', 'B07', 'B08', 'B8A',
            'B09', 'B11', 'B12', 'SCL'
        ]
        self.patch_size_in_meters = patch_size * out_res
        self.out_res = out_res
        self.buffer_size = patch_size // 2 * out_res # in meters
        self.dem_res = 30
        dem_buffer_size_in_pixel = self.patch_size_in_meters // self.dem_res // 2 + 2 # 2 pixels buffer for slope and upsampling
        self.dem_buffer_size = dem_buffer_size_in_pixel * self.dem_res
        self.patch_size = (self.buffer_size * 2 + out_res) / out_res
        self.comp = {
            'image':{
                "zlib": True,
                "complevel": comp_level,
                "fletcher32": True,
                "chunksizes": (1,14,self.patch_size,self.patch_size)
            },
            'rhs':{
                "zlib": True,
                "complevel": comp_level,
                "fletcher32": True,
                "chunksizes": (1,101)
            },
            'gedi_attrs':{
                "zlib": True,
                "complevel": comp_level,
                "fletcher32": True,
                "chunksizes": (1,30)
            },
            'slope':{
                "zlib": True,
                "complevel": comp_level,
                "fletcher32": True,
                "chunksizes": (1,self.patch_size,self.patch_size)
            }
        }

      
    def get_zone_bbox(self, zone:str=None):
        """
        Retrieves the bounding box coordinates for a given zone.

        Parameters:
        - zone (str): The MGRS zone identifier.

        Returns:
        - bounds (list): The bounding box coordinates [min_lon, min_lat, max_lon, max_lat].
        """
        # check code https://code.earthengine.google.com/4dcc2d69fa3cab567be1a8b3f43d64e7
        bounds = self.mgrs_df[self.mgrs_df['MGRS_UTM'] == zone]['geometry'].bounds.values[0]
        if bounds[2] > 180 and bounds[2] < 185:
            bounds[2] = 180
        elif bounds[2] > 185:
            bounds[2] = -bounds[0] + 6
            bounds[0] = -180
        return bounds
    
    def download_zone(self, zone:str=None, rewrite:bool=False):
        '''
        Filter S2 tiles for each GEDI zone.

        Args:
            zone (str): The GEDI zone to filter S2 tiles for.

        Returns:
            str: A string indicating the status of the processing. Returns 'done' if the zone and year have already been processed.
        '''
        if isinstance(zone, str):
            (self.save_dir/zone).mkdir(exist_ok=True, parents=True)
            flag = self.save_dir/ f'{zone}_{self.year}_done'
            if rewrite:
                if flag.exists(): os.remove(flag)
                for f in (self.save_dir/zone).glob('*'):
                    os.remove(f)
            if flag.exists():
                logger.info(f'{zone} {self.year} has been processed.')
                return
            
            gediDf = dgp.read_parquet(self.gediFolder / f'{zone}/partition*.parquet', dropna=True) #, split_row_groups=True
        else:
            paths = []
            for z in zone:
                (self.save_dir/z).mkdir(exist_ok=True, parents=True)
                flag = self.save_dir/ f'{z}_{self.year}_done'
                if rewrite:
                    if flag.exists(): os.remove(flag)
                    for f in (self.save_dir/z).glob('*'):
                        os.remove(f)
                if flag.exists():
                    logger.info(f'{z} {self.year} has been processed.')
                    continue
                paths.extend(list(self.gediFolder.glob(f'{z}/partition*.parquet')))
            if len(paths) == 0:
                logger.info(f'All zones: {zone} in {self.year} have been processed.')
                return
            paths = [str(p) for p in paths]
            paths = sorted(paths, key=natural_sort_key)
            divisions = tuple(paths + [paths[-1]])
            gediDf = dgp.read_parquet(paths, dropna=True)
            gediDf.divisions = divisions
        
        logger.info(f'Processing {gediDf.npartitions} partitions...')
        with_growing_season = (gediDf['leaf_on_doy'] < 366) & (gediDf['leaf_off_flag'] == 0)
        # leaf off date is in the next year
        reverse = gediDf['leaf_on_doy'] > gediDf['leaf_off_doy']
        gediDf['leaf_off_doy'] = gediDf['leaf_off_doy'].mask(reverse, gediDf['leaf_off_doy'] + 365)

        leaf_on_doy = dd.to_timedelta(gediDf['leaf_on_doy'], unit='D')
        leaf_off_doy = dd.to_timedelta(gediDf['leaf_off_doy'], unit='D')

        gediDf['start'] = dd.to_datetime(gediDf['date']) - self.queryDaysRange
        gediDf['end'] = dd.to_datetime(gediDf['date']) + self.queryDaysRange
        gediDf['start'] = gediDf['start'].mask(with_growing_season, self.yearStart + leaf_on_doy)
        gediDf['end'] = gediDf['end'].mask(with_growing_season, self.yearStart + leaf_off_doy)


        # number = 0
        # df = self.get_patch_for_partition(gediDf.get_partition(number).compute()[:5], zone, rewrite=True, partition_info={'number': number, 'division': None})
        # logger.info('test done')
        df = gediDf.map_partitions(self.get_patch_for_partition, zone, rewrite, meta=(None, 'string'))
        self.schedule_tasks(df)
        
        # Create flags
        if isinstance(zone, str):
            flags = list(self.save_dir.glob(f'{zone}/{self.year}*'))
            if len(flags) == df.npartitions: # all partitions are done
                flag.touch()
        else:
            for p in paths:
                zone = Path(p).parent.name
                # check if if #partition parquet files == #zone/year_partition_done files
                flags = list(self.save_dir.glob(f'{zone}/{self.year}*'))
                if len(flags) == len(list((self.gediFolder/zone).glob('partition*.parquet'))):
                    (self.save_dir/ f'{zone}_{self.year}_done').touch()
        return

    def get_patch_for_partition(self, partition, zone:str, rewrite:bool=False, partition_info:dict=None):
        """
        Query, filter, and stack S2 and ESA world cover patches for each GEDI partition.
        GEDI data is partitioned to cache a number of locations for the sake of memory efficiency.

        Args:
            partition (pandas.DataFrame): The partition containing the data.
            zone (str): The zone identifier.
            esa_wc_items (pystac.ItemCollection): The collection of ESA WC items.
            partition_info (dict, optional): Information about the partition.
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

        esa_wc_items = api.search(
            collections=['esa-worldcover'],
            bbox=partition.total_bounds,
            datetime=f'{self.esa_wc_year}-01-01/{self.esa_wc_year}-12-31').item_collection()
        glo30_itmes = api.search(collections=['cop-dem-glo-30'],
                                 bbox=partition.total_bounds).item_collection()
        partition = partition.reset_index(drop=True) # original index is not unique, shot_number is slow when partition.loc[xrrs.index]
        xrrs = []
        keep = []
        cols = ['shot_number', 'date', 'start', 'end', 'geometry']
        for row in partition[cols].itertuples():
            best = self.get_best_s2_for_point(row, esa_wc_items, glo30_itmes)
            if best is not None:
                xrrs.append(best)
                keep.append(row.Index)
        if len(xrrs) == 0:
            (self.save_dir / zone / f'{self.year}_partition_{partition_number}_0_found').touch()
            return

        partition = partition.loc[keep].set_index('shot_number')
        rh_da = partition[rh_dtype.keys()].to_xarray().to_dataarray('rh', 'rhs')
        gedi_attr_da = partition[gedi_attr_dtype.keys()].to_xarray().to_dataarray('attr', 'gedi_attrs')
        latlon_da = partition[latlon_dtype.keys()].to_xarray().to_dataarray('xy', 'latlon')
        xrrs = dask.compute(*xrrs)
        slope_da = xr.concat([s['slope'] for s in xrrs], dim='shot_number', compat='override', coords='minimal', join='override')
        xrrs = xr.concat([s['xrr'] for s in xrrs], dim='shot_number', compat='override', coords='minimal', join='override')

        slope_da.attrs['res'] = self.dem_res #self.out_res
        slope_da = slope(slope_da) # (band, x, y)
        w, h = slope_da.shape[-2:]
        slope_da = slope_da.assign_coords(x=range(1,3*w, 3), y=range(1,3*h, 3)) # set xy coords to the center of the pixel (to match s2 xrr coords)
        slope_da = slope_da.interp(x=range(3*w), y=range(3*h))
        slope_da = slope_da.isel(x=slice(6,-6), y=slice(6,-6)) #remove nan
        slope_da = slope_da.assign_coords(x=xrrs.x, y=xrrs.y) # set xy coords back to 0-14
        slope_da = slope_da.drop_vars(['x', 'y'])

        xrrs.name = 'image'
        xrrs = xr.merge([xrrs, slope_da, rh_da.transpose(), gedi_attr_da.transpose(), latlon_da.transpose()])
        xrrs = xrrs.assign_attrs(partition_bounds=partition.total_bounds)
        with Lock('netcdf_lock'):
            xrrs.to_netcdf(self.save_dir / f'{zone}.h5', group=f'{self.year}/{partition_number}', format='NETCDF4', engine='h5netcdf', encoding=self.comp, mode='a')
        flag.touch()

    def get_best_s2_for_point(self, point, esa_wc_items:pystac.ItemCollection,glo30_itmes:pystac.ItemCollection,):
        '''
        Qeury and Filter S2 tiles for each GEDI point
        - Query by date, cloud cover, water percentage, and Filter by leaf on/off dates
        - Filter by defective cover (patch level)
        '''
        geom = point.geometry 
        items = self.query_s2_for_p(point.start, point.end, geom)
        
        if items is None:
            return None

        # get patch and calculate defective cover
        epsg = get_most_common_epsg(items)
        geom = gpd.GeoSeries(geom, crs='EPSG:4326').to_crs(epsg)[0]
        bounds = geom.buffer(self.buffer_size).bounds
        bounds_slope = geom.buffer(self.dem_buffer_size).bounds # self.buffer_size+self.out_res

        best = self.calculate_defective_cover(items, bounds, point.date, epsg)
        if best is None:
            return
        best_item = [item for item in items if item.id == best.id.iloc[0]][0]
        s2xrr = get_patch(best_item, assets=self.bands, bounds=bounds, epsg=epsg, resolution=self.out_res)
        s2xrr = harmonize_to_old(s2xrr)
        s2xrr = s2xrr.squeeze()

        wc_xrr = get_patch(esa_wc_items, assets=['map'], bounds=bounds, epsg=epsg, resolution=self.out_res)
        # glo_xrr = get_patch(glo30_itmes, assets=['data'], bounds=bounds_slope, epsg=epsg, fill_value=np.nan, dtype='float32', resolution=self.out_res, resampling=Resampling.bilinear)
        glo_xrr = get_patch(glo30_itmes, assets=['data'], bounds=bounds_slope, epsg=epsg, fill_value=np.nan, dtype='float32', resolution=self.dem_res)
        
        if glo_xrr.shape[0] == 0 or wc_xrr.shape[0] == 0:
            return None
        glo_xrr = glo_xrr.max(dim='time', skipna=True)[0]

        wc_xrr = wc_xrr.max(dim='time', skipna=True)
        wc_xrr = wc_xrr.assign_coords(band=['esa_wc'])

        xrr = xr.concat([s2xrr, wc_xrr], dim='band', compat='override', coords='minimal', combine_attrs='drop')
        xrr = xrr.expand_dims(dim={'shot_number': [point.shot_number]}, axis=0) # return a view, not a copy

        new_coords = {k: ("shot_number", best[k]) for k in ['delta_day','defective_cover']}
        for coord in ['time', 'id']: # time, id, epsg
            new_coords.update({coord: ("shot_number", [xrr[coord].data])})
        new_coords.update({'epsg': ("shot_number", [xrr.epsg.data.astype('uint16')])})
        new_coords.update({'spec': ("shot_number", [pickle.dumps(s2xrr.spec)])})
        xrr = xrr.assign_coords(new_coords)
        return {'xrr': xrr, 'slope': glo_xrr.drop_vars(['epsg'])}

    
    def calculate_defective_cover(self, items, bounds, date, epsg):
        '''
        Calculate defective cover (patch level) for tiles with the same epsg
        
        Args:
            group (pandas.DataFrame): DataFrame of sentinel-2 tiles with the same epsg
            geom (geopandas.GeometryArray): geometry of the point
            
        Returns:
            pandas.Series: Series containing the best item
        '''
        patch = get_patch(items, ['SCL'], resolution=self.out_res, bounds=bounds, epsg=epsg, dtype='uint8')
        
        if patch.shape[0] == 0 or patch.shape[-2:] != (self.patch_size, self.patch_size): # why there're cases that the output shape is (14,15)? fill_value doesn't work?
            return None

        patch = patch.compute() # simplify compute graph, not sure if this is necessary, 
        # patch = patch.isin(defective_SCL).sum(dim=['x', 'y']) / np.prod(patch.shape[-2:]) 
        # even patch is computed, patch.isin().sum() will still be lazy
        scl = patch.data
        defective_cover = np.any([(scl == k) for k in defective_SCL], 0).sum((-2,-1)) / np.prod(scl.shape[-2:])
        if np.isnan(defective_cover).all() or defective_cover.min() > 0.9:
            return
        patch_df = pd.DataFrame({
            'id': patch.id.values,
            'defective_cover': defective_cover.squeeze(),
            'delta_day': [np.abs(t - pd.Timestamp(date)).days for t in patch.time.values],
        })
        
        patch_df = patch_df.sort_values(['defective_cover', 'delta_day'])
        best = patch_df.iloc[:1]
        best = best.astype({
            'defective_cover': 'float32',
            'delta_day': 'uint16'
        })
        return best
    
    def query_s2_for_p(self, start, end, geom):
        """
        Query Sentinel-2 data for a given time range and geometry. 
        If no items are found, extend the time range by 60 days and try again, util the time range covers 365 days.

        Args:
            start (datetime): Start date of the time range.
            end (datetime): End date of the time range.
            geom (shapely.geometry): Geometry object representing the area of interest.

        Returns:
            list: List of Sentinel-2 items matching the query criteria, or None if no items found.
        """
        with self.sem: #? what will happen if sem times out, will we get items?
            search = api.search(collections=['sentinel-2-l2a'],
                                query={
                                    "eo:cloud_cover": {
                                        "lt": self.maxCloudCover
                                    },
                                    's2:water_percentage': {
                                        'lt': self.maxWaterPercentage
                                    }
                                },
                                intersects=geom,
                                datetime=f'{str(start)[:10]}/{str(end)[:10]}')
            items = search.item_collection()
        if len(items) > 0:
            return items
        if len(items) == 0 and (end - start).days < 365:
            logger.info(f'No S2 tile found between {start} - {end}, extend the range by {self.extendDays.days*2} days')
            items = self.query_s2_for_p(start - self.extendDays,
                                           end + self.extendDays, geom)
            return items
        else:
            return None

#%%
@hydra.main(config_path="../config", config_name="s2_download", version_base="1.2")
def main(cfg):
    from dask.distributed import Client, LocalCluster
    from dask import config 
    config.set({'distributed.scheduler.locks.lease-timeout': 60}) 
    # might fix the communication error caused by I/O. ref: https://github.com/dask/distributed/issues/3129#issuecomment-1684858307
    dask.config.set({"distributed.comm.retry.count": 10})
    dask.config.set({"distributed.comm.timeouts.connect": 30}) 
    cluster = LocalCluster()
    client = Client(cluster)#timeout
    print(client)

    s2downloader = S2Downloader(**cfg)
    t0 = time.time()

    logger.info(f'processing zone: {cfg.zone}')
    res = s2downloader.download_zone(cfg.zone, cfg.rewrite)
    logger.info(f'time taken for {cfg.zone}: {time.time() - t0}')
    client.close()


#%%
if __name__ == "__main__":
    # from omegaconf import DictConfig, OmegaConf
    # cfg = OmegaConf.load('config/s2_download.yaml')
    main()

# %%

# %%
