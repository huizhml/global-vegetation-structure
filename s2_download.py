#%%
import os
import time
import json
from pathlib import Path
from typing import Dict, Union, List
from collections import defaultdict
import pickle

import numpy as np
import xarray as xr

import ee
import pystac
import pystac_client
import planetary_computer
import retry

import dask
import dask.dataframe as dd
from dask.distributed import Lock, as_completed
from distributed import get_client
import pandas as pd
import geopandas as gpd
import pyproj
from xrspatial import slope

import ipdb
import hydra
from dotenv import load_dotenv

from utils._stackstac import stack
from const import dtypes, gedi_attr_dtype, rh_dtype, s2_item_props

load_dotenv('.planetarycomputer/settings.env')
os.environ["GDAL_HTTP_MAX_RETRY"] = "3"

#%%
stac_endpoint = 'https://planetarycomputer.microsoft.com/api/stac/v1'
api = pystac_client.Client.open(stac_endpoint, modifier=planetary_computer.sign_inplace)

dtypes.pop('.geo')
gedi_attr_dtype.pop('.geo')
gedi_attr_dtype.pop('shot_number')
defective_SCL = [0, 1, 8, 9, 10, 11]  # keep cloud shadows, model should learn to be invariant to cloud shadows

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

class S2Downloader:

    def __init__(self,
                 year: int = 2019,
                 n_parallel: int = 100,
                 save_dir: str = 'data/GEDI',
                 patch_size: int = 15,
                 maxCloudCover: int = 50,
                 maxWaterPercentage: int = 100,
                 queryDaysRange: int = 90,
                 extendDays: int = 30,
                 esa_wc_year: int = 2021,
                 comp_level: int = 7,
                 **kwargs
                ) -> None:
        self.gediFolder = Path.home() / f'GEDI{year}'
        self.save_dir = Path.home() / save_dir
        self.year = year 
        self.esa_wc_year = esa_wc_year
        self.yearStart = pd.Timestamp(f'{self.year}-01-01', tz='UTC')
        self.queryDaysRange = pd.to_timedelta(queryDaysRange, unit='D')
        self.extendDays = pd.to_timedelta(extendDays, unit='D')
        self.maxCloudCover = maxCloudCover
        self.maxWaterPercentage = maxWaterPercentage
        self.n_parallel = n_parallel
        self.bands = [
            'B01', 'B04', 'B03', 'B02', 'B05', 'B06', 'B07', 'B08', 'B8A',
            'B09', 'B11', 'B12', 'SCL'
        ]
        self.patch_size = patch_size
        self.buffer_size = patch_size // 2 * 10
        self.comp = {
            'input':{
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
                "chunksizes": (1,32)
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
        
        key_file = 'keys/private-key.json'
        key = json.load(open(key_file))
        credentials = ee.ServiceAccountCredentials(key['client_email'], key_file)
        ee.Initialize(credentials)
        mgrs = ee.FeatureCollection('projects/gisproject-1/assets/gedi_count_mgrs_aggregated_landmass')
        bounds = mgrs.filter(ee.Filter.eq('MGRS_UTM', zone)).first().geometry().bounds()
        bounds = bounds.getInfo()['coordinates'][0]
        bounds = [*bounds[0], *bounds[2]]
        if bounds[2] > 180:
            bounds[0] = -bounds[0]
            bounds[2] = bounds[0] + 6
        return bounds
    
    def download_zone(self, zone:str=None, rewrite:bool=False):
        '''
        Filter S2 tiles for each GEDI zone.

        Args:
            zone (str): The GEDI zone to filter S2 tiles for.

        Returns:
            str: A string indicating the status of the processing. Returns 'done' if the zone and year have already been processed.
        '''
        zoneFolder = self.gediFolder / zone
        (self.save_dir/zone).mkdir(exist_ok=True, parents=True)
        flag = self.save_dir/ f'{zone}_{self.year}_done'
        if rewrite:
            if flag.exists(): os.remove(flag)
            for f in (self.save_dir/zone).glob('*'):
                os.remove(f)
        if flag.exists():
            print(f'{zone} {self.year} has been processed.')
            return
        
        bounds = self.get_zone_bbox(zone)
        esa_wc_items = api.search(
            collections=['esa-worldcover'],
            bbox=bounds,
            datetime=f'{self.esa_wc_year}-01-01/{self.esa_wc_year}-12-31').item_collection()
        glo30_itmes = api.search(collections=['cop-dem-glo-30'],
                                 bbox=bounds).item_collection()

        gediDf = dd.read_parquet(zoneFolder / 'partition_*.parquet', dropna=True, usecols=list(dtypes.keys()), dtype=dtypes)
        print(f'Processing {gediDf.npartitions} partitions...')
        # number = 4
        # df = self.get_patch_for_partition(gediDf.get_partition(number).compute(), zone, esa_wc_items, glo30_itmes, rewrite=True, partition_info={'number': number})
        # print('test done')
        df = gediDf.map_partitions(self.get_patch_for_partition, zone, esa_wc_items, glo30_itmes, rewrite, meta=(None, 'string'))
        self.n_parallel = min(self.n_parallel, gediDf.npartitions)
        client = get_client()
        futures = []
        for i in range(self.n_parallel):
            future = client.compute(df.get_partition(i))
            futures.append(future)
        
        futures_monitor = as_completed(futures, with_results=False)
        n_left = gediDf.npartitions - self.n_parallel
        max_retries = 3
        retry_counter: Dict[str, int] = defaultdict(lambda: 0)   
        res = []
        while futures_monitor.count() > 0:
            f = next(futures_monitor)
            if f.status == 'error':
                if retry_counter.get(future, 0) < max_retries:
                    try:
                        f.retry()
                        futures_monitor.add(f)
                        retry_counter[future.key] += 1
                    except Exception as e:
                        print(e)
                        f.retry() #TODO: key eror in self.futures[key] when first retry, why? related to distributed.scheduler - ERROR - Couldn't gather keys: {('sum-aggregate-ce2045d27a178c14f0a6884069ecef48', 0): 'processing'}?
                        futures_monitor.add(f)
                    continue
            # if (result := f.result()) is not None and result.iloc[0] is not None:
            #     res.extend(*result)
            f.release()
            if n_left > 0:
                future = client.compute(df.get_partition(gediDf.npartitions - n_left))
                futures_monitor.add(future)
                print(f'************ partition {gediDf.npartitions - n_left} submitted ****************')
                print(f'{futures_monitor.count()} in processing, {n_left} waiting')
                n_left -= 1
            
        if len(res) > 0:
            xrrs = xr.concat(res, dim='time', compat='override', coords='minimal', join='override')
            xrrs = xrrs.to_dataset('input')
            xrrs['time'].encoding['dtype'] = 'float32'
            xrrs.to_netcdf(self.save_dir / 'GEDI.h5', group=f'zone{i}/{self.year}', format='NETCDF4', engine='h5netcdf', encoding=self.comp, mode='a')
        flag.touch()
        return

    def get_patch_for_partition(self, partition, zone:str, esa_wc_items:pystac.ItemCollection,glo30_itmes:pystac.ItemCollection, rewrite:bool=False, partition_info:dict=None):
        """
        Query, filter, and stack S2 and ESA world cover patches for each GEDI partition.
        GEDI data is partitioned to cache a number of locations for the sake of memory efficiency.

        Args:
            partition (pandas.DataFrame): The partition containing the data.
            zone (str): The zone identifier.
            esa_wc_items (pystac.ItemCollection): The collection of ESA WC items.
            partition_info (dict, optional): Information about the partition.
        """
        
        flag = self.save_dir / zone / f'{self.year}_partition_{partition_info["number"]}_done'
        if flag.exists() and not rewrite:
            print(f'{zone} {self.year}_partition_{partition_info["number"]} has been processed.')
            return
        xrrs = partition.apply(self.get_best_s2_for_point, axis=1, args=(esa_wc_items, glo30_itmes)).dropna()
        if xrrs.empty:
            return

        partition = partition.loc[xrrs.index].set_index('shot_number')
        rh_da = partition[rh_dtype.keys()].to_xarray().to_dataarray('rh', 'rhs')
        gedi_attr_da = partition[gedi_attr_dtype.keys()].to_xarray().to_dataarray('attr', 'gedi_attrs')
        xrrs = dask.compute(*xrrs)
        slope_da = xr.concat([s['slope'] for s in xrrs], dim='time', compat='override', coords='minimal', join='override')
        xrrs = xr.concat([s['xrr'] for s in xrrs], dim='time', compat='override', coords='minimal', join='override')
        xrrs.name = 'input'
        xrrs = xr.merge([xrrs, slope_da, rh_da.transpose(), gedi_attr_da.transpose()])
        with Lock('netcdf_lock'):
            xrrs.to_netcdf(self.save_dir / f'{zone}.h5', group=f'{self.year}/{partition_info["number"]}', format='NETCDF4', engine='h5netcdf', encoding=self.comp, mode='a')
        flag.touch()

    def get_best_s2_for_point(self, point, esa_wc_items, glo30_itmes):
        '''
        Qeury and Filter S2 tiles for each GEDI point
        - Query by date, cloud cover, water percentage, and Filter by leaf on/off dates
        - Filter by defective cover (patch level)
        '''
        # get the time range
        if point['leaf_on_doy'] < 366 and point['leaf_off_flag'] == 0: # point captured in growing season
            leaf_on_doy = pd.to_timedelta(int(point['leaf_on_doy']), unit='D')
            leaf_off_doy = pd.to_timedelta(int(point['leaf_off_doy']), unit='D')
            if leaf_on_doy > leaf_off_doy:
                leaf_off_doy += pd.to_timedelta(365, unit='D')
            start = self.yearStart + leaf_on_doy
            end = self.yearStart + leaf_off_doy
        else:
            start = pd.Timestamp(point['date'], tz='UTC') - self.queryDaysRange
            end = pd.Timestamp(point['date'], tz='UTC') + self.queryDaysRange
        geom = gpd.points_from_xy([point['lat']], [point['lon']], crs='epsg:4326')
        items = self.query_s2_for_p(start, end, geom) #? how to make it non-blocking, return a future
        
        if items is None:
            return None

        # get patch and calculate defective cover
        epsg = get_most_common_epsg(items)
        geom = geom.to_crs(epsg)[0]
        bounds = geom.buffer(self.buffer_size).bounds
        bounds_slope = geom.buffer(self.buffer_size+10).bounds

        best = self.calculate_defective_cover(items, bounds, point['date'], epsg)
        if best is None:
            return
        best_item = [item for item in items if item.id == best.id][0]
        s2xrr = get_patch(best_item, assets=self.bands, bounds=bounds, epsg=epsg)
        wc_xrr = get_patch(esa_wc_items, assets=['map'], bounds=bounds, epsg=epsg)
        glo_xrr = get_patch(glo30_itmes, assets=['data'], bounds=bounds_slope, epsg=epsg, fill_value=np.nan, dtype='float32')
        if glo_xrr.shape[0] == 0 or wc_xrr.shape[0] == 0:
            return None
        glo_xrr = glo_xrr.max(dim='time', skipna=True)
        slope_xrr = slope(glo_xrr[0])
        slope_xrr = slope_xrr[1:-1, 1:-1] #remove nan
        slope_xrr = slope_xrr.expand_dims(dim={'time': s2xrr['time'].data}, axis=0)

        wc_xrr = wc_xrr.max(dim='time', skipna=True)
        wc_xrr = wc_xrr.expand_dims(dim={'time': s2xrr['time'].data}, axis=0)
        wc_xrr = wc_xrr.assign_coords(band=['esa_wc'])

        s2xrr = s2xrr.assign_coords(spec=('time', [pickle.dumps(s2xrr.spec)]))
        xrr = xr.concat([s2xrr, wc_xrr], dim='band', compat='override', coords='minimal', combine_attrs='drop')
        xrr = xrr.drop_vars('epsg')
        best.delta_day = best.delta_day.astype('uint16')
        best.defective_cover = best.defective_cover.astype('float32')
        new_coords = {k: ("time", [best[k]]) for k in ['delta_day','defective_cover']}
        xrr = xrr.assign_coords(new_coords)
        return {'xrr': xrr, 'slope': slope_xrr}

    
    def calculate_defective_cover(self, items, bounds, date, epsg):
        '''
        Calculate defective cover (patch level) for tiles with the same epsg
        
        Args:
            group (pandas.DataFrame): DataFrame of sentinel-2 tiles with the same epsg
            geom (geopandas.GeometryArray): geometry of the point
            
        Returns:
            pandas.Series: Series containing the best item
        '''
        patch = get_patch(items, ['SCL'], resolution=10, bounds=bounds, epsg=epsg, dtype='uint8')
        
        if patch.shape[0] == 0 or patch.shape[-2:] != (self.patch_size, self.patch_size): # why there're cases that the output shape is (14,15)? fill_value doesn't work?
            return None

        patch = patch.compute() # simplify compute graph, not sure if this is necessary, 
        # patch = patch.isin(defective_SCL).sum(dim=['x', 'y']) / np.prod(patch.shape[-2:]) 
        # even patch is computed, patch.isin().sum() will still be lazy
        scl = patch.data
        defective_cover = np.any([(scl == k) for k in defective_SCL], 0).sum() / np.prod(scl.shape[-2:])
        patch_df = pd.DataFrame({
            'id': patch.id.values,
            'defective_cover': defective_cover,
            'delta_day': [np.abs(t - pd.Timestamp(date)).days for t in patch.time.values],
        })
        if patch_df.empty or patch_df['defective_cover'].isna().all():
            return None
        
        patch_df = patch_df.sort_values(['defective_cover', 'delta_day'])
        best = patch_df.iloc[0]

        return best

    @retry.retry(tries=10, delay=1)
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
        search = api.search(collections=['sentinel-2-l2a'],
                            query={
                                "eo:cloud_cover": {
                                    "lt": self.maxCloudCover
                                },
                                's2:water_percentage': {
                                    'lt': self.maxWaterPercentage
                                }
                            },
                            bbox=geom.total_bounds,
                            datetime=f'{str(start)[:10]}/{str(end)[:10]}')
        items = search.item_collection()
        if len(items) > 0:
            return items
        if len(items) == 0 and (end - start).days < 365:
            print(f'No S2 tile found between {start} - {end}, extend the range by {self.extendDays.days*2} days')
            items = self.query_s2_for_p(start - self.extendDays,
                                           end + self.extendDays, geom)
            return items
        else:
            return None


#%%
@hydra.main(config_path="config", config_name="s2_download", version_base="1.2")
def main(cfg):
    from dask.distributed import Client, LocalCluster
    from dask import config 
    config.set({'interface': 'lo'}) # failed to fix the error: dask.distributed - ERROR - Failed to gather key
    cluster = LocalCluster()
    client = Client(cluster)#timeout

    t0 = time.time()
    s2downloader = S2Downloader(**cfg)
    res = s2downloader.download_zone(cfg.zone, cfg.rewrite)
    print('time: ', time.time() - t0)


def download(func, zone):
    try:
        func(zone)
    except Exception as e:
        print(e)
        time.sleep(60*10)
        print("restarting")
        download(func, zone)

#%%
if __name__ == "__main__":
    # from omegaconf import DictConfig, OmegaConf
    # cfg = OmegaConf.load('config/s2_download.yaml')
    main()

# %%

# %%
