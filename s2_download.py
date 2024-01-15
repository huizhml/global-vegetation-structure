#%%
import os
import time
import json
from pathlib import Path
from typing import Dict
from collections import defaultdict

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

import ipdb
import hydra
from dotenv import load_dotenv

from utils._stackstac import stack
from const import dtypes, s2_item_props

load_dotenv('.planetarycomputer/settings.env')
os.environ["GDAL_HTTP_MAX_RETRY"] = "3"

#%%
stac_endpoint = 'https://planetarycomputer.microsoft.com/api/stac/v1'
api = pystac_client.Client.open(stac_endpoint, modifier=planetary_computer.sign_inplace)

dtypes.pop('.geo')
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
        self.comp = {
            'input':{
                "zlib": True,
                "complevel": comp_level,
                "fletcher32": True,
                "chunksizes": (1,14,15,15)
            },
            'rhs':{
                "zlib": True,
                "complevel": comp_level,
                "fletcher32": True,
                "chunksizes": (1,101)
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
    
    def download_zone(self, zone:str=None, update:bool=False):
            '''
            Filter S2 tiles for each GEDI zone.

            Args:
                zone (str): The GEDI zone to filter S2 tiles for.

            Returns:
                str: A string indicating the status of the processing. Returns 'done' if the zone and year have already been processed.
            '''
            zoneFolder = self.gediFolder / zone
            (self.save_dir/zone).mkdir(exist_ok=True)
            flag = self.save_dir/ f'{zone}_{self.year}_done'
            if flag.exists():
                print(f'{zone} {self.year} has been processed.')
                return
            
            bounds = self.get_zone_bbox(zone)
            esa_wc_items = api.search(
                collections=['esa-worldcover'],
                bbox=bounds,
                datetime=f'{self.esa_wc_year}-01-01/{self.esa_wc_year}-12-31').item_collection()
            
            gediDf = dd.read_parquet(zoneFolder / 'partition_*.parquet', dropna=True, usecols=list(dtypes.keys()))
            print(f'Processing {gediDf.npartitions} partitions...')
            # number = 6
            # df = self.get_patch_for_partition(gediDf.get_partition(number).compute(), zone, esa_wc_items, partition_info={'number': number})
            res = gediDf.map_partitions(self.get_patch_for_partition, zone, esa_wc_items, meta=(None, 'string'))
            if gediDf.npartitions <= self.n_parallel:
                res.compute() #? how to retry failed partitions in this way?
            else:
                client = get_client()
                futures = []
                for i in range(self.n_parallel):
                    future = client.compute(res.get_partition(i))
                    futures.append(future)
                
                futures_monitor = as_completed(futures, with_results=False)
                n_left = gediDf.npartitions - self.n_parallel
                max_retries = 3
                retry_counter: Dict[str, int] = defaultdict(lambda: 0)   
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
                    if n_left > 0:
                        future = client.compute(res.get_partition(gediDf.npartitions - n_left))
                        futures_monitor.add(future)
                        print(f'************ partition {gediDf.npartitions - n_left} submitted ****************')
                        print(f'{futures_monitor.count()} in processing, {n_left} waiting')
                        n_left -= 1
            flag.touch()
            return 'done'

    def get_patch_for_partition(self, partition, zone:str, esa_wc_items:pystac.ItemCollection, partition_info:dict=None):
            """
            Query, filter, and stack S2 and ESA world cover patches for each GEDI partition.
            GEDI data is partitioned to cache a number of locations for the sake of memory efficiency.

            Args:
                partition (pandas.DataFrame): The partition containing the data.
                zone (str): The zone identifier.
                esa_wc_items (pystac.ItemCollection): The collection of ESA WC items.
                partition_info (dict, optional): Information about the partition.
            """
            
            flag = self.save_dir / zone / f'partition_{partition_info["number"]}_done'
            if flag.exists():
                print(f'{zone} partition_{partition_info["number"]} has been processed.')
                return
            xrrs = partition.apply(self.get_best_s2_for_point, axis=1, args=(esa_wc_items,)).dropna()
            if xrrs.empty:
                return
            xrrs = dask.compute(*xrrs)
            xrrs = xr.concat(xrrs, dim='time', compat='override', coords='minimal', join='override')
            xrrs = xrrs.to_dataset('input')
            xrrs['time'].encoding['dtype'] = 'float32'

            with Lock('netcdf_lock'):
                xrrs.to_netcdf(self.save_dir / f'{zone}.h5', group=f'partition_{partition_info["number"]}', format='NETCDF4', engine='h5netcdf', encoding=comp, mode='w')
            print(f'finish {zone} partition {partition_info["number"]}')
            flag.touch()

    def get_best_s2_for_point(self, point, esa_wc_items):
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
        geom = gpd.points_from_xy([point['x']], [point['y']], crs='epsg:4326')
        items = self.query_s2_for_p(start, end, geom) #? how to make it non-blocking, return a future
        
        if items is None:
            return None

        # get patch and calculate defective cover
        epsg = get_most_common_epsg(items)
        bounds = geom.to_crs(epsg)[0].buffer(70).bounds
        s2_kwargs = dict(
            assets=self.bands, resolution=10, bounds=bounds, band_coords=False, properties=s2_item_props, epsg=epsg, dtype="uint16", fill_value=0
        )
        wc_kwargs = dict(
            assets=['map'],
            band_coords=False,
            resolution=10, 
            bounds=bounds, epsg=epsg, properties=False, dtype="uint8", fill_value=0
        )
        rh_arr = np.array([[point[f'rh{x}'] for x in range(101)]])
        gedi_attr = {k: ("time", [point[k]]) for k  in dtypes.keys()}

        best = self.calculate_defective_cover(items, bounds, point['date'], epsg)
        if best is None:
            return 
        best_item = [item for item in items if item.id == best.id][0]
        try:
            s2xrr = stack(best_item, **s2_kwargs)
        except:
            # token might expire, sign again
            items = resign_items(best_item)
            s2xrr = stack(items, **s2_kwargs)
        try:
            xrr = stack(esa_wc_items, **wc_kwargs)
        except:
            # token might expire, sign again
            items = resign_items(esa_wc_items)
            xrr = stack(items, **wc_kwargs)

        xrr = xrr.dropna(dim='time', how='all') # drop nan time slices
        if xrr.shape[0] == 1: # bbox cross two grid cells of esa wc
            xrr['time'] = s2xrr['time'].data
        elif xrr.shape[0] == 0:
            return None
        else:
            xrr = xrr.max(dim='time', keep_attrs=True) #TODO: check if this is correct
            xrr = xrr.expand_dims(dim={'time': s2xrr['time'].data}, axis=0)
        del s2xrr.attrs['spec']
        xrr = xrr.assign_coords(band=['esa_wc'])
        xrr = xr.concat([s2xrr, xrr], dim='band', compat='override', coords='minimal')
        xrr.name = 'input'
        xrr = xrr.to_dataset()
        xrr = xrr.assign(rhs=(('time', 'rhs'), rh_arr))
        new_coords = {k: ("time", [best[k]]) for k in ['delta_day','defective_cover']}
        new_coords.update(gedi_attr)
        xrr = xrr.assign_coords(new_coords)
        xrr = xrr.to_dataarray('input')
        return xrr

    
    def calculate_defective_cover(self, items, bounds, date, epsg):
        '''
        Calculate defective cover (patch level) for tiles with the same epsg
        
        Args:
            group (pandas.DataFrame): DataFrame of sentinel-2 tiles with the same epsg
            geom (geopandas.GeometryArray): geometry of the point
            
        Returns:
            pandas.Series: Series containing the best item
        '''
        try:
            patch = stack(items, ['SCL'], resolution=10, bounds=bounds, epsg=epsg, fill_value=0, band_coords=False, properties=False, dtype='uint8')
        except:
            # token might expire, sign again
            items = resign_items(items)
            patch = stack(items, ['SCL'], resolution=10, bounds=bounds, epsg=epsg, fill_value=0, band_coords=False, properties=False, dtype='uint8')
        
        if patch.shape[0] == 0:
            return None

        patch = patch.sel(band='SCL').compute() # simplify compute graph, not sure if this is necessary
        patch = patch.isin(defective_SCL).sum(dim=['x', 'y']) / np.prod(patch.shape[-2:])
        patch.name = 'defective_cover'
        patch_df = pd.DataFrame({
            'id': patch.id.values,
            'defective_cover': patch.data,
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
    s2downloader = S2Downloader(2019, n_parallel=cfg.n_parallel, save_dir='data/GEDI')
    res = s2downloader.download_zone(cfg.zone)
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
    main()

# %%

# %%
