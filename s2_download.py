#%%
import os
import time
import json
from pathlib import Path
import matplotlib.pyplot as plt

import numpy as np
import xarray as xr

import ee
import pystac
import pystac_client
import planetary_computer
import retry


import dask
import dask.array as da
import dask.dataframe as ddf
from dask.distributed import Lock, as_completed, futures_of
from distributed import get_client
import pandas as pd
import geopandas as gpd
import dask_geopandas as dgd
from shapely.geometry import MultiPoint

import ipdb
import hydra
from dotenv import load_dotenv

from utils._stackstac import stack
from const import dtypes, s2_item_props, wc_item_props

load_dotenv('.planetarycomputer/settings.env')
# MPC_API_KEY = os.environ.get('PC_SDK_SUBSCRIPTION_KEY')

#%%
stac_endpoint = 'https://planetarycomputer.microsoft.com/api/stac/v1'
api = pystac_client.Client.open(stac_endpoint, modifier=planetary_computer.sign_inplace)

s2asset = api.get_collection("sentinel-2-l2a").assets["geoparquet-items"]
defective_SCL = [
    0, 1, 8, 9, 10, 11
]  # keep cloud shadows, model learns to be invariant to cloud shadows
dropped_vars = [
    'full_width_half_max', 'center_wavelength', 'common_name', 'title', 'gsd', 'proj:bbox',
    's2:datastrip_id', 's2:datatake_id', 's2:datatake_type', 's2:degraded_msi_data_percentage', 
    's2:generation_time', 's2:granule_id', 's2:mean_solar_azimuth', 's2:mean_solar_zenith', 's2:processing_baseline', 's2:product_type',
    's2:product_uri', 's2:reflectance_conversion_factor'
]
comp = {
    "zlib": True,
    "complevel": 7,
    "fletcher32": True,
    "chunksizes": (1,101,14,15,15)
}


def resign_items(items):
    res = []
    for item in items:
        item = planetary_computer.sign(item)
        res.append(item)
    return res


def filter_S2_by_gs(leaf_on_date, leaf_off_date, tile_group):
    """
    Filters a tile group by leaf on/off dates.

    Args:
        leaf_on_date (np.datetime64): The date when the leaves start to appear.
        leaf_off_date (np.datetime64): The date when the leaves start to disappear.
        tile_group (pandas.DataFrame): The group of sentinel-2 tiles to be filtered.

    Returns:
        pandas.DataFrame: The filtered tile group.
    """
    if leaf_on_date > leaf_off_date:
        leaf_off_date += 365
    filtered = tile_group[(tile_group['s2_datetime'] >= leaf_on_date) &
                          (tile_group['s2_datetime']
                           <= leaf_off_date)]  # filter by leaf on/off

    if len(filtered) == 0:
        filtered = filter_S2_by_gs(
            leaf_on_date - 30, leaf_off_date + 30,
            tile_group)  # widen the range #? infinite loop?
    return filtered

def get_tile_by_id(tile_id):
    """
    Get the sentinel-2 tile by tile id.
    """
    url = f'{stac_endpoint}/collections/sentinel-2-l2a/items/{tile_id}'
    item = pystac.Item.from_file(url)
    return planetary_computer.sign_inplace(item)

class S2Downloader:
    test_file = 'GEDI02_A_2019111131802_O02014_01_T03046_02_003_01_V002.parquet'
    test_zone = '01G'

    def __init__(self,
                 gediFolder='GEDI2019',
                 partition_size='64K',
                 debug=False):
        self.gediFolder = Path.home() / gediFolder
        self.save_folder = Path.home() / 'data'/ 'GEDI'
        self.year = int(gediFolder[-4:])
        self.gedi_start = pd.Timestamp('2018-01-01', tz='UTC')
        self.yearStart = pd.Timestamp(f'{self.year}-01-01', tz='UTC')  # np.datetime64(f'{self.year}-01-01', 'D') #
        self.esa_wc_year = 2020 if self.year <= 2020 else 2021
        self.maxCloudCover = 50
        self.maxWaterPercentage = 100
        self.queryDaysRange = pd.to_timedelta(90, unit='D')
        self.extendDays = pd.to_timedelta(30, unit='D')
        self.partition_size = partition_size
        self.debug = debug
        self.bands = [
            'B01', 'B04', 'B03', 'B02', 'B05', 'B06', 'B07', 'B08', 'B8A',
            'B09', 'B11', 'B12', 'SCL'
        ]
        self.patch_size = 15
        self.stac_item_keys = [
            'id', 'type', 'stac_version', 'x', 'y', 'bbox', 'collection',
            'assets'
        ]
        self.stac_item_props = [
            'datetime', 'proj:epsg', 's2:mgrs_tile', 'eo:cloud_cover',
            'sat:orbit_state', 's2:generation_time', 's2:water_percentage',
            's2:nodata_pixel_percentage'
        ]
      
    def get_zone_bbox(self, zone):
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
    
    def get_s2_for_zone(self, zone):
        '''
        Filter S2 tiles for each GEDI zone
        '''
        zoneFolder = self.gediFolder / zone
        (self.save_folder/zone).mkdir(exist_ok=True)
        flag = self.save_folder/ f'{zone}_done'
        if flag.exists():
            print(f'{zone} has been processed.')
            return
        gediDf = ddf.read_parquet(zoneFolder / 'partition_*.parquet', dropna=True, usecols=list(dtypes.keys()))
        print(f'Processing {gediDf.npartitions} partitions...')
        # with_growing_season = gediDf['leaf_on_doy'] < 366
        # # leaf off date is in the next year
        # reverse = gediDf['leaf_on_doy'] > gediDf['leaf_off_doy']
        # gediDf['leaf_off_doy'] = gediDf['leaf_off_doy'].mask(reverse, gediDf['leaf_off_doy'] + 365)
        # gediDf['leaf_on_doy'] = gediDf['leaf_on_doy'].astype('int16')
        # gediDf['leaf_on_doy'] = ddf.to_timedelta(gediDf['leaf_on_doy'], unit='D')
        # gediDf['leaf_off_doy'] = gediDf['leaf_off_doy'].astype('int16')
        # gediDf['leaf_off_doy'] = ddf.to_timedelta(gediDf['leaf_off_doy'], unit='D')

        # gediDf['start'] = ddf.to_datetime(gediDf['date']) - self.queryDaysRange
        # gediDf['end'] = ddf.to_datetime(gediDf['date']) + self.queryDaysRange
        # gediDf['start'] = gediDf['start'].mask(with_growing_season, self.yearStart + gediDf['leaf_on_doy'])
        # gediDf['end'] = gediDf['end'].mask(with_growing_season, self.yearStart + gediDf['leaf_off_doy'])

        bounds = self.get_zone_bbox(zone)
        esa_wc_items = api.search(
            collections=['esa-worldcover'],
            bbox=bounds,
            datetime=f'{self.esa_wc_year}-01-01/{self.esa_wc_year}-12-31').item_collection()

        df = gediDf.map_partitions(self.get_patch_for_partition, zone, esa_wc_items, meta=(None, 'string')).compute()
        # df = self.get_patch_for_partition(gediDf.get_partition(0).compute(), zone, esa_wc_items)
        # xrrs = gediDf.apply(self.get_best_s2_for_p, axis=1, args=(esa_wc_items,), meta=('arr', 'object'))
        # xrrs = xr.concat(xrrs, dim='time', compat='override', coords='minimal', join='override')
        # xrrs['time'].encoding['dtype'] = 'float64'
        # xrrs = xrrs.to_dataset()
        # paths = [self.save_folder/zone/f'partition_{i}' for i in range(gediDf.npartitions)]
        # xr.save_mfdataset(xrrs, [self.save_folder/f'{zone}.h5'])
        # client = get_client()
        # res = client.compute(df)
        # res.result()
        # flag.touch()
        # flag.write_text(f'partition size used: {self.partition_size}')
        return

    def get_patch_for_partition(self, partition, zone, esa_wc_items, partition_info=None):
        # if (self.save_folder / f'{zone}'/f'partition_{partition_info["number"]}.h5').exists():
        #     print(f'{zone} partition_{partition_info["number"]} has been processed.')
        #     return
        # client = get_client()
        xrrs = partition.apply(self.get_best_s2_for_p, axis=1, args=(esa_wc_items,))
        xrrs = dask.compute(*xrrs)
        xrrs = xr.concat(xrrs, dim='time', compat='override', coords='minimal', join='override')
        xrrs['time'].encoding['dtype'] = 'float32'
        # xrrs = client.compute(xrrs)
        # xrrs = xrrs.chunk({'time': 1, 'RHs': 101, 'band':14, 'x': 15, 'y': 15})
        with Lock('netcdf_lock'):
            xrrs.to_netcdf(self.save_folder / f'{zone}'/f'partition_{partition_info["number"]}.h5', format='NETCDF4', engine='h5netcdf', encoding={xrrs.name: comp}, mode='w')
        print(f'finish {zone} partition {partition_info["number"]}')
        del partition
        del xrrs

    def save_patches(self, patches, zone, partition_num):
        # client = get_client()
        # patches = client.compute(patches)
        with Lock('netcdf_lock'):
            patches.to_netcdf(self.save_folder / f'{zone}'/f'partition_{partition_num}.h5', format='NETCDF4', engine='h5netcdf', encoding={patches.name: comp}, mode='w')
        print(f'finish {zone} partition {partition_num}')

    # @dask.delayed        
    def get_best_s2_for_p(self, point, esa_wc_items):
        '''
        Filter S2 tiles for each GEDI point
        Query by date, cloud cover, water percentage, and Filter by leaf on/off dates
        '''
        delta_time = pd.to_timedelta(int(point['delta_time']), unit='S')
        date = self.gedi_start + delta_time
        # get the time range
        if point['leaf_on_doy'] < 366 and point['leaf_off_flag'] == 0: # point captured in growing season
            leaf_on_doy = pd.to_timedelta(int(point['leaf_on_doy']), unit='D')
            leaf_off_doy = pd.to_timedelta(int(point['leaf_off_doy']), unit='D')
            if leaf_on_doy > leaf_off_doy:
                leaf_off_doy += pd.to_timedelta(365, unit='D')
            start = self.yearStart + leaf_on_doy
            end = self.yearStart + leaf_off_doy
        else:
            start = date  - self.queryDaysRange
            end = date + self.queryDaysRange
        geom = gpd.points_from_xy([point['x']], [point['y']], crs='epsg:4326')
        items_df = self.query_s2_for_p(start, end, geom) #? how to make it non-blocking, return a future
        
        if items_df is None:
            return None

        # get patch and calculate defective cover
        # items_df = gpd.GeoDataFrame.from_dict(items.to_dict(), crs="epsg:4326")
        items_df = items_df.groupby('epsg', group_keys=False).apply(self.calculate_defective_cover, geom, date)
        items_df = items_df.sort_values(['defective_cover', 'delta_day'])
        best = items_df.iloc[0]
        if items_df['defective_cover'].isna().all():
            return None

        # get patch
        epsg = best['items'].properties['proj:epsg']
        geom = geom.to_crs(epsg)
        bounds = geom[0].buffer(70).bounds
        
        kwargs = dict(
            assets=self.bands, resolution=10, bounds=bounds, band_coords=False, properties=s2_item_props       
        )
        # items = get_tile_by_id(best['id'])
        try:
            s2xrr = stack(best['items'], **kwargs)
        except:
            # token might expire, sign again
            items = resign_items(best['items'])
            s2xrr = stack(items, **kwargs)

        kwargs = dict(
            assets=['map'],
            band_coords=False,
            resolution=10, 
            bounds=bounds, epsg=epsg, properties=wc_item_props
        )
        try:
            xrr = stack(esa_wc_items, **kwargs)
        except:
            # token might expire, sign again
            items = resign_items(esa_wc_items)
            xrr = stack(items, **kwargs)

        xrr = xrr.dropna(dim='time', how='all') # drop nan time slices
        if xrr.shape[0] == 1: # bbox cross two grid celss of esa wc
            xrr['time'] = s2xrr['time'].data
        else:
            xrr = xrr.max(dim='time', keep_attrs=True) #TODO: check if this is correct
            xrr = xrr.expand_dims(dim={'time': s2xrr['time'].data}, axis=0)
        del s2xrr.attrs['spec']
        xrr = xr.concat([s2xrr, xrr], dim='band', compat='override', coords='minimal')
        # xrr = xr.DataArray(da.concatenate([s2xrr.data, xrr.data], axis=1), dims=['time', 'band', 'x', 'y'], coords={'time': s2xrr['time'].data, 'band': self.bands+['esa_wc'], 'x': s2xrr['x'], 'y': s2xrr['y']}, attrs=s2xrr.attrs)
        xrr = xrr.expand_dims(dim={'RHs': np.array(point[f'rh{x}'] for x in range(101))}, axis=1)
        point = point.drop(['track_id']) #, 'start', 'end'
        new_coords = {k: ("time", [v]) for k, v in best[['delta_day', 'defective_cover']].items()}
        new_coords.update({k: ("time", [v]) for k, v in point.items() if not k.startswith('rh')})
        xrr = xrr.assign_coords(new_coords)
        # xrr = xrr.chunk({'time': 1, 'RHs': 101, 'band':14, 'x': 15, 'y': 15})
        return xrr
    
    def calculate_defective_cover(self, group, geom, date):
        '''
        Calculate defective cover (patch level) for tiles with the same epsg
        
        Args:
            group (pandas.DataFrame): DataFrame of sentinel-2 tiles with the same epsg
            geom (geopandas.GeometryArray): geometry of the point
            
        Returns:
            pandas.DataFrame: DataFrame containing calculated defective cover information
        '''
        items = group['items'].values.tolist()
        epsg = items[0].properties['proj:epsg']
        # Do the buffer for different epsg
        geom = geom.to_crs(epsg)
        bounds = geom[0].buffer(70).bounds

        try:
            patch = stack(items, ['SCL'], resolution=10, bounds=bounds, fill_value=0, band_coords=False, properties=False)
        except:
            # token might expire, sign again
            items = resign_items(items)
            patch = stack(items, ['SCL'], resolution=10, bounds=bounds, fill_value=0, band_coords=False, properties=False)
        
        patch = patch.sel(band='SCL').isin(defective_SCL).sum(dim=['x', 'y']) / np.prod(patch.shape[-2:])

        # patch = patch.assign_coords({'delta_day':('time', [np.abs(t - pd.Timestamp(date)).days for t in patch.time.values])})
        patch.name = 'defective_cover'

        # patch.to_dataframe()
        # patch = patch.to_dask_dataframe()#.sort_values(['defective_cover', 'delta_day'])
        if len(group) != len(patch):
            group['id'] = group.apply(lambda x: x['items'].id, axis=1)
            group = group[group['id'].isin(patch.id.values)]
            items = group['items'].values.tolist()

        try:
            df = pd.DataFrame({
                's2_id': patch.id.values,
                'defective_cover': patch.data,
                'delta_day': [np.abs(t - pd.Timestamp(date)).days for t in patch.time.values],
                'items': items,
            })
        except:
            time.sleep(60*10)
            df = self.calculate_defective_cover(group, geom, date)
        return df

    @retry.retry(tries=10, delay=1)
    def query_s2_for_p(self, start, end, geom):   
        api = pystac_client.Client.open(stac_endpoint, modifier=planetary_computer.sign_inplace)
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
            # return items
            return pd.DataFrame(({
                'id': item.id,
                'items': item,
                'epsg':item.properties['proj:epsg']
                # 'delta_day':
                # np.abs(
                #     pd.Timestamp(item.properties['datetime'], tz='UTC') -
                #     pd.Timestamp(date, tz='UTC')).days
            } for item in items))
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
    cluster = LocalCluster()
    client = Client(cluster)
    time_start = time.time()
    s2downloader = S2Downloader("GEDI2019", partition_size=cfg.partition_size)
    # client.submit(s2downloader.get_s2_for_zone, cfg.zone)
    res = client.submit(s2downloader.get_s2_for_zone, '40M')
    print('time:', time.time() - time_start)

    # from dask.distributed import Client
    # client = Client(n_workers=1, threads_per_worker=1, memory_limit='8GB')
    # res = client.submit(s2downloader.get_s2_for_zone, '40M')
    # ipdb.set_trace()
    # print(res)

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
    key_file = 'keys/private-key.json'
    key = json.load(open(key_file))
    credentials = ee.ServiceAccountCredentials(key['client_email'], key_file)
    ee.Initialize(credentials)
    from dask.distributed import Client, LocalCluster
    cluster = LocalCluster()
    client = Client(cluster)

    t0 = time.time()
    s2downloader = S2Downloader("GEDI2019", partition_size='64K')
    # download(s2downloader.get_s2_for_zone, '56H')
    res = s2downloader.get_s2_for_zone('01G')
    print('time: ', time.time() - t0)
    # with ipdb.launch_ipdb_on_exception():
    # main()

# %%

# %%
