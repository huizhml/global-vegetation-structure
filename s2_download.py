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

defective_SCL = [0, 1, 8, 9, 10, 11]  # keep cloud shadows, model should learn to be invariant to cloud shadows
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

def get_tile_by_id(tile_id):
    """
    Get the sentinel-2 tile by tile id.
    """
    url = f'{stac_endpoint}/collections/sentinel-2-l2a/items/{tile_id}'
    item = pystac.Item.from_file(url)
    return planetary_computer.sign_inplace(item) #TO CHCEK: the token generated seems to be only valid for 1 hour

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
        
        bounds = self.get_zone_bbox(zone)
        esa_wc_items = api.search(
            collections=['esa-worldcover'],
            bbox=bounds,
            datetime=f'{self.esa_wc_year}-01-01/{self.esa_wc_year}-12-31').item_collection()
        
        
        gediDf = ddf.read_parquet(zoneFolder / 'partition_*.parquet', dropna=True, usecols=list(dtypes.keys()))
        # gediDf = gediDf.set_index('system:index')
        # gediDf = gediDf.repartition(npartitions=16)
        print(f'Processing {gediDf.npartitions} partitions...')
        # number = 98
        # df = self.get_patch_for_partition(gediDf.get_partition(number).compute(), zone, esa_wc_items, partition_info={'number': number})
        df = gediDf.map_partitions(self.get_patch_for_partition, zone, esa_wc_items, meta=(None, 'string')).compute()
        # client = get_client()
        # res = client.compute(df)
        # res.result()
        # flag.touch()
        # flag.write_text(f'partition size used: {self.partition_size}')
        return

    def get_patch_for_partition(self, partition, zone, esa_wc_items, partition_info=None):
        if (self.save_folder / f'{zone}'/f'partition_{partition_info["number"]}.h5').exists():
            print(f'{zone} partition_{partition_info["number"]} has been processed.')
            return
        xrrs = partition.apply(self.get_best_s2_for_p, axis=1, args=(esa_wc_items,)).dropna()
        xrrs = dask.compute(*xrrs)
        xrrs = xr.concat(xrrs, dim='time', compat='override', coords='minimal', join='override')
        xrrs['time'].encoding['dtype'] = 'float32'

        with Lock('netcdf_lock'):
            xrrs.to_netcdf(self.save_folder / f'{zone}'/f'partition_{partition_info["number"]}.h5', format='NETCDF4', engine='h5netcdf', encoding={xrrs.name: comp}, mode='w')
        print(f'finish {zone} partition {partition_info["number"]}')
        del partition
        del xrrs


    # @dask.delayed        
    def get_best_s2_for_p(self, point, esa_wc_items):
        '''
        Filter S2 tiles for each GEDI point
        Query by date, cloud cover, water percentage, and Filter by leaf on/off dates
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
        items_df = self.query_s2_for_p(start, end, geom) #? how to make it non-blocking, return a future
        
        if items_df is None:
            return None

        # get patch and calculate defective cover
        items_df = items_df.groupby('epsg', group_keys=False).apply(self.calculate_defective_cover, geom, point['date']).dropna()
        if items_df.empty:
            return None

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
            bounds=bounds, epsg=epsg, properties=False
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
        elif xrr.shape[0] == 0:
            return None
        else:
            xrr = xrr.max(dim='time', keep_attrs=True) #TODO: check if this is correct
            xrr = xrr.expand_dims(dim={'time': s2xrr['time'].data}, axis=0)
        del s2xrr.attrs['spec']
        xrr = xr.concat([s2xrr, xrr], dim='band', compat='override', coords='minimal')
        xrr = xrr.expand_dims(dim={'RHs': np.array(point[f'rh{x}'] for x in range(101))}, axis=1)
        new_coords = {k: ("time", [v]) for k, v in best[['delta_day', 'defective_cover']].items()}
        new_coords.update({k: ("time", [v]) for k, v in point.items() if not k.startswith('rh')})
        xrr = xrr.assign_coords(new_coords)
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
        
        if patch.shape[0] == 0:
            return None

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
    res = s2downloader.get_s2_for_zone('56H')
    print('time:', time.time() - time_start)


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
    from dask.distributed import Client, LocalCluster
    cluster = LocalCluster(threads_per_worker=2)
    client = Client(cluster)

    t0 = time.time()
    s2downloader = S2Downloader("GEDI2019", partition_size='64K')
    # download(s2downloader.get_s2_for_zone, '56H')
    res = s2downloader.get_s2_for_zone('56H')
    print('time: ', time.time() - t0)
    # with ipdb.launch_ipdb_on_exception():
    # main()

# %%

# %%
