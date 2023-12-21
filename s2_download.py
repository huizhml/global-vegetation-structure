#%%
import os
import time
from pathlib import Path
import matplotlib.pyplot as plt

import numpy as np
import xarray as xr

import pystac
import pystac_client
import planetary_computer
import retry


import dask
import dask.array as da
import dask.dataframe as ddf
import pandas as pd
import geopandas as gpd
import dask_geopandas as dgd
from shapely.geometry import MultiPoint

import ipdb
import hydra

from utils._stackstac import stack
from const import dtypes

#%%
stac_endpoint = 'https://planetarycomputer.microsoft.com/api/stac/v1'
api = pystac_client.Client.open(stac_endpoint,
                                modifier=planetary_computer.sign_inplace)
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


class S2Downloader:
    test_file = 'GEDI02_A_2019111131802_O02014_01_T03046_02_003_01_V002.parquet'
    test_zone = '01G'

    def __init__(self,
                 gediFolder='GEDI2019',
                 partition_size=100,
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
        self.stac_item_keys = [
            'id', 'type', 'stac_version', 'x', 'y', 'bbox', 'collection',
            'assets'
        ]
        self.stac_item_props = [
            'datetime', 'proj:epsg', 's2:mgrs_tile', 'eo:cloud_cover',
            'sat:orbit_state', 's2:generation_time', 's2:water_percentage',
            's2:nodata_pixel_percentage'
        ]
      

    
    def get_s2_for_zone(self, zone):
        '''
        Filter S2 tiles for each GEDI zone
        '''
        zoneFolder = self.gediFolder / zone
        flag = self.save_folder/ f'{zone}_done'
        if flag.exists():
            print(f'{zone} has been processed.')
            return
        gediDf = ddf.read_parquet(zoneFolder / '*.parquet', dropna=True, usecols=list(dtypes.keys()))

        gediDf = gediDf.set_index('system:index')
        # get the number of partitions based on the number of points
        ll = gediDf.map_partitions(len).compute()
        total_points = ll.sum()
        print(f'processing {total_points} locations...')
        partitions = total_points // self.partition_size + 1 
        gediDf = gediDf.repartition(npartitions=partitions)
        res = gediDf.map_partitions(self.get_patch_for_partition, zone, meta=(None, 'string')).compute()
        
        flag.touch()
        flag.write_text(f'partition size used: {self.partition_size}')
        return 

    def get_patch_for_partition(self, partition, zone, partition_info=None):
        flag = self.save_folder/ f'{zone}_partition_{partition_info["number"]}_done'
        if flag.exists():
            print(f'{zone} partition {partition_info["number"]} already exists')
            return
        geom = gpd.points_from_xy(partition['x'], partition['y'])
        esa_wc_items = api.search(
            collections=['esa-worldcover'],
            bbox=geom.total_bounds,
            datetime=f'{self.esa_wc_year}-01-01/{self.esa_wc_year}-12-31').item_collection()
        xrrs = partition.apply(self.get_best_s2_for_p, axis=1, args=(esa_wc_items,))
        xrrs = dask.compute(*xrrs)
        xrrs = xr.concat(xrrs, dim='time', compat='override', coords='minimal', join='override')
        # xrrs = xrrs.chunk({'time': 1, 'RHs': 101, 'band':14, 'x': 15, 'y': 15})
        xrrs['time'].encoding['dtype'] = 'float64'


        xrrs.to_netcdf(self.save_folder / f'{zone}.h5', format='NETCDF4', engine='h5netcdf', encoding={xrrs.name: comp}, group=f'partition_{partition_info["number"]}', mode='a')
        print(f'finish {zone} partition {partition_info["number"]}')
            # flag.touch()
            # flag.write_text(f'{xrrs.shape[0]} locations downloaded, {len(partition) - xrrs.shape[0]} locations failed')
        # except:
        #     print(f'{zone} partition {partition_info["number"]} failed')
        #     failed = self.save_folder / f'{zone}_partition_{partition_info["number"]}_failed'
        #     failed.touch()
        return


    # @dask.delayed        
    def get_best_s2_for_p(self, point, esa_wc_items):
        '''
        Filter S2 tiles for each GEDI point
        Query by date, cloud cover, water percentage, and Filter by leaf on/off dates
        '''
        # get the time range
        leaf_on_doy = pd.to_timedelta(int(point['leaf_on_doy']), unit='D')
        leaf_off_doy = pd.to_timedelta(int(point['leaf_off_doy']), unit='D')
        if leaf_on_doy > leaf_off_doy:
            leaf_off_doy += pd.to_timedelta(365, unit='D')
        if leaf_on_doy.days < 366:
            start = self.yearStart + leaf_on_doy
            end = self.yearStart + leaf_off_doy
        else:
            start = pd.Timestamp(point['date'], tz='UTC') - self.queryDaysRange
            end = pd.Timestamp(point['date'], tz='UTC') + self.queryDaysRange
        geom = gpd.points_from_xy([point['x']], [point['y']], crs='epsg:4326')
        items_df = self.query_s2_for_p(start, end, geom, point['date'])
        
        if items_df is None:
            return None

        # get patch and calculate defective cover

        items_df = items_df.groupby('epsg', group_keys=False).apply(self.calculate_defective_cover, geom, point['shot_number'])

        if items_df['defective_cover'].isna().all():
            return None

        items_df = items_df.sort_values(['defective_cover', 'delta_day'])
        best = items_df.iloc[0]

        # get patch
        epsg = best['items'].properties['proj:epsg']
        geom = geom.to_crs(epsg)
        bounds = geom[0].buffer(70).bounds
        

        try:
            s2xrr = stack(best['items'], self.bands, resolution=10, bounds=bounds)
        except:
            # token might expire, sign again
            items = resign_items(best['items'])
            s2xrr = stack(items, self.bands, resolution=10, bounds=bounds)

        try:
            xrr = stack(esa_wc_items, ['map'], resolution=10, bounds=bounds, epsg=epsg)
        except:
            # token might expire, sign again
            items = resign_items(esa_wc_items)
            xrr = stack(items, ['map'], resolution=10, bounds=bounds, epsg=epsg)

        xrr = xrr.dropna(dim='time', how='all') # drop nan time slices
        if xrr.shape[0] == 1: # bbox cross two grid celss of esa wc
            xrr['time'] = s2xrr['time'].data
        else:
            xrr = xrr.max(dim='time', keep_attrs=True) #TODO: check if this is correct
            xrr = xrr.expand_dims(dim={'time': s2xrr['time'].data}, axis=0)
        del s2xrr.attrs['spec']
        xrr = xr.DataArray(da.concatenate([s2xrr.data, xrr.data], axis=1), dims=['time', 'band', 'x', 'y'], coords={'time': s2xrr['time'].data, 'band': self.bands+['esa_wc'], 'x': s2xrr['x'], 'y': s2xrr['y']}, attrs=s2xrr.attrs)
        xrr = xrr.expand_dims(dim={'RHs': np.array(point[f'rh{x}'] for x in range(101))}, axis=1)
        del best['items']
        point = point.drop(['track_id'])
        new_coords = {k: ("time", [v]) for k, v in best.items()}
        new_coords.update({k: ("time", [v]) for k, v in point.items() if not k.startswith('rh')})
        xrr = xrr.assign_coords(new_coords)
        return xrr
    
    def calculate_defective_cover(self, group, geom, shot_number):
        '''
        Calculate defective cover (patch level) for tiles with the same epsg
        
        Args:
            group (pandas.DataFrame): DataFrame of sentinel-2 tiles with the same epsg
            geom (geopandas.GeometryArray): geometry of the point
            shot_number (int): unique id of the point
            
        Returns:
            pandas.DataFrame: DataFrame containing calculated defective cover information
        '''
        items = group['item'].values.tolist()
        epsg = items[0].properties['proj:epsg']
        # Do the buffer for different epsg
        geom = geom.to_crs(epsg)
        bounds = geom[0].buffer(70).bounds
        try:
            patch = stack(items, ['SCL'], resolution=10, bounds=bounds, fill_value=0)
        except:
            # token might expire, sign again
            items = resign_items(items)
            patch = stack(items, ['SCL'], resolution=10, bounds=bounds, fill_value=0)
        
        patch = patch.sel(band='SCL').isin(defective_SCL).sum(dim=['x', 'y']) / np.prod(patch.shape[-2:])
        if len(group) != len(patch):
            group['id'] = group.apply(lambda x: x['item'].id, axis=1)
            group = group[group['id'].isin(patch.id.values)]
            items = group['item'].values.tolist()

        df = pd.DataFrame({
            's2_id': patch.id.values,
            'defective_cover': patch.data,
            'delta_day': group.delta_day,
            'items': items,
            'shot_number': shot_number
        })

        return df

    @retry.retry(tries=10, delay=1)
    def query_s2_for_p(self, start, end, geom, date):
        api = pystac_client.Client.open(
            stac_endpoint, modifier=planetary_computer.sign_inplace)
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
            return pd.DataFrame(({
                'item': item,
                'epsg':item.properties['proj:epsg'],
                'delta_day':
                np.abs(
                    pd.Timestamp(item.properties['datetime'], tz='UTC') -
                    pd.Timestamp(date, tz='UTC')).days
            } for item in items))
        if len(items) == 0 and (end - start).days < 365:
            print(f'No S2 tile found between {start} - {end}, extend the range by {self.extendDays.days*2} days')
            items_df = self.query_s2_for_p(start - self.extendDays,
                                           end + self.extendDays, geom, date)
            return items_df
        else:
            return None

    def query_parquet_by_date(self, start, end, p):
        s2l2a = dgd.read_parquet(
            s2asset.href,
            filters=[[('datetime', '>=', start), ('datetime', '<=', end)]],
            storage_options=s2asset.extra_fields["table:storage_options"],
            chunksize='10MB')
        if s2l2a.npartitions < 1:
            print('No S2 tile found')
            items_df = self.query_parquet_by_date(start - self.extendDays,
                                                  end + self.extendDays, p)

            # query s2 by location and quality
        geom = gpd.points_from_xy(p['x'], p['y'], crs='epsg:4326') # new data has separated x and y columns
        mask = ((ddf["eo:cloud_cover"] < self.maxCloudCover) &
                (ddf["s2:nodata_pixel_percentage"] < self.maxWaterPercentage)
                & ddf.intersects(geom))

        # if mask.sum() == 0: # TODO: this operation need to call compute()
        #     # ? should we increase the cloud cover threshold?
        #     return

        ddf = ddf[mask]

        gediDate = pd.Timestamp(p['date'], tz='UTC')
        # get stac items & calculate delta days
        items_df = ddf.apply(self.row_to_stac_item,
                             args=(gediDate, ),
                             axis=1,
                             meta={
                                 'item': object,
                                 'epsg': 'string'
                             })
        return items_df

    def row_to_stac_item(self, row, gediDate):
        item = row[self.stac_item_keys].to_dict()
        props = row[self.stac_item_props].to_dict()
        props['delta_day'] = np.abs(row['datetime'] - gediDate).days
        props['datetime'] = props['datetime'].strftime('%Y-%m-%d %H:%M:%S.%f')
        item['properties'] = props
        item = planetary_computer.sign(pystac.Item.from_dict(item))
        return pd.Series([item, item.properties['proj:epsg']],
                         index=['item', 'epsg'])

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

#%%
if __name__ == "__main__":
    from dask.distributed import Client, LocalCluster
    cluster = LocalCluster()
    client = Client(cluster)
# client.submit(s2downloader.get_s2_for_zone, '01G')
    t0 = time.time()
    s2downloader = S2Downloader("GEDI2019", partition_size=100)
    res = s2downloader.get_s2_for_zone('01G')
    print('time: ', time.time() - t0)
    # with ipdb.launch_ipdb_on_exception():
    # main()

# %%

# %%
