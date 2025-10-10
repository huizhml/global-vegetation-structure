import ee
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
import geodatasets
from shapely.ops import transform
from shapely.geometry import mapping
import dask
import requests
import json
from io import StringIO
from shapely.geometry import shape
from dataclasses import dataclass
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig
import hydra
import time
from pathlib import Path
from download._utils import authenticate
from download._dask_downloader import DaskDownloader
from download._5_download_inference import get_s2_tiles_by_landmass

from dotenv import load_dotenv
load_dotenv()
authenticate()

def drop_z(x, y, z=None):
    return (x, y)

def plot_with_basemap(gdf, column, title, cmap='viridis'):
    world = gpd.read_file(geodatasets.get_path('naturalearth.land'))
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    world.boundary.plot(ax=ax)
    gdf.plot(column=column, ax=ax, legend=True, cmap=cmap)
    plt.title(title)
    plt.savefig('output/'+title+'.png', dpi=300)

class GrowingSeason:
    default_onset_north = pd.to_datetime('2020-04-01')
    default_end_north = pd.to_datetime('2020-10-30')
    default_onset_south = pd.to_datetime('2020-10-01')
    default_end_south = pd.to_datetime('2021-04-30')

    def __init__(self, year, asset_id:str=None, save_dir:str=None, **kwargs):
        self.year = year
        self.offset = (year - 2000) * 366
        self.base_date = pd.Timestamp(f'{year}-01-01')
        self.asset_id = asset_id
        self.save_dir = Path(save_dir).expanduser()


    def get_growing_months_per_tile(self):
        '''Number of growing pixels for each month has been aggregated to Sentinel-2 tile in GEE'''
        df_raw = gpd.read_file('~/data/GEDI/Sentinel-2-Shapefile-Index-master/sentinel_2_index_shapefile.shp')
        df_old = get_s2_tiles_by_landmass('~/data/GEDI/Sentinel-2-Shapefile-Index-master/sentinel_2_index_shapefile.shp')# the old one has complete list of tiles
        fc = ee.FeatureCollection(self.asset_id)
        download_id = ee.data.getTableDownloadId({'table': fc, 'fileFormat': 'csv'})
        res = requests.get(ee.data.makeTableDownloadUrl(download_id))
        if res.status_code == 200:
            data = StringIO(res.content.decode('utf-8'))
            df = pd.read_csv(data)            
            df = df.drop(columns=['system:index'])
            df = df.groupby('Name').sum()
            for i in range(1,13):
                df[f'freq_month{i}'] = df[f'month{i}']/df['mask']
            df = df.merge(df_old, left_index=True, right_on='Name', how='outer', right_index=False)
            df = gpd.GeoDataFrame(df, geometry='geometry')
            df = df.drop(columns=['index_right'])
            # get the geometry from the raw sentinel-2 grid cell
            df = df.merge(df_raw, left_on='Name', right_on='Name', how='left', right_index=False)
            df = df.rename(columns={'geometry_y': 'geometry'})
            df = df.drop(columns=['geometry_x'])
            df = df.apply(self.get_growing_months, axis=1)
            df = gpd.GeoDataFrame(df, geometry='geometry', crs='EPSG:4326')
            df.to_parquet(self.save_dir / 's2_tiles_with_growing_months.parquet')
            print(f'Saved to {self.save_dir / "s2_tiles_with_growing_months.parquet"}')

    def get_growing_months(self, row):
        if np.isnan(row['mask']):
            if row.geometry.centroid.y > 0:
                row['growing_months'] = [5,6,7,8,9]
            else:
                row['growing_months'] = [12,1,2,3,4,5]
            return row
        
        valid_months = [ i for i in range(1,13) if row[f'freq_month{i}'] > 0.5]
        if len(valid_months) >= 5:
            row['growing_months'] = valid_months
        else:
            # get the top 5 months with highest frequency
            row['growing_months'] = sorted(range(1,13), key=lambda i: row[f'freq_month{i}'])[-5:]
        return row


    def check_tile_growing_season(self, tile_grid, data, tile_name):
        tile = tile_grid[tile_grid['Name'] == tile_name]
        geometry = transform(drop_z, tile.geometry.item())
        geojson_poly = mapping(geometry)
        ee_geometry = ee.Geometry(geojson_poly)
        stats = data.reduceRegion(
                reducer=ee.Reducer.toList(),
                geometry=ee_geometry,
                scale=500,  # VIIRS native resolution
                maxPixels=1e9
            )
        stats = stats.getInfo()
        for cycle in range(1,3):
            onset =np.array(stats[f'Onset_{cycle}']) - self.offset
            end = np.array(stats[f'End_{cycle}']) - self.offset
            plt.figure()
            plt.hist(onset, bins=10, histtype='step', linewidth=2, label='onset')
            plt.hist(end, bins=10, histtype='step', linewidth=2, label='end')
            plt.legend()
            plt.xlabel('DOY')
            plt.ylabel('Count')
            plt.title(f'{tile_name} Growing Season {cycle}, total count: {len(onset)}')
            plt.savefig(f'output/{tile_name}_growing_season{cycle}.png')
    
    @dask.delayed
    def agg_growing_season_per_tile_from_viirs(self, tile, data):
        geometry = transform(drop_z, tile.geometry)
        geojson_poly = mapping(geometry)
        ee_geometry = ee.Geometry(geojson_poly)
        stats = data.reduceRegion(
                reducer=ee.Reducer.mode(),
                geometry=ee_geometry,
                scale=500,  # VIIRS native resolution
                maxPixels=1e9
            )
        stats = stats.getInfo()
        if not stats['Onset_1'] or not stats['End_1']:
            print(f'No growing season found for {tile["Name"]}')
            print(tile.geometry.bounds)
            if tile.geometry.bounds[1] < 0:
                tile['onset'] = self.default_onset_south
                tile['end'] = self.default_end_south
            else:
                tile['onset'] = self.default_onset_north
                tile['end'] = self.default_end_north
            return tile
        onset =np.array(stats[f'Onset_1']) - self.offset
        end = np.array(stats[f'End_1']) - self.offset
        onset = onset.round()
        end = end.round()

        tile['onset'] = pd.to_datetime(onset, unit='D', origin=self.base_date)
        
        if onset > end:
            tile['end'] = pd.to_datetime(end+365, unit='D', origin=self.base_date)
        else:
            tile['end'] = pd.to_datetime(end, unit='D', origin=self.base_date)
        return tile

    def cal_cycle2_occurence(self):
        # Load VIIRS land surface phenology data for 2020
        viirs = ee.Image('NOAA/VIIRS/001/VNP22Q2/2020_01_01').select(
            ['Onset_Greenness_Increase_1', 'Onset_Greenness_Increase_2','Onset_Greenness_Minimum_1', 'Onset_Greenness_Minimum_2']
        )
        for cycle in range(1,3):
            onset = viirs.select(f'Onset_Greenness_Increase_{cycle}')
            end = viirs.select(f'Onset_Greenness_Minimum_{cycle}')
            onset = onset.updateMask(end)
            end = end.updateMask(onset)
            viirs = viirs.addBands(onset.rename(f'Onset_{cycle}'))
            viirs = viirs.addBands(end.rename(f'End_{cycle}'))

        viirs = viirs.select(['Onset_1', 'End_1', 'Onset_2', 'End_2'])

        # end = ee.Image('NOAA/VIIRS/001/VNP22Q2/2020_01_01').select(
        #     ['Onset_Greenness_Minimum_1', 'Onset_Greenness_Minimum_2']
        # )

        # Load Sentinel-2 MGRS tile grid
        s2_shp = '~/data/GEDI/Sentinel-2-Shapefile-Index-master/sentinel_2_index_shapefile.shp'
        tile_grid = get_s2_tiles_by_landmass(s2_shp)
        # tasks = [ agg_growing_season(tile, viirs) for idx, tile in tile_grid.iterrows()]

        # results = dask.compute(*tasks)
        # df = pd.DataFrame(results)
        # df = gpd.GeoDataFrame(df, geometry='geometry')
        # df['growth_period_len'] = (df['end'] - df['onset']).days
        # df.to_parquet('~/data/gvs/S2_tiles_with_growing_season.parquet')

        tasks = [ self.cal_cycle2_occurence_per_tile(tile, viirs) for idx, tile in tile_grid.iterrows()]
        results = dask.compute(*tasks)
        df = pd.DataFrame(results)
        df = gpd.GeoDataFrame(df, geometry='geometry')
        df['cycle2_ratio'] = df['count_cycle2'] / (df['count_cycle1'] + df['count_cycle2'])
        df.to_parquet('~/data/gvs/S2_tiles_with_growing_season.parquet')

    @dask.delayed
    def cal_cycle2_occurence_per_tile(tile, data):
        geometry = transform(drop_z, tile.geometry)
        geojson_poly = mapping(geometry)
        ee_geometry = ee.Geometry(geojson_poly)
        stats = data.reduceRegion(
                reducer=ee.Reducer.toList(),
                geometry=ee_geometry,
                scale=500,  # VIIRS native resolution
                maxPixels=1e9
            )
        stats = stats.getInfo()
        tile['count_cycle1'] = len(stats[f'Onset_1'])
        tile['count_cycle2'] = len(stats[f'Onset_2'])
        return tile


@dataclass
class MyConfig:
    asset_id: str = 'projects/gisproject-1/assets/sentinel_2_shapefile_with_growing_month_counts'
    year: int = 2019
    save_dir: str = '~/data/gvs/'
    version: str = 'v1'

cs = ConfigStore.instance()
cs.store(name="my_config", node=MyConfig)


@hydra.main(config_name="my_config", version_base='1.2')
def main(cfg: DictConfig) -> None:
    from dask.distributed import Client, LocalCluster
    from dask import config
    cluster = LocalCluster()
    client = Client(cluster)
    t0 = time.time()
    config.set({'distributed.scheduler.allowed-failures': 10})
    growing_season = GrowingSeason(**cfg)
    growing_season.get_growing_months_per_tile()

    t1 = time.time()
    print(f"Time taken: {t1-t0}")

if __name__ == '__main__':
    main()

