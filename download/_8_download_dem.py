from pathlib import Path
from dataclasses import dataclass
import pandas as pd
from shapely.geometry import box, shape
import hydra
from hydra.core.config_store import ConfigStore
from retry import retry
import requests
import geopandas as gpd
import pystac_client
import planetary_computer
import pystac
import ee
from io import StringIO
import numpy as np
import dask
import xarray as xr
from dask.distributed import Client, LocalCluster
from download._dask_downloader import DaskDownloader
from download._utils import get_aux_df, row_to_stac_item, get_patch, get_epsg_from_tile, buffer_and_snap_bounds, get_total_bounds, check_unfinished_files
# from download._slope import slope
from xrspatial import slope

stac_endpoint = 'https://planetarycomputer.microsoft.com/api/stac/v1'
api = pystac_client.Client.open(stac_endpoint, modifier=planetary_computer.sign_inplace)

def set_fc_properties(row):
    geom = row.geometry
    row = row.drop('geometry')
    fc = ee.Feature(ee.Geometry.Point([geom.x, geom.y]), row.to_dict())
    fc = fc.set('index', row.name)
    return fc



class DEMDownloader(DaskDownloader):
    """
    A class for downloading DEM data for given locations from Google Earth Engine.
    
    Attributes:
        location_files: The file(s) containing the locations to download the DEM data.
        save_dir: The root directory to save the DEM data.
        n_parallel: The number of parallel tasks to download the DEM data.
        debug: Whether to print debug information.
    """
    def __init__(self, 
                 loc_dir: str=None,
                 n_parallel: int=100,  
                 rewrite: bool=False,
                 debug: bool=False, 
                 dem_meta_file: str=None,
                 **kwargs):
        super().__init__(n_parallel=n_parallel, max_retries=3, **kwargs)
        self.rewrite = rewrite
        self.debug = debug
        self.location_files = Path(loc_dir).expanduser().glob('*.parquet')
        self.dem_df = get_aux_df(dem_meta_file, 'cop-dem-glo-30', time_col='datetime')
        self.dem_df['datetime'] = self.dem_df['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S.%f')
    
    def add_slope(self, save_dir: Path):
        self.save_dir = save_dir
        self.save_dir.mkdir(exist_ok=True, parents=True)
        self.location_files = check_unfinished_files(self.location_files, self.save_dir, check_exists=True)
        tasks = []
        for file in self.location_files:
            tasks.append(self.download_file(file))
        self.schedule_tasks(delayed_tasks=tasks)
    
    
    def download(self):
        tasks = []
        for file in self.location_files:
            tasks.append(self.download_file(file))
        self.schedule_tasks(delayed_tasks=tasks)

    @dask.delayed
    @retry(requests.HTTPError, tries=10, delay=1)
    def download_file(self, file: Path):
        output_file = self.save_dir / f'{file.stem}.parquet'
        if output_file.exists() and not self.rewrite:
            print(f'{output_file} exists, skipping')
            return
        tile_id = file.stem
        epsg = get_epsg_from_tile(tile_id)
        loc_df = gpd.read_parquet(file)
        loc_ = loc_df.to_crs(epsg)
        total_bounds = buffer_and_snap_bounds(loc_.geometry, 60, 30) # buffer 2 pixels
        total_bounds = get_total_bounds(total_bounds)
        bbox = box(*loc_df.geometry.total_bounds)
        dem_df = self.dem_df[self.dem_df.geometry.intersects(bbox)]
        
        if not dem_df.geometry.union_all().covers(bbox):
            dem_items = api.search(collections=['nasadem'], intersects=bbox).item_collection()
            dem_items.asset_name = 'elevation'
        else:
            dem_items = row_to_stac_item(dem_df, ['datetime'])
            dem_items = pystac.item_collection.ItemCollection(dem_items)
            dem_items.asset_name = 'data'
        
        if len(dem_items.items) == 0:
            loc_df['slope'] = np.nan
            loc_df.to_parquet(output_file)
            print(f'{output_file} does not have DEM data, skipping')
            return
        
        dem_image = get_patch(dem_items.items, [dem_items.asset_name], resolution=30, epsg=epsg, bounds=total_bounds, dtype='float32', fill_value=np.float32(np.nan))
        dem_image = dem_image.max(dim='time', skipna=True).squeeze()
        if dem_image.shape[0] == 0:
            loc_df['slope'] = np.nan
            loc_df.to_parquet(output_file)
            print(f'{output_file} does not have DEM data, skipping')
            return
        dem_image.attrs['res'] = 30
        slope_da = slope(dem_image)
        target_x = xr.DataArray(loc_.geometry.x.values, dims="points")
        target_y = xr.DataArray(loc_.geometry.y.values, dims="points")
        sampled_values = slope_da.sel(x=target_x, y=target_y, method='nearest')
        sampled_values = sampled_values.compute()
        loc_df['slope'] = sampled_values.values
        loc_df.to_parquet(output_file)
        

@dataclass
class MyConfig:
    year: int=2020
    loc_dir: str=f'~/data/gvs/gedi/veg_sensitivity_gt0p95/{year}/subset_4k/original/'
    dem_meta_file: str=f'~/data/gvs/assets/dem/cop-dem-glo-30_items.parquet'
    n_parallel: int=40
    rewrite: bool=False
    debug: bool=False
    task: str='add_slope'
    
    
    
cs = ConfigStore.instance()
cs.store(name="my_config", node=MyConfig)


@hydra.main(config_name="my_config", version_base="1.2")
def main(cfg):
    cluster = LocalCluster()
    client = Client(cluster)
    
    downloader = DEMDownloader(
        loc_dir=cfg.loc_dir,
        dem_meta_file=cfg.dem_meta_file,
        n_parallel=cfg.n_parallel,
        rewrite=cfg.rewrite,
        debug=cfg.debug
    )
    if cfg.task == 'add_slope':
        loc_dir = Path(cfg.loc_dir).expanduser()
        save_dir = cfg.get('save_dir', str(loc_dir.parent / 'with_slope'))
        downloader.add_slope(save_dir)
    else:
        raise ValueError(f'Invalid task: {cfg.task}')

if __name__ == '__main__':
    main()
