import time
import ee
import requests
from pathlib import Path
import pandas as pd
import xarray as xr
from hydra.core.config_store import ConfigStore
from dataclasses import dataclass
import hydra
from omegaconf import DictConfig
import dask
import geopandas as gpd
import shutil
import rasterio

import logging

# Silence ONLY this specific GDAL warning:
logging.getLogger("rasterio").setLevel(logging.ERROR)
logging.getLogger('rasterio._env').setLevel(logging.ERROR)

from .utils import authenticate
authenticate()

class GEEDownloader:
    """
    A class for downloading GEDI data from Google Earth Engine.
    
    """
    def __init__(self, s2_grid_file: str=None, output_dir: str=None, patch_size: int=15, out_res: int=10, **kwargs):
        self.buffer_size = (patch_size * out_res) // 2  # in meters
        self.patch_size = patch_size
        self.out_res = out_res
        self.s2_grid_file = Path(s2_grid_file).expanduser()
        self.s2_grid = gpd.read_parquet(self.s2_grid_file, columns=['Name', 'geometry']) # TODO: growing month?
        self.s2_grid = self.s2_grid.set_crs(epsg=4326)
        self.output_dir = Path(output_dir).expanduser()
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.encoding = {
            'data': {
                "zlib": True,
                "complevel": 7,
                "fletcher32": True,
                "chunksizes": (1, 64, self.patch_size, self.patch_size)
            }
        }

    def assign_s2_grid_cells(self, locations: gpd.GeoDataFrame):
        '''
        Assign S2 grid cells to given locations.
        '''
        locations = locations.sjoin(self.s2_grid, how='left', predicate='intersects')
        locations = locations.drop(columns=['index_right'])
        locations = locations.drop_duplicates(subset=['rowid'])
        return locations

    def sample_locations(self, asset_url: str, asset_type: str='', locations: str=None, merge_as_h5: bool=False):

        ''' 
        Sample patches from given locations. point by point
        @2025-11-18:14s/100 points, patch
        
        '''
        band_list = [f'A{str(i).zfill(2)}' for i in range(64)]
        patch_dir = self.output_dir / f'patches_{self.patch_size}x{self.patch_size}'
        patch_dir.mkdir(exist_ok=True, parents=True)
        if locations is not None:
            locations = Path(locations).expanduser()
            locations = pd.read_csv(locations)
            locations = gpd.GeoDataFrame(locations, geometry=gpd.points_from_xy(locations.Longitude, locations.Latitude), crs="EPSG:4326")
        if asset_type == 'ImageCollection':
            asset = ee.ImageCollection(asset_url)
            asset = asset.filterDate('2020-01-01', '2020-12-31')
        
        if locations is not None:
            locations = self.assign_s2_grid_cells(locations)
        
        @dask.delayed
        def sample_a_tile(row: gpd.GeoSeries):

            # epsg = get_epsg_from_tile(row['Name'])
            point = ee.Geometry.Point(row.geometry.x, row.geometry.y)
            roi = point.buffer(self.buffer_size) # in meters
            img = asset.filterBounds(point).first()
            crs = img.projection().crs()
            
            url = img.multiply(1e4).toInt16().getDownloadURL({
                'region': roi,
                'dimensions': f'{self.patch_size}x{self.patch_size}',
                'crs': crs,
                'format': 'GEO_TIFF'
            })
            # Handle downloading the actual pixels.
            r = requests.get(url, stream=True)
            if r.status_code != 200:
                r.raise_for_status()
                
            with open(patch_dir / f'row_{row.rowid}.tif', 'wb') as out_file:
                shutil.copyfileobj(r.raw, out_file)
                
            props = row._asdict()
            del props['geometry']
            del props['Index']
            with rasterio.open(patch_dir / f'row_{row.rowid}.tif', 'r+') as src:
                for i in range(1, src.count + 1):
                    if i <= len(band_list):
                        src.set_band_description(i, band_list[i - 1])
                src.update_tags(**props)
                
            print(f'row {row.rowid} done!')
        
        das = []
        # locations = locations.iloc[:100]
        for row in locations.itertuples():
            if (patch_dir / f'row_{row.rowid}.tif').exists():
                continue
            da = sample_a_tile(row)
            das.append(da)
        dask.compute(*das)
        
        if merge_as_h5:
            # NOTE: 52s to open, concat and save 100 patches, 6MB in RAM, 1.3M on disk
            keep_cols = locations.columns
            keep_cols = [col for col in keep_cols if col not in ['geometry', 'Index']]
            x_coords = range(self.patch_size)
            y_coords = range(self.patch_size)
            das = []
            with rasterio.Env(GDAL_ERROR_LEVEL="CPLE_Fatal"):
                for row in locations.itertuples():
                    ds = xr.open_dataset(patch_dir / f'row_{row.rowid}.tif')
                    da = ds.band_data.assign_coords({'x': x_coords, 'y': y_coords})
                    da.attrs = {} 
                    das.append(da)
                # das = [xr.open_dataset(patch_dir / f'row_{row.rowid}.tif') for row in locations[:10].itertuples()]
            das = xr.concat(das, dim='loc', join='override', compat='override', coords='minimal')
            das = das.assign_coords({col: (['loc'], locations[col]) for col in keep_cols})
            das.name ='data'
            das.to_netcdf(self.output_dir / 'alphaearth_embeddings.h5', format='NETCDF4', engine='h5netcdf', encoding=self.encoding, mode='w')
        return
    
    def sample_locations_xee(self, asset_url: str, asset_type: str='', locations: str=None, merge_as_h5: bool=False):

        ''' 
        # ----------------
        #  Old, using xee
        # ----------------
        # using xee is extremely slow (~2s/point), not optimal for point by point case.
        Sample patches from given locations.
        '''
        patch_dir = self.output_dir / f'patches_{self.patch_size}x{self.patch_size}'
        patch_dir.mkdir(exist_ok=True, parents=True)
        if locations is not None:
            locations = Path(locations).expanduser()
            locations = pd.read_csv(locations)
            locations = gpd.GeoDataFrame(locations, geometry=gpd.points_from_xy(locations.Longitude, locations.Latitude), crs="EPSG:4326")
        if asset_type == 'ImageCollection':
            asset = ee.ImageCollection(asset_url)
            asset = asset.filterDate('2020-01-01', '2020-12-31')
        
        if locations is not None:
            locations = self.assign_s2_grid_cells(locations)
        
        @dask.delayed
        def sample_a_tile(row: gpd.GeoSeries):

            # epsg = get_epsg_from_tile(row['Name'])
            point = ee.Geometry.Point(row.geometry.x, row.geometry.y)
            img = asset.filterBounds(point)
            epsg = img.projection().crs()
            
            
            gdf = gpd.GeoDataFrame(
                [row],
                geometry=[row.geometry],
                crs=locations.crs  # or explicitly "EPSG:4326"
            ).to_crs(epsg=epsg.replace('EPSG:', ''))
            # bounds = buffer_and_snap_bounds(gdf.geometry, self.buffer_size, self.out_res)
            bounds = gdf.geometry.buffer(self.buffer_size).bounds
            bounds = bounds.astype('float32').values.tolist()[0]
            proj = ee.Projection(epsg)
            rect = ee.Geometry.Rectangle(coords=bounds, proj=proj, geodesic=False)
            ds = xr.open_dataset(img, geometry=rect, crs=epsg, scale=self.out_res)
            da = ds.to_array(dim="band")
            da = da.sel(X=slice(bounds[0], bounds[2]), Y=slice(bounds[1], bounds[3]))
            assert da.shape[2] == self.patch_size and da.shape[3] == self.patch_size, 'Patch size mismatch'
            attrs = row._asdict()
            del attrs['geometry']
            del attrs['Index']
            da = da.assign_attrs(attrs)
            print('row sampled, row id: ', row.Index)
            return da
        
        
        das = []
        for row in locations.itertuples():
            if (patch_dir / f'row_{row.row_id}.tif').exists():
                continue
            da = sample_a_tile(row)
            das.append(da)
        das = dask.compute(*das)
        das = xr.concat(das, dim='time')
        das.to_netcdf(self.output_dir / 'alphaearth_embeddings.h5', format='NETCDF4', engine='h5netcdf', encoding=self.encoding, mode='w')
        return 

@dataclass
class Config:
    asset_url: str = 'GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL'
    asset_type: str = 'ImageCollection'
    s2_grid_file: str = '~/data/gvs/s2_tiles_with_growing_months.parquet'
    locations: str = '~/data/gvs/downstream_task_data/naturalness/reference_data_set_updated.csv'
    output_dir: str = '~/data/gvs/downstream_task_data/alphaearth_embeddings'
    merge_as_h5: bool = True
    
cs = ConfigStore.instance()
cs.store(name="config", node=Config)

@hydra.main(config_name="config", version_base='1.2')
def main(cfg: DictConfig):
    downloader = GEEDownloader(**cfg)
    t0 = time.time()
    downloader.sample_locations(asset_url=cfg.asset_url, asset_type=cfg.asset_type, locations=cfg.locations, merge_as_h5=cfg.merge_as_h5)
    t1 = time.time()
    print(f'Time taken: {t1 - t0} seconds')
if __name__ == '__main__':
    main()


            
        