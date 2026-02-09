import geopandas as gpd
from pathlib import Path
from dataclasses import dataclass
import hydra
from hydra.core.config_store import ConfigStore
import pystac_client
import planetary_computer
from pystac_client.stac_api_io import StacApiIO
import xarray as xr
import stackstac
import numpy as np
import rasterio
import dask
import time


stac_api_io = StacApiIO()
stac_endpoint = 'https://planetarycomputer.microsoft.com/api/stac/v1'



def mask_snow_water_preds(year: int = 2020, tile_id: str = None, s2_grid_file: str = None, save_dir: str = None, translate: bool = False):
    save_dir = Path(save_dir).expanduser()
    save_dir = save_dir / f'{tile_id}'
    save_dir.mkdir(parents=True, exist_ok=True)
    s2_grid = gpd.read_parquet(s2_grid_file)
    tile = s2_grid[s2_grid['Name'] == tile_id]
    if len(tile) == 0:
        raise ValueError(f'Tile {tile_id} not found in S2 grid file {s2_grid_file}')
    tile = tile.iloc[0]

    pred_dir = Path(f'~/data/gvs/deploy/predictions_{year}/{tile_id}_cog').expanduser()
    pred_gtiff_dir = Path(f'~/data/gvs/deploy/predictions_GTiff_{year}/{tile_id}_GTiff').expanduser()
    # If the tile is repredicted, use the GTiff directory
    if pred_gtiff_dir.exists():
        count = len(list(pred_gtiff_dir.glob('*_uncompressed.tif')))
        if count >= 303:
            pred_dir = pred_gtiff_dir
    pred_files = list(pred_dir.glob('*.tif'))
    # query esa world cover for the tile
    with rasterio.open(pred_files[0]) as src:
        epsg = src.crs.to_epsg()
        bounds = src.bounds # (west, south, east, north)
        width = src.width
        height = src.height
    bounds = (bounds.left, bounds.bottom, bounds.right, bounds.top)
    api = pystac_client.Client.open(stac_endpoint, modifier=planetary_computer.sign_inplace, stac_io=stac_api_io)
    search = api.search(collections='esa-worldcover', bbox=tile.geometry.bounds, datetime=f'2021-01-01/2021-12-31')
    items = search.item_collection()
    if len(items) == 0:
        raise ValueError(f'No ESA World Cover items found for tile {tile_id}')
    wc_image = stackstac.stack(items, ['map'], bounds=bounds, epsg=epsg, resolution=10, dtype='uint16', fill_value=np.uint16(0),rescale=False)
    wc_image = wc_image.max(dim='time', skipna=True).squeeze()
    assert wc_image.shape == (height, width)
    water_snow_mask = wc_image.isin([0, 70, 80]) # nodata, snow and ice, water
    water_snow_mask = water_snow_mask.compute()
    
    def process_file(file):
        with rasterio.open(file) as src:
            nodata = src.nodata
            profile = src.profile
            data = src.read(1)
            data[water_snow_mask] = nodata
            filename = file.name.replace('_uncompressed.tif', '.tif')
            filename = filename.replace('.cog.tif', '.tif')
            with rasterio.open(save_dir / filename, 'w', **profile) as dst:
                dst.write(data, indexes=1)
        return True
    
    futures = [dask.delayed(process_file)(file) for file in pred_files]
    results = dask.compute(*futures)
    if all(results):
        print(f"Successfully masked {len(results)} files")
    else:
        print(f"Failed to mask {len(results)} files")

@dataclass
class MaskConfig:
    arctic_regions_file: str = '~/data/gvs/ne_10m_admin_0_countries/arctic_regions.gpkg'
    s2_grid_file: str = '~/data/gvs/s2_tiles_with_growing_months.parquet'
    year: int = 2020
    save_dir: str = f'~/data/gvs/deploy/predictions_gtiff_masked_{year}'
    tile_id: str = '11XMG'
    countries_file: str = '~/data/gvs/ne_10m_admin_0_countries/ne_10m_admin_0_countries.shp'
    ocean_file: str = '~/data/gvs/ne_10m_land/ne_10m_ocean.shp'
    task: str = 'mask_snow_water_preds'

    

cs = ConfigStore.instance()
cs.store(name='mask', node=MaskConfig)

@hydra.main(config_name='mask', version_base='1.2')
def main(cfg):
    print(cfg)
    t0 = time.time()
    if cfg.task == 'mask_snow_water_preds':
        mask_snow_water_preds(cfg.year, cfg.tile_id, cfg.s2_grid_file, cfg.save_dir)

if __name__ == '__main__':
    main()