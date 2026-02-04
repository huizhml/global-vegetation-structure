import numpy as np
import pandas as pd
import xarray as xr
import time
import lexcube
import matplotlib.pyplot as plt
import pystac
from pathlib import Path
import geopandas as gpd
from shapely.geometry import Point
from pyproj import CRS, Transformer
import stackstac
import datetime

def get_epsg_from_tile(tile_name):
    zone = int(tile_name[:2])
    band = tile_name[2]
    hemisphere = 'south' if band <= 'M' else 'north'
    epsg = 32700 + zone if hemisphere == 'south' else 32600 + zone
    return epsg


def get_patch_by_latlon(lat, lon, s2_grid: gpd.GeoDataFrame = None, year: int = 2020, buffer_km=3, q_idx=1):
    # Get the tile and bounds
    points = gpd.GeoDataFrame(geometry=[Point(lon, lat)], crs="EPSG:4326")
    matched = gpd.sjoin(points, s2_grid, predicate="within", how="left")
    matched = matched.sort_values(f'type_{year}', ascending=False).iloc[0]
    tile = s2_grid[s2_grid['Name'] == matched['Name']].iloc[0]
    tile_id = matched['Name']
    epsg = get_epsg_from_tile(tile_id)

    tile_crs = CRS.from_epsg(epsg)  # example: Sentinel-2 tile UTM zone 33N
    points_utm = points.to_crs(tile_crs)

    # --- Create 3x3 km buffer ---
    # A circular buffer of radius 1.5 km creates an area roughly 3 km wide.
    buffer_m = (buffer_km / 2) * 1000
    points_buffer = points_utm.buffer(buffer_m)

    
    pred_dir = Path(f'~/data/gvs/deploy/predictions_{year}/{tile_id}_cog').expanduser() # TODO: chose the latest predictions
    tif_files = list(pred_dir.glob(f'*_Q{q_idx}.cog.tif'))
    # Create STAC item
    item = pystac.Item(id=tile_id, geometry=tile.geometry, bbox=tile.geometry.bounds, datetime=datetime.datetime(year, 1, 1, 0, 0, 0), properties={})
    for f in tif_files:
        item.add_asset(f.stem, pystac.Asset(href=str(f), media_type=pystac.MediaType.COG))

    image = stackstac.stack(
        item,
        epsg=epsg,
        resolution=10,
        bounds=points_buffer.total_bounds.tolist(),  # crop
    )
    return image.squeeze()
    
    
s2_grid_file = '~/data/gvs/deploy/deploy_status.parquet'
s2_grid = gpd.read_parquet(s2_grid_file).to_crs("EPSG:4326")
year = 2020
lat, lon = -2.346071, 114.03641  
patch = get_patch_by_latlon(lat, lon, s2_grid, year, buffer_km=3, q_idx=1)
wd = lexcube.Cube3DWidget(patch, cmap='inferno', vmin=0, vmax=500, isometric_mode=True)
wd.plot()
import ipdb; ipdb.set_trace()
wd.savefig('output/patch.png', include_ui=True, dpi_scale=2)