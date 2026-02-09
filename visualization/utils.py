import numpy as np
import pystac
from pathlib import Path
import geopandas as gpd
from shapely.geometry import Point, box
from pyproj import CRS
import stackstac
import xarray as xr
from download.core.utils import get_epsg_from_tile

def get_tile_id_by_coords(coords, s2_grid: gpd.GeoDataFrame = None, year: int = 2020):
    '''
    Get the tile id and points by the coordinates
    Args:
        coords: coordinates of the points, a list of tuples (lat, lon)
        s2_grid: geopandas dataframe of the s2 grid
        year: year
    Returns:
        matched: geopandas dataframe of the matched tile
        points: geopandas dataframe of the points
    '''
    coords = np.array(coords)
    points = gpd.GeoDataFrame(geometry=gpd.points_from_xy(coords[:, 1], coords[:, 0]), crs="EPSG:4326")
    bounds = points.total_bounds
    matched = s2_grid.cx[bounds[0]:bounds[2], bounds[1]:bounds[3]] # minx:maxx, miny:maxy
    matched = matched.sort_values(f'type_{year}', ascending=False).iloc[0]
    return matched, points


def get_patch_by_coords(
        coords, s2_grid: gpd.GeoDataFrame = None, year: int = 2020, buffer_m=100, q_idx=1, prediction_dir: Path = None, stac_collection: Path = None,
        input_image_dir: str = None, best_image_time: str = None):
    """
    Get a small patch from the global prediction.
    1. Get the tile and bounds based on lat, lon and the s2_grid
    2. Get the patch prediction with buffer_km around lat, lon, if buffer_km is None, return the whole tile
    ------
    Args:
        coords: coordinates of the points, a list of tuples (x, y)
        s2_grid: geopandas dataframe of the s2 grid
        year: year
        buffer_km: buffer around the lat, lon, only buffer if there is only one point
        q_idx: quality index
        prediction_dir: root directory of all tiles' predictions
        input_image_dir: directory of the input images
    """
    # Get the tile and bounds
    matched, points = get_tile_id_by_coords(coords, s2_grid, year)
    tile = s2_grid[s2_grid['Name'] == matched['Name']].iloc[0]
    tile_id = matched['Name']
    epsg = get_epsg_from_tile(tile_id)
    tile_crs = CRS.from_epsg(epsg)  # example: Sentinel-2 tile UTM zone 33N
    points_utm = points.to_crs(tile_crs)

    if buffer_m is not None: # only buffer if there is only one point
        # Create 3x3 km buffer ---
        # A circular buffer of radius 1.5 km creates an area roughly 3 km wide.
        # Buffer the bounds as a single polygon instead of individual point buffers
        bounds = points_utm.total_bounds  # [minx, miny, maxx, maxy]
        minx, miny, maxx, maxy = bounds
        # Create a shapely box from bounds
        bounds_box = box(minx, miny, maxx, maxy)
        # Buffer the box by buffer_m/2
        buffer_m = buffer_m / 2
        buffered_box = bounds_box.buffer(buffer_m)
        bounds = buffered_box.bounds  # returns (minx, miny, maxx, maxy)
        bounds = list(bounds)
    else:
        bounds = points_utm.total_bounds.tolist()
    
    tif_files = list(prediction_dir.glob(f'{tile_id}/*_Q{q_idx}*.tif'))
    tif_files = sorted(tif_files, key=lambda x: int(x.stem.split('_')[0][2:]))
    # Load STAC item from local STAC collection, href might be outdated, but metadata like bbox, shape, transform, gsd are still valid
    item = pystac.Item.from_file( str(stac_collection / tile_id / f'{tile_id}.json'))
    for f in tif_files:
        asset_name = f.stem.split('_')[0]
        item.assets[asset_name].href = str(f)

    image = stackstac.stack(
        item,
        epsg=epsg,
        resolution=10,
        bounds=bounds,  # crop
    )
    if input_image_dir is not None:
        ds = xr.open_zarr(input_image_dir/tile_id)
        ds = ds.sel(x=image.x.data, y=image.y.data)
        if best_image_time is not None:
            ds = ds.sel(time=~ds.get_index("time").duplicated())
            ds = ds.sel(time=f'{best_image_time}T00:00:00', method='nearest')
            print('Using the best image time: ', best_image_time)
            print('Found image time: ', ds.time.values)
        else:
            ds = ds.isel(time=0)
        import ipdb; ipdb.set_trace()
        return image.squeeze(), ds.s2.sel(band=['B04', 'B03', 'B02']), points_utm # return the RGB image
    return image.squeeze()
