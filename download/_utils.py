import os
import json
import ee
import pystac
import planetary_computer
import pyproj
import ctypes
import numpy as np
import geopandas as gpd
from stackstac.raster_spec import RasterSpec
from _const import STAC_ITEM_KEYS

stac_endpoint = 'https://planetarycomputer.microsoft.com/api/stac/v1'

def authenticate():
    """
    Authenticates the Earth Engine service using the key file specified in the env variable.

    If the environment variable 'KEY_FILE' is set, it will be used as the path to the key file.
    Otherwise, the default path 'keys/private-key.json' will be used.

    """
    key_file = os.environ.get('KEY_FILE')
    key_file = key_file or 'keys/private-key.json'
    print('Authenticating from', key_file)
    key = json.load(open(key_file))
    credentials = ee.ServiceAccountCredentials(key['client_email'], key_file)
    ee.Initialize(credentials, url='https://earthengine-highvolume.googleapis.com')
    print('Authenticated Earth Engine successfully')

def get_tile_by_id(tile_id):
    """
    Get the sentinel-2 scene STAC item by tile id.
    """
    url = f'{stac_endpoint}/collections/sentinel-2-l2a/items/{tile_id}'
    item = pystac.Item.from_file(url)
    return planetary_computer.sign_inplace(item)  # TO CHCEK: the token generated seems to be only valid for 1 hour


def get_most_common_epsg(items):
    """
    Get the most common epsg code from a list of STAC items.
    """
    epsgs = [item.properties['proj:epsg'] for item in items]
    return max(set(epsgs), key=epsgs.count)

def reproject_bounds(raster_spec: RasterSpec, crs_to='EPSG:4326'):
    """
    Reprojects the bounds of a stackstac.raster_spec.RasterSpec object to a specified coordinate reference system (CRS).

    Parameters
    -----------
    * raster_spec (RasterSpec): The RasterSpec object containing the bounds to be reprojected.
    * crs_to (str, optional): The target CRS to reproject the bounds to. Defaults to 'EPSG:4326'.

    Returns
    -----------
    * list: A list containing the reprojected bounds in the order [minx, miny, maxx, maxy].
    """
    transformer = pyproj.Transformer.from_crs(f'EPSG:{raster_spec.epsg}', crs_to, always_xy=True)
    # Transform the bounds
    minx, miny = transformer.transform(raster_spec.bounds[0], raster_spec.bounds[1])
    maxx, maxy = transformer.transform(raster_spec.bounds[2], raster_spec.bounds[3])
    return [minx, miny, maxx, maxy]


def buffer_and_snap_bounds(geom: gpd.GeoSeries, buffer_size:int, res:int=10):
    """
    Buffers the given geometry by the specified buffer size and snaps the resulting bounds to sentinel-2 pixel grid based on the specified resolution.

    Parameters
    -----------
    * geom (gpd.GeoSeries): The input geometry to be buffered.
    * buffer_size (int): in meters, the buffered width & height will be (2 * buffer_size) + res
    * res (int): resolution in meters

    Returns
    -----------
    * bounds (pandas.DataFrame): The snapped bounds of the buffered geometry, with 'minx', 'miny', 'maxx', and 'maxy' columns.

    """
    bounds = geom.buffer(buffer_size).bounds
    # snap to grid
    bounds['minx'] = np.floor(bounds['minx'] / res) * res
    bounds['miny'] = np.floor(bounds['miny'] / res) * res
    bounds['maxx'] = np.ceil(bounds['maxx'] / res) * res
    bounds['maxy'] = np.ceil(bounds['maxy'] / res) * res
    return bounds.astype('int')

def get_total_bounds(geom:gpd.GeoSeries):
    """
    Get the bbox of a GeoSeries.
    Returns
    -----------
    * tuple: The bounding box in the order (minx, miny, maxx, maxy).
    """
    return (
        geom['minx'].min(),
        geom['miny'].min(),
        geom['maxx'].max(),
        geom['maxy'].max())

def resign_items(items):
    """
    Resigns a list of STAC items using the planetary_computer.sign() function.

    Returns
    -----------
    * list: A list of resigned items.
    """
    res = []
    for item in items:
        item = planetary_computer.sign(item)
        res.append(item)
    return res


def row_to_stac_item(df, props):
    """
    Convert geoparquet items to STAC items.

    Parameters
    -----------
    * df (geopandas.DataFrame): The DataFrame containing the geoparquet items(rows) to convert.
    * props (list): A list of column names to include as properties in the STAC items.

    Returns
    -----------
    * list: A list of STAC items converted from the DataFrame rows.
    """
    items = []
    for idx, row in df.iterrows():
        row['id'] = idx
        item = row[STAC_ITEM_KEYS].to_dict()
        item['geometry'] = item['geometry'].__geo_interface__
        item['properties'] = row[props].to_dict()
        item['type'] = 'Feature'
        item['stac_version'] = '1.0.0'
        item = planetary_computer.sign_inplace(pystac.Item.from_dict(item))
        items.append(item)
    return items


def trim_memory() -> int:
    """
    Trims the memory allocated by the C library to the minimum required.
    This function is used to tackle the problem of unmanaged memory high issue when using Dask.
    It calls the `malloc_trim` function from the C library to release unused memory back to the system.
    
    Returns
    -----------
    * int: The amount of memory (in bytes) that was trimmed.
    """
    libc = ctypes.CDLL("libc.so.6")
    return libc.malloc_trim(0)
