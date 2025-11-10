import os
import json
from pathlib import Path
import ee
import pystac
import planetary_computer
import backoff
import pyproj
import ctypes
import numpy as np
import geopandas as gpd
from typing import Union, List
from stackstac.raster_spec import RasterSpec
from rasterio.errors import RasterioIOError
from pyproj import Transformer
import pystac_client
import adlfs
import pandas as pd
import requests
from io import StringIO
from shapely.geometry import shape
import json
from dask.utils import natural_sort_key

from ._const import STAC_ITEM_KEYS
from utils._stackstac import stack


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
    return item #planetary_computer.sign_inplace(item)  not sign to update meta table
    # TO CHCEK: the token generated seems to be only valid for 1 hour


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
    bounds['maxx'] = np.ceil(bounds['maxx'] / res + 1e-12) * res # for point with coords 0
    bounds['maxy'] = np.ceil(bounds['maxy'] / res + 1e-12) * res
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

def backoff_hdlr(details):
    print("Backing off {wait:0.1f} seconds after {tries} tries "
          "calling function {target} with args {args} and kwargs "
          "{kwargs}".format(**details))


def fatal_code(e):
    return 400 <= e.response.status_code < 500

@backoff.on_exception(backoff.expo,
                      RasterioIOError,
                      max_time=300,
                      on_backoff=backoff_hdlr,
                    #   giveup=fatal_code,
)
def get_patch(items,
              assets: Union[str, List[str]] = None,
              resolution: int = 10,
              fill_value: Union[int, float] = 0,
              band_coords: bool = False,
              properties: bool = False,
              dtype: str = 'uint16',
              xy_coords: bool = 'topleft',
              snap_bounds=False,
              rescale=False,
              **kwargs):
    default_args = dict(assets=assets,
                        resolution=resolution,
                        fill_value=fill_value,
                        band_coords=band_coords,
                        properties=properties,
                        dtype=dtype,
                        xy_coords=xy_coords,
                        snap_bounds=snap_bounds,
                        rescale=rescale)
    #TODO: how to check if the error is caused by the expired token?
    patch = stack(items, **default_args, **kwargs)
    return patch

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

def utm_to_wgs84(bounds, utm_epsg=32632, wgs84_epsg=4326):
    """
    Convert bounds from UTM (EPSG:32632) to WGS84 (EPSG:4326).
    
    Parameters:
    - bounds: Tuple of (min_x, min_y, max_x, max_y) in UTM coordinates.
    - utm_epsg: EPSG code for the input UTM projection (default: 32632).
    - wgs84_epsg: EPSG code for WGS84 (default: 4326).
    
    Returns:
    - Tuple of (min_lon, min_lat, max_lon, max_lat) in WGS84.
    """
    transformer = Transformer.from_crs(utm_epsg, wgs84_epsg, always_xy=True)
    
    min_x, min_y, max_x, max_y = bounds
    
    # Transform all corners
    bottom_left = transformer.transform(min_x, min_y)
    bottom_right = transformer.transform(max_x, min_y)
    top_left = transformer.transform(min_x, max_y)
    top_right = transformer.transform(max_x, max_y)
    
    # Calculate new bounds
    min_lon = min(bottom_left[0], bottom_right[0], top_left[0], top_right[0])
    min_lat = min(bottom_left[1], bottom_right[1], top_left[1], top_right[1])
    max_lon = max(bottom_left[0], bottom_right[0], top_left[0], top_right[0])
    max_lat = max(bottom_left[1], bottom_right[1], top_left[1], top_right[1])
    
    return min_lon, min_lat, max_lon, max_lat


def build_parquet_file_table(collection):
    """
    Sentinel-2 file is partitioned by week. 
    Builds a table with three columns: file name, start date, and end date.
    File names of Sentinel-2 geoparquet files are retrieved from Azure Blob Storage.
    Each row is a STAC geoparquet item.

    Parameters:
    ------------
    * collection: str: The name of the collection.

    Returns:
    ------------
    * pandas.DataFrame: A DataFrame containing the file names, start dates, and end dates.
    """
    api = pystac_client.Client.open(stac_endpoint, modifier=planetary_computer.sign_inplace)
    asset = api.get_collection(collection).assets['geoparquet-items']
    fs = adlfs.AzureBlobFileSystem(
        **asset.extra_fields["table:storage_options"]).ls("items/sentinel-2-l2a.parquet")
    fs_df = pd.DataFrame(fs, columns=['fname'])
    date_range = fs_df.fname.str.findall(r'\d{4}-\d{2}-\d{2}')
    fs_df['start'] = pd.to_datetime(date_range.str[0])
    fs_df['end'] = pd.to_datetime(date_range.str[1])
    fs_df['fname'] = 'abfs://' + fs_df['fname']
    return fs_df

def filter_parquet_files(fs_df, start, end):
    """
    Filter Sentinel-2 STAC geoparquet items based on the given start and end timestamps.

    Parameters:
    ------------
    * start (datetime): The start timestamp for filtering.
    * end (datetime): The end timestamp for filtering.

    Returns:
    ------------
    * list: A list of filtered file names.
    """
    filtered = fs_df[(fs_df['start'] < end) & (fs_df['end'] > start)]
    return filtered['fname'].to_list()

def shapely_to_geojson(geom: gpd.GeoSeries):
    """
    Convert a shapely geometry to a GeoJSON geometry in dict format
    so that it can be used for ee.FeatureCollection.filterBounds(geom)
    """
    geom = ee.Geometry.BBox(*geom.bounds).toGeoJSON()
    last_coords = geom['coordinates'][0][0].copy()
    geom['coordinates'][0].append(last_coords)
    return geom


def ee_fc_to_gpd(fc: ee.FeatureCollection):
    """
    Convert an Earth Engine feature collection to a GeoPandas GeoDataFrame.
    """
    download_id = ee.data.getTableDownloadId({'table': fc, 'fileFormat': 'csv'})
    res = requests.get(ee.data.makeTableDownloadUrl(download_id))
    if res.status_code == 200:
        data = StringIO(res.content.decode('utf-8'))
        if data.getvalue().strip() == '':
            return None
        df = pd.read_csv(data)
        df['.geo'] = df['.geo'].apply(lambda x: shape(json.loads(x)))
        df = df.rename(columns={'.geo': 'geometry'})
        df = gpd.GeoDataFrame(df, geometry='geometry')
        df = df.set_crs(epsg=4326)
        return df
    else:
        raise requests.HTTPError(f'Failed to download {fc.getInfo()}')
    
def gdf_to_gpkg(gdf: gpd.GeoDataFrame, file: str):
    """
    Convert a GeoPandas GeoDataFrame to a GeoPackage file.
    """
    gdf.to_file(file, driver='GPKG')
    print(f'Saved {file}')
    
    
def check_unfinished_files(input_files: Path, output_dir: Path, check_exists=True, sort=True, output_format=None):
    """
    Check if the input files have been processed according to the output files.
    """
    input_files = list(input_files)
    output_format = output_format or input_files[0].name.split('.')[-1]
    if sort:
        input_files = [str(f) for f in input_files]
        input_files = sorted(input_files, key=natural_sort_key)
        input_files = [Path(f) for f in input_files]
        
    unfinished_files = []
    for file in input_files:
        output_file = output_dir / f'{file.stem}.{output_format}'
        if check_exists and output_file.exists():
            continue
        unfinished_files.append(file)
    return unfinished_files