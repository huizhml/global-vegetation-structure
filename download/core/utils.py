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
import dask.dataframe as dd
import requests
from io import StringIO
from shapely.geometry import shape
import json
import dask
from dask.utils import natural_sort_key
import dask_geopandas as dgp
import os
import sys
import logging
from datetime import datetime
from pyproj import Transformer
import torch
import numpy as np
from multiprocessing import shared_memory
from typing import Union
from pathlib import Path
import geopandas as gpd
from dotenv import load_dotenv
import xarray as xr
import geodatasets
from .constants import STAC_ITEM_KEYS
from .stackstac_lib import stack

load_dotenv()
stac_endpoint = 'https://planetarycomputer.microsoft.com/api/stac/v1'

def setup_default_logging(log_path, string = 'Train', default_level=logging.INFO,
                          format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s"):

    output_dir = os.path.join(log_path)
    os.makedirs(output_dir, exist_ok=True)

    logger = logging.getLogger(string)

    def time_str(fmt=None):
        if fmt is None:
            fmt = '%Y-%m-%d_%H:%M:%S'
        return datetime.today().strftime(fmt)

    logging.basicConfig(  # unlike the root logger, a custom logger can’t be configured using basicConfig()
        filename=os.path.join(output_dir, f'{time_str()}.log'),
        format=format,
        datefmt="%m/%d/%Y %H:%M:%S",
        level=default_level)

    # print
    # file_handler = logging.FileHandler(filename=os.path.join(output_dir, f'{time_str()}.log'), mode='a')
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(default_level)
    console_handler.setFormatter(logging.Formatter(format))
    # logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def df2gdf(df):
    """
    Convert a pandas.DataFrame to a geopandas.GeoDataFrame
    """
    import geopandas as gpd
    from shapely.geometry import Point
    return gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df[['.geo']]))

def plotPlygonPoints(pointdf, polygondf):
    from matplotlib import pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 10))
    polygondf.plot(ax=ax, facecolor="none", alpha=0.4, color='grey')
    pointdf.plot(ax=ax, color='red', markersize=1)
    plt.savefig('test.png')

def sizeof_fmt(num, suffix="B"):
    for unit in ("", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"):
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f}Yi{suffix}"

def get_class(name):
    class_module, class_name = name.rsplit(".", 1)
    module = __import__(class_module, fromlist=[class_name])
    args_class = getattr(module, class_name)
    return args_class

def get_dense_latlon(central_coords, epsg, resolution=10, grid_size=15):
    """
    Calculate densified latitude and longitude tensors for grids around multiple central points.

    Parameters:
    - central_coords: (latitude, longitude) for the central pixels.
    - resolution: Spatial resolution of the pixels in meters.
    - grid_size: Size of the grid (default is 15x15).

    Returns:
    - Two arrays of size (15, 15) for latitude and longitude.
    """

    # Prepare grid offsets
    half_grid = grid_size // 2
    offsets = np.arange(-half_grid, half_grid + 1) * resolution

    # Extract central latitudes and longitudes
    transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    # Transform central coordinates to the local CRS
    x_center, y_center = transformer.transform(central_coords[0], central_coords[1]) #lon, lat
    # Calculate neighboring coordinates in the local CRS
    lon_vector, lat_vector = transformer.transform(x_center + offsets, y_center - offsets, direction="INVERSE")
    # lon_grid, lat_grid = np.meshgrid(lon_vector, lat_vector, indexing="xy")
    return lon_vector, lat_vector

def get_dense_latlon_array(central_coords_array, resolution=10, grid_size=15):
        """
        Calculate densified latitude and longitude tensors for grids around multiple central points.

        Parameters:
        - central_coords_array: List of tuples (latitude, longitude) for the central pixels.
        - resolution: Spatial resolution of the pixels in meters.
        - grid_size: Size of the grid (default is 15x15).

        Returns:
        - Two tensors of size (n, 15, 15) for latitude and longitude.
        """
        # Convert central coordinates to a tensor
        n_points = central_coords_array.size(0)
        device = central_coords_array.device

        # Prepare grid offsets
        half_grid = grid_size // 2
        offsets = torch.arange(-half_grid, half_grid + 1, dtype=torch.float32, device=device)
        row_offsets, col_offsets = torch.meshgrid(offsets, offsets, indexing="ij")  # Shape: (15, 15)
        
        # Flatten the grid offsets for easy broadcasting
        row_offsets = row_offsets.flatten()  # Shape: (15*15,)
        col_offsets = col_offsets.flatten()  # Shape: (15*15,)

        # Extract central latitudes and longitudes
        central_lats = central_coords_array[:, 0].unsqueeze(1)  # Shape: (n, 1)
        central_lons = central_coords_array[:, 1].unsqueeze(1)  # Shape: (n, 1)

        # Compute conversion factors for degrees per meter
        central_lats_radians = central_lats * (torch.pi / 180)  # Convert degrees to radians
        meters_per_degree_lat = 111132.92  # Approximate mean value for latitude
        meters_per_degree_lon = 111320 * torch.cos(central_lats_radians)  # Adjust for latitude

        degree_per_pixel_lat = resolution / meters_per_degree_lat  # Shape: (n, 1)
        degree_per_pixel_lon = resolution / meters_per_degree_lon  # Shape: (n, 1)

        # Broadcast and compute lat/lon offsets for all points
        lat_offsets = row_offsets.unsqueeze(0) * degree_per_pixel_lat  # Shape: (n, 15*15)
        lon_offsets = col_offsets.unsqueeze(0) * degree_per_pixel_lon  # Shape: (n, 15*15)

        # Add offsets to central coordinates
        lat_pixels = central_lats + lat_offsets  # Shape: (n, 15*15)
        lon_pixels = central_lons + lon_offsets  # Shape: (n, 15*15)

        # Reshape into (n, 15, 15)
        latitudes = lat_pixels.view(n_points, grid_size, grid_size)
        longitudes = lon_pixels.view(n_points, grid_size, grid_size)

        return latitudes, longitudes

def get_deep_size(obj, seen=None):
    """Recursively calculates the deep memory usage of an object."""
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    seen.add(obj_id)
    size = sys.getsizeof(obj)
    
    # Handle containers (lists, tuples, sets, etc.)
    if isinstance(obj, (list, tuple, set, frozenset)):
        for item in obj:
            size += get_deep_size(item, seen)
    elif isinstance(obj, dict):
        for key, value in obj.items():
            size += get_deep_size(key, seen)
            size += get_deep_size(value, seen)
    
    # Handle objects with __dict__ (e.g., class instances)
    try:
        # Directly access __dict__ values to avoid triggering descriptors
        if hasattr(obj, '__dict__'):
            for attr_value in obj.__dict__.values():
                size += get_deep_size(attr_value, seen)
    except AttributeError:
        pass  # Skip if __dict__ is inaccessible
    
    return size


def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    print('Size (MB):', os.path.getsize("temp.p")/1e6)
    os.remove('temp.p')


def create_shared_array(name: str, shape, dtype=np.float32):
    dtype = np.dtype(dtype)
    shm = shared_memory.SharedMemory(create=True, size=np.prod(shape) * dtype.itemsize, name=name)
    array = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
    array[:] = -1  # initialize cache with dummy values
    return shm


def get_wandb_ckpt_path(run= None, run_id: str=None, wandb_project: str=None, model_alias: Union[str, int] = 'best'):
    if run is None:
        import wandb
        api = wandb.Api()
        run = api.run(f"{wandb_project}/{run_id}")
    artifacts = run.logged_artifacts()
    artifacts = [artifact for artifact in artifacts if artifact.type == 'model']
    if isinstance(model_alias, str):
        artifact = [art for art in artifacts if model_alias in art.aliases][0]
    else:
        artifact = artifacts[model_alias]
    ckpt_path = artifact.file()
    return ckpt_path

def get_geom_for_countries(countries_file: str, world_countries_shp: str=None):
    '''
    Get the geometry for the countries in the countries_file
    Args:
        countries_file: str, path to the file containing the name of countries, one country per line
        world_countries_shp: str, path to the world countries shapefile, default to the one in the environment variable WORLD_COUNTRIES_SHP
    Returns:
        geopandas.GeoDataFrame, the geometry for the countries in the countries_file
    '''
    if world_countries_shp is None:
        world_countries_shp = os.getenv('WORLD_COUNTRIES_SHP')
    world_countries_df = gpd.read_file(world_countries_shp)
    with open(Path(countries_file).expanduser(), 'r') as file:
        countries = [line.strip() for line in file]
    countries_df = world_countries_df[world_countries_df['ADMIN'].isin(countries)]
    return countries_df


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

def get_aux_df(parquet_file: str,collection_id:str, filters=None, time_col=None):
    """
    Retrieves auxiliary geodataframe for a specified collection (ESA world cover or DEM).

    Parameter
    ---------
    * collection_id (str): The ID of the collection to retrieve data from.
    * filters (dict, optional): Additional filters to apply to the data. Defaults to None.
    * time_col (str, optional): The name of the time column to include in the retrieved data. Defaults to None.

    Returns
    -------
    * pandas.DataFrame: The retrieved auxiliary data as a pandas DataFrame.
    """
    api = pystac_client.Client.open(stac_endpoint, modifier=planetary_computer.sign_inplace)
    parquet_file = Path(parquet_file).expanduser()
    if parquet_file.exists():
        print(f'Loading STAC parquet files for {collection_id} from: {parquet_file}')
        df = gpd.read_parquet(parquet_file, columns=STAC_ITEM_KEYS+[time_col])
    else:
        print(f'Downloading STAC parquet files for {collection_id}')
        asset = api.get_collection(collection_id).assets["geoparquet-items"]
        df = dgp.read_parquet(
            asset.href, storage_options=asset.extra_fields["table:storage_options"],
            gather_spatial_partitions=False, filters=filters, columns=STAC_ITEM_KEYS+[time_col])
        df = df.compute()
        df.to_parquet(parquet_file)
    df = df.set_index('id')
    return df


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


def get_epsg_from_tile(tile_name):
    zone = int(tile_name[:2])
    band = tile_name[2]
    hemisphere = 'south' if band <= 'M' else 'north'
    epsg = 32700 + zone if hemisphere == 'south' else 32600 + zone
    return epsg


def read_parquets_with_file_name(parquet_dir: Path, columns: List[str] = None):
    """
    Read parquets with file name
    """
    files = list(parquet_dir.glob('*.parquet'))
    
    @dask.delayed
    def read_parquet_and_add_tile_id(file: Path):
        df = gpd.read_parquet(file, columns=columns)
        df['Name'] = file.stem
        return df
    
    tasks = []
    for file in files:
        tasks.append(read_parquet_and_add_tile_id(file))
    dfs = dask.compute(*tasks)
    df = pd.concat(dfs)
    return df


cfg = {
    "sentinel-2-l2a": {
        "assets": {
            "*": {"data_type": "uint16", "nodata": 0},
            "WVP": {"data_type": "uint16", "nodata": 0},
            "B04": {"data_type": "uint16", "nodata": 0},
            "B03": {"data_type": "uint16", "nodata": 0},
            "B02": {"data_type": "uint16", "nodata": 0},
            "B08": {"data_type": "uint16", "nodata": 0},
            "SCL": {"data_type": "uint8", "nodata": 0},
            "visual": {"data_type": "uint16", "nodata": 0},
        },
    },
    "*": {"warnings": "ignore"},
}

def harmonize_to_old(data):
    """
    Harmonize new Sentinel-2 data to the old baseline.

    Parameters
    ----------
    data: xarray.DataArray
        A DataArray with four dimensions: time, band, y, x

    Returns
    -------
    harmonized: xarray.DataArray
        A DataArray with all values harmonized to the old
        processing baseline.
    """
    cutoff = datetime.datetime(2022, 1, 25)
    offset = 1000
    bands = [
        "B01",
        "B02",
        "B03",
        "B04",
        "B05",
        "B06",
        "B07",
        "B08",
        "B8A",
        "B09",
        "B10",
        "B11",
        "B12",
    ]

    old = data.sel(time=slice(cutoff))

    to_process = list(set(bands) & set(data.band.data.tolist()))
    new = data.sel(time=slice(cutoff, None)).drop_sel(band=to_process)

    new_harmonized = data.sel(time=slice(cutoff, None), band=to_process).clip(offset)
    new_harmonized -= offset

    new = xr.concat([new, new_harmonized], "band").sel(band=data.band.data.tolist())
    return xr.concat([old, new], dim="time")


def get_s2_tiles_by_landmass(s2_shfile):
    s2_df = gpd.read_file(s2_shfile)
    world = gpd.read_file(geodatasets.get_path('naturalearth.land'))
    world = world[world.bounds.miny > -60]
    s2_df = s2_df.sjoin(world) # 
    # S2 shapefile from this repo has tiles with more than one geometry. https://github.com/justinelliotmeyers/Sentinel-2-Shapefile-Index
    return s2_df.drop_duplicates(subset='Name', keep='first')
