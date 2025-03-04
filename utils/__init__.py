import os
import sys
import logging
from datetime import datetime
from pyproj import Transformer
import torch

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

def get_dense_latlon(central_coords_array, resolution=10, grid_size=15):
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
