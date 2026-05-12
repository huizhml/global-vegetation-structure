import os
# os.environ["PYVISTA_OFF_SCREEN"] = "true"
# os.environ["VTK_DEFAULT_OPENGL_WINDOW"] = "vtkEGLRenderWindow"

import pandas as pd
import xarray as xr
import rioxarray
from rasterio.enums import Resampling
import matplotlib.pyplot as plt
import pystac
from pathlib import Path
import geopandas as gpd
from shapely.geometry import Point
from pyproj import CRS, Transformer
import stackstac
import datetime
from dask.utils import natural_sort_key
import rasterio
import pyvista as pv
import numpy as np

from download.core.utils import get_epsg_from_tile

pv.OFF_SCREEN = True

def plot_datacube(data: np.array, lons: np.array, lats: np.array, save_path: str, cmap: str='inferno'):
    # latitude ascending
    if lats[0] > lats[-1]:
        lats = lats[::-1]
        data = data[::-1, :, :]

    # longitude: convert 0..360 to -180..180, then sort
    if np.nanmax(lons) > 180:
        lons = ((lons + 180) % 360) - 180
        order = np.argsort(lons)
        lons = lons[order]
        data = data[:, order, :]

    data = data/10
    ny, nx, nz = data.shape

    vmin = 0 # np.nanpercentile(data, 2)
    vmax = 50 #np.nanpercentile(data, 98)
    print('vmin, vmax:', vmin, vmax)

    # Create a sharp cutoff: transparent only for sentinel values, opaque for everything else
    sentinel = -1  # vmin - 50.0
    # opacity_points = [
    #     (sentinel, 0.0),        # NaN sentinel: invisible
    #     (sentinel + 1, 0.8),    # just above sentinel: visible
    #     (vmax * 0.5, 0.85),
    #     (vmax, 1.0),
    # ]
    # Extract just the opacity values for the linear mapping
    opacity = [0.0, 0.8, 0.8, 0.85, 0.9, 1.0]
    vol = np.nan_to_num(data, nan=sentinel)
    vol = np.clip(vol, sentinel, vmax)

    # CRITICAL FIX:
    # input is lat, lon, time
    # VTK wants x, y, z = lon, lat, time
    vol_vtk = np.transpose(vol, (1, 0, 2))

    z_spacing = 6.0

    grid = pv.ImageData()
    grid.dimensions = (nx, ny, nz)
    grid.origin = (float(lons.min()), float(lats.min()), 0.0)
    grid.spacing = (
        float((lons.max() - lons.min()) / max(nx - 1, 1)),
        float((lats.max() - lats.min()) / max(ny - 1, 1)),
        z_spacing,
    )

    grid.point_data["values"] = vol_vtk.ravel(order="F")

    p = pv.Plotter(off_screen=True, window_size=(2200, 1200))
    
    p.enable_parallel_projection()

    p.add_volume(
        grid,
        scalars="values",
        cmap=cmap,
        clim=(sentinel, vmax),
        opacity=opacity,
        shade=False,
        show_scalar_bar=False,
    )

    # Dummy mesh for clean scalar bar
    dummy = pv.PolyData(np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]))
    dummy["values"] = np.array([0.0, vmax])
    p.add_mesh(
        dummy,
        scalars="values",
        cmap=cmap,
        clim=(0, vmax),
        show_scalar_bar=True,
        opacity=0.0,
        scalar_bar_args={
            "title": "Height [m]",
            "color": "white",
            "vertical": True,
            "position_x": 0.2,
            "position_y": 0.26,
            "height": 0.52,
            "width": 0.03,
        },
    )

    # p.add_mesh(grid.outline(), color="black", opacity=0.7, line_width=1)

    # -----------------------
    # More top-down oblique camera
    # -----------------------
    bounds = grid.bounds
    xmin, xmax, ymin, ymax, zmin, zmax = bounds

    cx = 0.5 * (xmin + xmax)
    cy = 0.5 * (ymin + ymax)
    cz = 0.5 * (zmin + zmax)

    p.camera_position = [
        (cx, ymin - 80, zmax + 300),  # 更高、更少侧向偏移
        (cx, cy, cz),
        (0, 0, 1),
    ]

    p.enable_parallel_projection()
    p.camera.zoom(1.2)

    p.set_background("black")
    p.show(auto_close=False)
    img = p.screenshot(transparent_background=False, return_img=True)
    p.close()

    # Auto-crop background borders. Anything brighter than `bg_tol` counts as content,
    # so the scalar bar / text / volume are all preserved.
    bg_tol = 6
    mask = img.max(axis=2) > bg_tol
    if mask.any():
        rows = np.where(mask.any(axis=1))[0]
        cols = np.where(mask.any(axis=0))[0]
        pad = 8
        r0 = max(int(rows[0]) - pad, 0)
        r1 = min(int(rows[-1]) + 1 + pad, img.shape[0])
        c0 = max(int(cols[0]) - pad, 0)
        c1 = min(int(cols[-1]) + 1 + pad, img.shape[1])
        img = img[r0:r1, c0:c1]

    plt.imsave(save_path, img)

def read_overview(file: str, overview_level: int=4):
    with rasterio.open(file, overview_level=overview_level) as src:
        data = src.read(1)  # Reads directly from the overview, very fast
    return data

def read_coords(file: str, overview_level: int=4):
    with rasterio.open(file, overview_level=overview_level) as src:
        transform = src.transform
        cols = np.arange(src.width)
        rows = np.arange(src.height)
        lons = transform[2] + cols * transform[0]
        lats = transform[5] + rows * transform[4]
        nodata = src.nodata
    return lons, lats, nodata

def read_datacube(data_dir: str, filename_pattern: str='.cog.tif', overview_level: int=4, rh_step: int=2):
    data_dir = Path(data_dir).expanduser()
    files = list(data_dir.glob(filename_pattern))
    files = [str(file) for file in files]
    files = sorted(files, key=natural_sort_key)
    data = []
    for file in files[::rh_step]:
        data_i = read_overview(file, overview_level=overview_level)
        data.append(data_i[:, :, None])
    data = np.concatenate(data, axis=2)
    lons, lats, nodata = read_coords(files[0], overview_level=overview_level)
    data =  data.astype(np.float32)
    data[data == nodata] = np.nan
    return data, lons, lats

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

def visualize_datacube(data_dir: str, save_path: str, cmap: str='viridis', rh_step: int=2, **kwargs):
    data, lons, lats = read_datacube(data_dir, filename_pattern='*cog.tif', rh_step=rh_step, **kwargs)
    plot_datacube(data, lons, lats, save_path, cmap=cmap)
    
if __name__ == '__main__':
    data_dir = '~/data/gvs/products/prediction_intervals/2020/masked/mosaic/'
    save_path = '/projects/dereeco/data/gvs/results/vsm_datacube/every2rhs_black_bg_v2.png'
    visualize_datacube(data_dir, save_path, cmap='viridis')