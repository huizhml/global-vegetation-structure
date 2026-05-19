# pip install rasterio plotly numpy affine
import numpy as np
import rasterio as rio
from affine import Affine
from pathlib import Path
import plotly.graph_objects as go
import xarray as xr
import matplotlib.pyplot as plt
from PIL import Image
import json
import yaml
from dataclasses import dataclass
from typing import Optional, List
from hydra.core.config_store import ConfigStore
import hydra
from omegaconf import OmegaConf
import pystac
import stackstac
from pyproj import CRS
from shapely.geometry import Point, box
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import gridspec
import geopandas as gpd
import pandas as pd
from visualization._utils import get_epsg_from_tile
from const import FONT_SIZES, set_plot_fonts

set_plot_fonts()


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
    matched = matched.iloc[0] # TODO: currently only support one tile
    return matched, points

class DataInterface:
    def __init__(self, 
                 zarr_path: str = None,
                 stac_collection_dir: str = None,
                 **kwargs):
        self.zarr_path = Path(zarr_path).expanduser()
        self.stac_collection_dir = Path(stac_collection_dir).expanduser()
        self.year = self.zarr_path.stem.split('_')[1]
        
    
    def _load_s2_patch(self, tile_id: str, image_time: int, window: rio.windows.Window=None, x_coords: List[float]=None, y_coords: List[float]=None):
        ds = xr.open_zarr(self.zarr_path, group=tile_id, consolidated=False, chunks='auto')
        ds = ds.sel(time=~ds.get_index("time").duplicated())
        ds = ds.s2.sel(time=f'{image_time}T00:00:00', method='nearest').sel(band=['B04', 'B03', 'B02'])
        if window is not None:
            s2_patch = ds.isel(x=slice(window.col_off, window.col_off + window.width), y=slice(window.row_off, window.row_off + window.height))
        else:
            s2_patch = ds.sel(x=x_coords, y=y_coords)
        return s2_patch.compute()
    
    def _load_item(self, tile_id: str):
        return pystac.Item.from_file( str(self.stac_collection_dir / f'{tile_id}_{self.year}' / f'{tile_id}_{self.year}.json'))
    
    def _load_rh_patch(self, tile_id: str, rh_idx: int, q_idx: int = 1, window: rio.windows.Window = None):
        item = pystac.Item.from_file( str(self.stac_collection_dir / f'{tile_id}_{self.year}' / f'{tile_id}_{self.year}.json'))
        tile_dir = Path(item.assets['RH98_Q1'].href.replace('file://', '')).parent
        file_path = tile_dir / f'RH{rh_idx}_Q{q_idx}.tif'
        with rio.open(file_path) as src:
            rh = src.read(1, window=window, masked=True)
            T: Affine = src.window_transform(window)
            nodata = src.nodata
        return rh, T, nodata
    
    def get_patch_by_coords(self,
            points: List[dict], s2_grid_file: str = None, buffer_m=100, q_idx=1,
            best_image_time: str = None):
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
        s2_grid = gpd.read_parquet(Path(s2_grid_file).expanduser())
        coords = [point['loc'] for point in points]
        matched, points = get_tile_id_by_coords(coords, s2_grid, self.year)
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
        

        # Load STAC item from local STAC collection, href might be outdated, but metadata like bbox, shape, transform, gsd are still valid        
        item = self._load_item(tile_id)
        assets = [f'RH{rh_idx}_Q{q_idx}' for rh_idx in range(101)]
        image = stackstac.stack(
            item,
            assets=assets,
            epsg=epsg,
            resolution=10,
            bounds=bounds,  # crop
        ).squeeze()
        s2_patch = self._load_s2_patch(tile_id, best_image_time, x_coords=image.x.data, y_coords=image.y.data)
        x_coords = xr.DataArray(points_utm.geometry.x, dims='points', coords=[points_utm.index])
        y_coords = xr.DataArray(points_utm.geometry.y, dims='points', coords=[points_utm.index])
        
        vsm_points = image.sel(x=x_coords, y=y_coords, method='nearest').compute()
        image = image.sel(band='RH98_Q1').compute()
        
        
        return image/10, s2_patch, vsm_points/10, points_utm
    

def rh_to_vertical_profile(rh_patch: np.ndarray, min_height: int=None, max_height: int=None, step: float=1.0, window: int=3):
    '''
    Convert the RH patch to a vertical profile
    Args:
        rh_patch: 2D array of of a given RH metric, (101,n_points)
        min_height: minimum height
        max_height: maximum height
        step: step size
        window: window size
    Returns:
        grad: gradient of the RH profile
        grad_resampled: resampled gradient of the RH profile
        grad_smoothed: smoothed gradient of the RH profile
    '''
    grad = np.gradient(rh_patch, 1, axis=0) # the 2nd param is the distance array, must be a 1D array or a scalar
    grad = np.where(grad == 0, 1, grad)
    grad = 1/grad/10 # bring back to decimeters
    grad = np.nan_to_num(grad, nan=1)
    x = np.arange(min_height, max_height+step, step)
    grad_resampled = np.zeros((len(x), rh_patch.shape[1]))
    grad_smoothed = np.zeros((len(x), rh_patch.shape[1]))
    for i in range(rh_patch.shape[1]):
        grad_inter = interp1d(rh_patch[:, i], grad[:, i], kind='linear', fill_value=0, bounds_error=False)
        grad_resampled[:, i] = grad_inter(x)
        grad_smoothed[:, i] = savgol_filter(grad_resampled[:, i], window, 1)
    return grad, grad_resampled, grad_smoothed

def plot_rhs_3d(rh_patch, rh_idx: int, transform: Affine, z_exaggeration=2.0, camera=None, aspectratio=(1, 1, 0.25), cmin=None, cmax=None, orthographic=False, title=None):
    '''
    Plot the RH as a 3D surface plot
    Args:
        rh_patch: 2D array of of a given RH metric
        rh_idx: RH index
        transform: Affine transform
        z_exaggeration: z-axis exaggeration factor
        camera: camera parameters
        aspectratio: aspect ratio of the plot
        cmin: minimum value of the colorbar
        cmax: maximum value of the colorbar
        orthographic: whether to use orthographic projection
        title: title of the plot
    Returns:
        fig: plotly figure
    '''
    
    # optional downsample
    factor = max(1, int(np.ceil(max(rh_patch.shape)/1500)))
    if factor > 1:
        rh_patch = rh_patch[::factor, ::factor]
        transform = transform * Affine.scale(factor)

    nrows, ncols = rh_patch.shape
    rows, cols = np.mgrid[0:nrows, 0:ncols]
    X = transform.c + cols*transform.a + rows*transform.b
    Y = transform.f + cols*transform.d + rows*transform.e
    Z = rh_patch.filled(0) * z_exaggeration

    xmin, xmax = float(X.min()), float(X.max())
    ymin, ymax = float(Y.min()), float(Y.max())

    # Create custom tick labels divided by 10
    tick_range = np.linspace(cmin, cmax, 6)

    fig = go.Figure(data=[go.Surface(z=Z, x=X[0, :], y=Y[:, 0], cmin=cmin, cmax=cmax,
                                        colorbar=dict(
        thickness=15,   # make bar thinner
        len=0.5,       # shorten (0–1 relative to plot height)
        y=0.4,          # center it vertically
        yanchor="middle",
        tickmode='array',
        tickvals=tick_range,
        ticktext=[f'{int(v/10)}' for v in tick_range]
    ))])

    cam = camera or dict(
        eye=dict(x=-1.2, y=-2.0, z=1.0),
        center=dict(x=0, y=0, z=0),
        up=dict(x=0, y=0, z=1),
        projection=dict(type='orthographic' if orthographic else 'perspective')
    )

    fig.update_layout(
        scene=dict(
            xaxis=dict(title=dict(text="Longitude", font=dict(size=18)), range=[xmin, xmax]),
            yaxis=dict(title=dict(text="Latitude", font=dict(size=18)),  range=[ymin, ymax]),
            zaxis=dict(
                title=dict(text=f"RH{rh_idx} (m)", font=dict(size=18)),
                range=[cmin, cmax],
                tickmode='array',
                tickvals=tick_range,
                ticktext=[f'{int(v/10)}' for v in tick_range]
            ),
            aspectmode="manual",
            aspectratio=dict(x=aspectratio[0], y=aspectratio[1], z=aspectratio[2]),
            camera=cam,
            bgcolor="white",
        ),
        width=1200,
        height=800,
        margin=dict(l=0, r=0, b=20, t=0),  # Zero margins
        paper_bgcolor="white",
        plot_bgcolor="white",
        title=dict(text=title, y=0.75, xanchor='left', yanchor='top', font=dict(size=16))
    )
    # fig.write_html(f"output/{tile}_{name}.html")
    return fig
    
    
def generate_rhs_3d_animation(img_files: list[Path]=None, save_path: Path=None, duration: int=60, loop: int=0):
    '''
    Generate a 3D animation of the RH metrics for a given tile
    Args:
        img_files: list of image files to visualize
        save_path: path to save the animation
        duration: duration of each frame in ms
        loop: number of times to loop the animation
    Returns:
        None
    '''
    frames = [Image.open(p) for p in img_files]
    frames[0].save(
        save_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration,  # ms per frame
        loop=loop        # 0 = infinite loop
    )

def plot_rhs_2d(rh_patch: np.ndarray, 
                rh_idx: int, 
                nodata: int = None, 
                cmin: float=None, 
                cmax: float=None, 
                figsize: tuple[int, int]=(8, 4),
                shrink: int=1, 
                aspect: int=20):
    '''
    Plot the RH as a 2D image
    Args:
        rh_patch: 2D array of of a given RH metric
        rh_idx: RH index
        nodata: nodata value
        cmin: minimum value of the colorbar
        cmax: maximum value of the colorbar
        figsize: figure size
        shrink: shrink factor of the colorbar
        aspect: aspect ratio of the colorbar
    Returns:
        fig: matplotlib figure
    '''
    fig = plt.figure(figsize=figsize)
    if nodata is not None:
        rh_patch[rh_patch == nodata] = 0
    plt.imshow(rh_patch, cmap='inferno', vmin=cmin, vmax=cmax)
    plt.xticks([])
    plt.yticks([])
    cbar = plt.colorbar(shrink=shrink, aspect=aspect)
    cbar.ax.set_ylabel(f'RH{rh_idx} [m]')
    cbar.set_ticks(np.linspace(cmin, cmax, 6))
    cbar.set_ticklabels([f'{int(tick/10)}' for tick in np.linspace(cmin, cmax, 6)])
    return fig

def plot_rgb(s2_patch: xr.DataArray, max_val: int = 2000):
    '''
    Plot the RGB image of the S2 patch
    Args:
        s2_patch: S2 patch
        max_val: maximum value of the RGB image
    Returns:
        fig: matplotlib figure
    '''
    s2_patch = s2_patch.clip(0, max_val) / max_val
    fig = plt.figure(figsize=(8, 4))
    img = s2_patch.plot.imshow(x='x', y='y', rgb='band')
    img.axes.set_aspect('equal')
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.title('')
    plt.xlabel('')
    plt.ylabel('')
    return fig


def plot_s2_rh_subplots(s2_patch: xr.DataArray, 
                        rh_patch: xr.DataArray, 
                        rh_idxs: list[int], 
                        rgb_max_val: int = 2000, 
                        figsize: tuple[int, int]=(12, 4), 
                        rh_vis_param: dict = None):
    '''
    Plot the S2 and RHs for given tiles
    Args:
        s2_patch: S2 patch
        rh_patch: RH patch
        max_val: maximum value of the RGB image
    Returns:
        fig: matplotlib figure
    '''
    cols = len(rh_idxs) + 1
    rows = 1

    # Pattern: [image, image, cbar, image, cbar]
    ncols = 2 * cols - 1
    widths = []
    for j in range(cols):
        widths.append(1.0)               # image column
        if j > 0:
            widths.append(0.05)          # cbar column right after each image (except first)

    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(
        nrows=1,
        ncols=ncols,
        width_ratios=widths,
        wspace=0.2,  # spacing between image and colorbar columns
    )

    axes_main = []
    caxes = []

    # Create axes: main image axes at even columns; colorbar axes at odd columns
    ax = fig.add_subplot(gs[0, 0])
    axes_main.append(ax)
    for j in range(1, cols):
        ax = fig.add_subplot(gs[0, 2*j-1])
        axes_main.append(ax)
        cax = fig.add_subplot(gs[0, 2*j])
        caxes.append(cax)

    s2_patch = s2_patch.clip(0, rgb_max_val) / rgb_max_val
    s2_patch.plot.imshow(x='x', y='y', rgb='band',  ax=axes_main[0], add_colorbar=False)
    axes_main[0].set(title=f'Sentinel-2 RGB', xticks=[], yticks=[], aspect='equal', xlabel='', ylabel='')

    for i, rh_idx in enumerate(rh_idxs):

        vmin = rh_vis_param[f'rh{rh_idx}']['cmin']
        vmax = rh_vis_param[f'rh{rh_idx}']['cmax']

        im = axes_main[i+1].imshow(rh_patch[i], cmap='inferno', vmin=vmin, vmax=vmax)
        axes_main[i+1].set(title=f'RH{rh_idx} [m]', xticks=[], yticks=[], aspect='equal', xlabel='', ylabel='')
        # Colorbar goes to the dedicated cax for this panel (the slot right of it)
        cax = caxes[i]  # note: i maps to panel i+1
        if cax is not None:
            pos = cax.get_position()
            # shrink to 80% height and center vertically
            cax.set_position([pos.x0, pos.y0 + pos.height*0.15, pos.width, pos.height*0.7])
            cb = fig.colorbar(im, cax=cax)
            cb.set_ticks(np.linspace(vmin, vmax, 4))
            cb.set_ticklabels([f'{int(t/10)}' for t in np.linspace(vmin, vmax, 4)])
    plt.tight_layout()
    return fig


def plot_vertical_profile(
            rgb: np.ndarray = None, 
            rh98_patch: xr.DataArray = None, 
            vsm_points: xr.DataArray = None, 
            points_utm: gpd.GeoDataFrame = None, 
            grad_resampled: np.ndarray = None, 
            grad_smoothed: np.ndarray = None, 
            min_height: int=None, 
            max_height: int=None,
            step: float=1.0,
            fig_kwargs: dict = None
            ):
    
    rgb = rgb.clip(0, 2000) / 2000
    rgb = rgb.data.transpose(1, 2, 0)
    extent = [
        float(rh98_patch.x.min()),
        float(rh98_patch.x.max()),
        float(rh98_patch.y.min()),
        float(rh98_patch.y.max())
    ]
    fig, axes = plt.subplots(2, len(points_utm)+1, figsize=(18, 9), gridspec_kw={'width_ratios': [2] + [1] * len(points_utm)})
    axes[0, 0].imshow(rgb, extent=extent, origin='upper')
    axes[0, 0].set(title='Sentinel-2 RGB', xticks=[], yticks=[], aspect='equal', xlabel='', ylabel='')
    
    axes[1, 0].imshow(rh98_patch, cmap='inferno', extent=extent, origin='upper', vmin=min_height, vmax=max_height)
    axes[1, 0].set(title='RH98 [m]', xticks=[], yticks=[], aspect='equal', xlabel='', ylabel='')
    
    ones = np.arange(vsm_points.shape[0])
    height_intervals = np.arange(min_height, max_height+step, step)
    for i, coord in enumerate(points_utm.geometry):

        for ax in axes[:, 0]:
            ax.plot(coord.x, coord.y, 'ro', markersize=6)
            ax.text(coord.x, coord.y, str(i), color='black', fontsize=FONT_SIZES['annot'], va='bottom')
        
        axes[0, i+1].plot(ones,vsm_points[:, i])
        axes[0, i+1].set_ylim(min_height, max_height)
        axes[0, i+1].set_xlabel('RH0-100')

        axes[1, i+1].plot(grad_resampled[:, i], height_intervals, "-", label="resampled")
        axes[1, i+1].plot(grad_smoothed[:, i], height_intervals, "-", lw=3)
        axes[1, i+1].set_ylim(min_height, max_height)
        axes[1, i+1].set_xlabel('Returned energy (%)')
    return fig

class VisRHS:
    def __init__(self, year: int, 
                 s2_zarr_path: str = None,
                 stac_collection_dir: str = None,
                 output_dir: str = None,
                 **kwargs):
        self.output_dir = Path(output_dir).expanduser()
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.year = year
        self.data_interface = DataInterface(s2_zarr_path, stac_collection_dir)
    
    def make_gif(self, 
                 tile_id: str, 
                 col_off: int, 
                 row_off: int, 
                 width: int, 
                 height: int, 
                 vis_params: dict = None, 
                 annimation_args: dict = None):
        save_dir = self.output_dir / f'{tile_id}/3d_viz'
        gif_png_dir = save_dir / f'3d_pngs'
        gif_png_dir.mkdir(exist_ok=True, parents=True)
        
        
        window = rio.windows.Window(
            col_off=col_off,
            row_off=row_off,
            width=width,
            height=height)

        for rh_idx in range(101):
            if not (gif_png_dir / f'{tile_id}_RH{rh_idx}.png').exists():
                rh_patch, transform = self.data_interface._load_rh_patch(tile_id, rh_idx, window)
                fig = plot_rhs_3d(rh_patch, rh_idx, transform, **vis_params)
                fig.write_image(gif_png_dir / f'{tile_id}_RH{rh_idx}.png')
                plt.close()
        
        gif_filename = f'{tile_id}_RH0-100_animation.gif'  
        generate_rhs_3d_animation(list(gif_png_dir.glob(f'{tile_id}_RH*.png')), save_dir / gif_filename, **annimation_args)      

    def make_rhs_2d_plot(self, tile_id: str, 
                         rh_idxs: list[int], 
                         q_idx: int = 1, 
                         col_off: int=None, 
                         row_off: int=None, 
                         width: int=None, 
                         height: int=None, 
                         vis_params: dict = None):
        '''
        Make 2D plots of the RHs for given RH indices of given tiles
        Args:
            tile_id: tile ID
            rh_idxs: list of RH indices to visualize
            q_idx: Q index of the RH
            col_off: column offset
            row_off: row offset
            width: width of the window
            height: height of the window
            vis_params: parameters for the visualization
        Returns:
            None
        '''

        save_dir = self.output_dir / f'{tile_id}/2d_viz'
        save_dir.mkdir(exist_ok=True, parents=True)
        window = rio.windows.Window(
            col_off=col_off,
            row_off=row_off,
            width=width,
            height=height)
        
        for rh_idx in rh_idxs:
            rh_patch, _, nodata = self.data_interface._load_rh_patch(tile_id, rh_idx, q_idx, window)
            fig = plot_rhs_2d(rh_patch, rh_idx, nodata, **vis_params)
            fig.savefig(save_dir / f'RH{rh_idx}_Q{q_idx}.pdf', bbox_inches='tight')
            plt.close()

    def make_rgb_plot(self, tile_id: str, 
                      time_idx: int, 
                      col_off: int=None, 
                      row_off: int=None, 
                      width: int=None, 
                      height: int=None, 
                      max_val: int=2000):
        '''
        Make RGB plot for given tile and time index
        Args:
            tile_id: tile ID
            time_idx: time index
            col_off: column offset
            row_off: row offset
            width: width of the window
            height: height of the window
            vis_params: parameters for the visualization
        Returns:
            None
        '''
        save_dir = self.output_dir / f'{tile_id}/rgb_viz'
        save_dir.mkdir(exist_ok=True, parents=True)
        window = rio.windows.Window(
            col_off=col_off,
            row_off=row_off,
            width=width,
            height=height)
        
        s2_patch = self.data_interface._load_s2_patch(tile_id, time_idx, window)
        fig = plot_rgb(s2_patch, max_val=max_val)
        fig.savefig(save_dir / f'{tile_id}_time{time_idx}.pdf', bbox_inches='tight')
        plt.close()
        
    def make_s2_rh_subplots(self, tile_id: str, rh_idxs: list[int], q_idx: int = 1, time_idx: int=None, col_off: int=None, row_off: int=None, width: int=None, height: int=None):
        '''
        Make subplots of S2 and RHs for given RH indices of given tiles, mainly for comparison of different RH products
        Args:
            tile_id: tile ID
            rh_idxs: list of RH indices to visualize
            q_idx: Q index of the RH
            time_idx: time index
            col_off: column offset
            row_off: row offset
            width: width of the window
            height: height of the window
        Returns:
            None
        '''
        save_dir = self.output_dir / f'{tile_id}/s2_rh_subplots'
        save_dir.mkdir(exist_ok=True, parents=True)
        
        window = rio.windows.Window(
            col_off=col_off,
            row_off=row_off,
            width=width,
            height=height)
        
        s2_patch = self.data_interface._load_s2_patch(tile_id, time_idx, window)
        rh_patches = []
        for rh_idx in rh_idxs:
            rh_patch, transform, nodata = self.data_interface._load_rh_patch(tile_id, rh_idx, q_idx, window)
            rh_patches.append(rh_patch)
        rh_patches = xr.concat(rh_patches, dim='rh_idx')
        fig = plot_s2_rh_subplots(s2_patch, rh_patches, rh_idxs)
        fig.savefig(save_dir / f'{tile_id}_time{time_idx}_RH{rh_idx}_{width}x{height}.pdf', bbox_inches='tight')
        plt.close()
        
    def make_vertical_profile_plot(self, 
                                   q_idx: int = 1,
                                   s2_grid_file: str = None,
                                   name: str = None,
                                   time_idx: int = None, 
                                   best_image_time: str = None,
                                   points: list[tuple[float, float]] = None,
                                   buffer_m: int = None,
                                   step: float = 1.0,
                                   window: int = 3,
                                   min_height: int = 0,
                                   max_height: int = 50,
                                   ):
        '''
        Make vertical profile plot for given tile and time index
        Args:
            tile_id: tile ID
            time_idx: time index
            col_off: column offset
            row_off: row offset
            width: width of the window
            height: height of the window
        Returns:
            None
        '''
        save_dir = self.output_dir / f'vertical_profile_viz/{name}'
        save_dir.mkdir(exist_ok=True, parents=True)
        rh98_patch, s2_patch, vsm_points, points_utm = self.data_interface.get_patch_by_coords(
                points, s2_grid_file=s2_grid_file, buffer_m=buffer_m, q_idx=q_idx,
                best_image_time=best_image_time)
        _, grad_resampled, grad_smoothed = rh_to_vertical_profile(vsm_points.data, min_height, max_height, step, window)
        fig = plot_vertical_profile(s2_patch, rh98_patch, vsm_points, points_utm, grad_resampled, grad_smoothed, min_height, max_height)
        fig.savefig(save_dir / f'{name}_time{time_idx}_vertical_profile.pdf', bbox_inches='tight')
        plt.close()

    def _make_profile_plot(self, axes, coord, vsm_patch, average_over:int=1, min_rh:int=-50, max_rh:int=400, step:float=1.0, window:int=20):
        # plot RH profile
        if average_over > 1: # average over a 3x3 pixel window, the param doesn't really take effect yet
            xcoord = round(coord.x)
            ycoord = round(coord.y)
            rhs = vsm_patch.sel(x=[xcoord-10, xcoord, xcoord+10], y=[ycoord-10, ycoord, ycoord+10], method='nearest')
            rhs = rhs.mean(dim=['x', 'y'])
        else:
            rhs = vsm_patch.sel(x=coord.x, y=coord.y, method='nearest')
            
        ones = np.arange(rhs.size)
        grad = np.gradient(ones, rhs)
        grad = np.nan_to_num(grad, nan=1)
        axes[0].plot(ones,rhs/10)
        axes[0].set_ylim(min_rh, max_rh)
        axes[0].xlabel('RH0-100')
        
        x = np.arange(min_rh, max_rh+step, step)
        grad_inter = interp1d(rhs, grad, kind='linear', fill_value=0, bounds_error=False)
        grad_resampled = grad_inter(x)
        axes[1].plot(grad_resampled, x, "-", label="resampled")
        axes[1].plot(savgol_filter(grad_resampled, window, 1), x, "-", lw=3)
        axes[1].set_ylim(min_rh, max_rh)
        axes[1].xlabel('Returned energy (%)')
        return axes
    
    def vis_density(self):
        '''
        Visualize the vertical density of energy derived from RHs (1/(RH_{i} - RH_{i-1})), the height interval that returns 1% of the total energy.
        Args:
            None
        Returns:
            None
        '''
        save_dir = self.root_save_dir / f'density_figures'
        save_dir.mkdir(exist_ok=True, parents=True)
        zarr_path = self.project_folder / 'deploy' / f'inference_{self.year}.zarr'
        
        s2_grid = gpd.read_parquet(self.project_folder / 'deploy' / 'deploy_status.parquet').to_crs("EPSG:4326")
        
        # load GT
        # gt_dfs = []
        # for idx in range(5):
        #     csv_file = save_dir / f'point{idx}.csv'
        #     gt_dfs.append(pd.read_csv(csv_file, index_col=0).T)
        # gt_df = pd.concat(gt_dfs)
        # rh_cols = [f'rh{i}' for i in range(101)]
        
        for name, cfg in self.tiles_info['density']['examples'].items():
            best_image_time = cfg['best_image_time']
            buffer_m = cfg['buffer_m']
            points = cfg['points']
            point_locs = [point['loc'] for point in points]
            max_rh = cfg.get('max_rh', 40)
            
            vsm_patch, rgb, points_utm = self.data_interface.get_patch_by_coords(
                point_locs, s2_grid, self.year, buffer_m=buffer_m, prediction_dir=self.prediction_dir, input_image_dir=zarr_path,
                stac_collection=self.project_folder / f'deploy/gvsm_stac_catalog/vsm_{self.year}', best_image_time=best_image_time)
            vsm_patch = vsm_patch.compute()
            rgb = rgb.compute()
            rgb = rgb.clip(0, 2000) / 2000
            rgb = rgb.data.transpose(1, 2, 0)
            extent = [
                float(vsm_patch.x.min()),
                float(vsm_patch.x.max()),
                float(vsm_patch.y.min()),
                float(vsm_patch.y.max())
            ]
            fig, axes = plt.subplots(2, len(points_utm)+1, figsize=(18, 9), gridspec_kw={'width_ratios': [2] + [1] * len(points_utm)})
            axes[0, 0].imshow(rgb, extent=extent, origin='upper')
            axes[0, 0].set(title='Sentinel-2 RGB', xticks=[], yticks=[], aspect='equal', xlabel='', ylabel='')
            
            axes[1, 0].imshow(vsm_patch.sel(band='RH98'), cmap='inferno', extent=extent, origin='upper')
            axes[1, 0].set(title='RH98 [m]', xticks=[], yticks=[], aspect='equal', xlabel='', ylabel='')

            for i, coord in enumerate(points_utm.geometry):
                for ax in axes[:, 0]:
                    ax.plot(coord.x, coord.y, 'ro', markersize=6)
                    ax.text(coord.x, coord.y, str(i), color='black', fontsize=FONT_SIZES['annot'], va='bottom')

                self._make_profile_plot(axes, coord, vsm_patch, average_over=1, min_rh=min_rh, max_rh=max_rh, step=1.0, window=20)
                # xcoord = round(coord.x)
                # ycoord = round(coord.y)
                # rhs = vsm_patch.sel(x=[xcoord-10, xcoord, xcoord+10], y=[ycoord-10, ycoord, ycoord+10], method='nearest')
                # rhs = rhs.mean(dim=['x', 'y'])
                # axes[0, i+1].plot(rhs/10, label='Pred')
                # axes[0, i+1].set_title(f'{i}: {points[i]["name"]}')
                # # gt_rhs = gt_df.iloc[i][rh_cols].astype(float)
                # # axes[0, i+1].plot(gt_rhs.values, label='GT')
                # # axes[0, i+1].legend()
                # axes[0, i+1].set_ylim(-10, max_rh)
                
                # density = 1/(rhs - rhs.shift(band=1))
                # density = density.fillna(0)
                # density[density == np.inf]=1
                # axes[1, i+1].plot(density, np.arange(len(density)))
                x = np.arange(len(rhs))
                f_interp = interp1d(rhs, x, kind='linear')
                rhs_fine = np.arange(rhs.min(), rhs.max(), 1)
                x_fine = f_interp(rhs_fine)
                derivative_fine = np.gradient(x_fine, rhs_fine)
                derivate = np.gradient(x, rhs)
                axes[1, i+1].plot(derivative_fine, rhs_fine/10)
                axes[1, i+1].set_ylim(-10, max_rh)
                
                # f_interp_gt = interp1d(gt_rhs, x, kind='linear')
                # rhs_fine_gt = np.arange(gt_rhs.min(), gt_rhs.max(), 0.2)
                # x_fine_gt = f_interp_gt(rhs_fine_gt)
                # derivative_fine_gt = np.gradient(x_fine_gt, rhs_fine_gt*10)
                # axes[1, i+1].plot(derivative_fine_gt, rhs_fine_gt)
                # axes[1, i+1].set_ylim(-10, 30)
                
            plt.savefig(save_dir / f'{name}_{buffer_m}m_{best_image_time}.pdf', bbox_inches='tight')
            plt.close()
            



@hydra.main(config_name='config', config_path='../config/vis_examples', version_base='1.2')
def main(cfg):
    vis = VisRHS(**cfg)
    task_dispatcher = {
        'make_rgb_plot':         vis.make_rgb_plot,
        'make_rhs_2d_plot':      vis.make_rhs_2d_plot,
        'make_vertical_profile_plot': vis.make_vertical_profile_plot,
    }
    # 4. Execution
    if cfg.task in task_dispatcher:
        print(f"Starting task: {cfg.task}")
        try:
            # Execute the mapped function
            task_func = task_dispatcher[cfg.task]
            task_func(**cfg.task_args, **cfg.example_args)
            print(f"Task '{cfg.task}' completed successfully.")
            
        except Exception as e:
            print(f"Error during task '{cfg.task}': {e}", exc_info=True)
            raise e
    else:
        print(f"Unknown task: '{cfg.task}'. Available tasks: {list(task_dispatcher.keys())}")


if __name__ == '__main__':
    main()
