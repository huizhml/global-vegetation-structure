from pathlib import Path
import pandas as pd
import rasterio
import re
import rioxarray as rio
from torch import clamp_min_
import h5py
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import Colormap
from matplotlib.ticker import FuncFormatter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from typing import Union
import copy
import os
import pystac
import stackstac
import geopandas as gpd
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import seaborn as sns

from postprocessing.core.s2_tiling import find_intersecting_s2_tiles
from const import VSM_VIS_PARAMS

os.environ['HYDRA_FULL_ERROR'] = '1'
BAND_NAMES = {
    'fhd': 'FHD',
    'enl1d': 'ENL1D',
    'enl2d': 'ENL2D',
    'cr': 'CR',
    'qskewness': 'Qskewness',
    'q2': 'Q2',
    'rh98_q1': 'RH98 [m]',
    'rh25_q1': 'RH25 [m]',
}
# --------- I/O Functions ---------
def rio_read(tif_file: Path, overview_level: int = 0):
    '''
    NOTE: overview_level 0 is the 1/2 resolution, 1 is 1/4, 2 is 1/8, etc.
    '''
    data =rio.open_rasterio(tif_file, overview_level=overview_level, masked=True)
    with rasterio.open(tif_file) as src:
        band_names = list(src.descriptions)

    data = data.assign_coords(band=band_names)

    if 'RH' in tif_file.stem:
        data = data / 10
        data['band'] = [f'{s.lower()}_q1' for s in tif_file.stem.split('_') if 'RH' in s]
    return data

def read_rh_8neighbors(tile_id: str, s2_grid: gpd.GeoDataFrame, stac_collection_dir: Path, year: int = 2020, resolution: int = 100):
    """
    For a given tile_id, find its 8 neighbors from the s2_grid, and load their stacked STAC assets (`RH98_Q1`)
    from the `stac_collection_dir` for the specified year. Resulting 3D mosaic is returned.

    Args:
        tile_id (str): The S2 tile identifier to center the neighbor search.
        s2_grid (gpd.GeoDataFrame): GeoDataFrame with S2 grid polygons, must include a 'Name' column.
        stac_collection_dir (Path): Root directory containing STAC Item JSON files named '{tile}_{year}/{tile}_{year}.json'.
        year (int, optional): Year for STAC collection subfolder / file naming. Default is 2020.
        resolution (int, optional): Target resolution for stacking the assets (in meters). Default is 100.

    Returns:
        xr.DataArray: Mosaic of the RH98_Q1 images for the target tile and its 8 neighbors.
    """
    tiles = find_intersecting_s2_tiles(s2_grid, tile_id)
    item_list = []
    for tile in tiles:
        item_file = stac_collection_dir / f'{tile}_{year}' / f'{tile}_{year}.json'
        if not item_file.exists():
            print(f'{item_file} not found')
            continue
        item = pystac.Item.from_file(item_file)
        item_list.append(item)
    images = stackstac.stack(item_list, assets=['RH98_Q1'], resolution=resolution, epsg=3857)
    mosaic = stackstac.mosaic(images).squeeze()
    return mosaic


def read_s2_images_from_zarr(zarr_dir: Path, tile_id: str, time_stamp: str=None, resolution: int = 100):
    """
    For a given tile_id, load the Sentinel-2 image from the `zarr_dir` for the specified time_stamp.
    """
    zarr_file = zarr_dir / f'{tile_id}'
    ds = xr.open_zarr(zarr_file)
    if time_stamp is not None:
        ds = ds.sel(time=time_stamp, method='nearest')
    ds = ds.s2.sel(band=['B04', 'B03', 'B02'])
    ds = ds.sel(x=slice(0, resolution), y=slice(0, resolution))
    return ds

def get_vis_params(tif_file: Path):
    q_idx = re.search(r'Q(\d+)', tif_file.stem)
    if q_idx is None: # for Qskewness
        return -120, 120, 'RdBu_r'
    if '-' in q_idx:
        return 0, 500, 'blues'
    else:
        rh_idx = re.search(r'RH(\d+)', tif_file.stem).group(1)
        vis = VSM_VIS_PARAMS[f'rh{rh_idx}_q1']
        return vis['cmin'], vis['cmax'], 'inferno'

# --------- Plotting Functions ---------
def plot_xr_rgb(image: xr.DataArray, *, title: str = None):
    """
    Plot the RGB image using xarray's plot.imshow(), without a colorbar. The image should already be clipped and normalized.

    Args:
        image (xarray.DataArray): The input image to plot. Should have a 'band' dimension representing RGB.
        title (str, optional): Title for the figure.
        cmap (matplotlib.colors.Colormap, optional): Colormap to use (not required for RGB, included for compatibility).
        vmin (float, optional): Minimum value for colormap normalization (not used for RGB).
        vmax (float, optional): Maximum value for colormap normalization (not used for RGB).

    Returns:
        None. The function creates and displays a matplotlib Figure with the plotted RGB image.
    """
    fig = plt.figure(figsize=(12, 10))
    image.plot.imshow(ax=fig.gca(), x='x', y='y', rgb='band')
    fig.gca().set_title(title)
    fig.gca().set_xticks([])
    fig.gca().set_yticks([])
    fig.gca().set_xlabel('')
    fig.gca().set_ylabel('')
    fig.gca().set_aspect('equal')

def plot_xr_image(image: xr.DataArray, *, title: str = None, cmap: Colormap=None, vmin: float = 0, vmax: float = 500):
    """
    Plot a single-band xarray.DataArray (such as the RH98 (Q1) metric), with a colorbar.

    Args:
        image (xarray.DataArray): The array to plot.
        title (str, optional): Title for the figure.
        cmap (matplotlib.colors.Colormap, optional): Colormap for the image.
        vmin (float, optional): Minimum value for colormap normalization (default: 0).
        vmax (float, optional): Maximum value for colormap normalization (default: 500).

    Returns:
        matplotlib.figure.Figure: The resulting figure object displaying the image.
    """
    fig = plt.figure(figsize=(12, 10))
    image.plot.imshow(ax=fig.gca(), cmap=cmap, vmin=vmin, vmax=vmax, add_colorbar=True)
    fig.gca().set_title(title)
    fig.gca().set_xticks([])
    fig.gca().set_yticks([])
    fig.gca().set_xlabel('')
    fig.gca().set_ylabel('')
    fig.gca().set_aspect('equal')
    fig.tight_layout()
    return fig

def plot_rh_pair(rh_top, rh_low, *, tile: str, top_rh: int, low_rh: int, cmap: Colormap=None):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    rh_top.plot.imshow(ax=axes[0], cmap=cmap, vmin=0, vmax=500, add_colorbar=True)
    rh_low.plot.imshow(ax=axes[1], cmap=cmap, vmin=0, vmax=120, add_colorbar=True)
    for ax, rh_idx in zip(axes, [top_rh, low_rh]):
        ax.set_title(f"{tile} RH{rh_idx} Q1")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel('')
        ax.set_ylabel('')
    fig.tight_layout()
    return fig

def plot_pdf_cover(params: dict, timestamp: str):
    fig_cover = plt.figure(figsize=(11, 6))
    
    header = "Processing Parameters\n"
    divider = "-" * 25 + "\n\n"
    timestamp = f"Generated: {timestamp}\n\n"
    
    # 3. Dynamically build the list of parameters
    # We filter out any complex objects that don't print well
    body = ""
    for key, value in params.items():
        # Format key for readability (replace underscores with spaces and capitalize)
        display_key = key.replace('_', ' ').title()
        
        # If the value is a Path, just show the filename or the full string
        if hasattr(value, 'name'):
            display_val = str(value)
        else:
            display_val = value
            
        body += f"{display_key:<20}: {display_val}\n"

    full_text = header + divider + timestamp + body
    fig_cover.text(
        0.1, 0.85, # Coordinates (X=10% from left, Y=85% from bottom)
        full_text, 
        fontsize=12, 
        family='monospace', 
        verticalalignment='top',
        linespacing=1.6
    )
    
    return fig_cover


def plot_tiff_image(image: xr.DataArray, cmin: int=None, cmax: int=None, cmap: str = None):
    '''
    Plot as single-band image with a colorbar.
    Args:
        image: xarray.DataArray
        cmin: minimum value of the colorbar
        cmax: maximum value of the colorbar
        cmap: colormap
    Returns:
        figs: dictionary of matplotlib figures
    '''
    if image.ndim == 2:
        image = image.expand_dims('band')
    figs = {}
    for band in image.band: # NOTE: single band tiff has no band name
        
        band_name = band.item()
        if band_name in VSM_VIS_PARAMS:
            vis = VSM_VIS_PARAMS[band_name] # TODO: Define vis params for other RH metrics, currently only RH98_Q1 (default), and diversity indices
        else:
            vis = VSM_VIS_PARAMS['rh98_q1']
            band_name = 'RH98_Q1'
            
        cmap = cmap or vis['cmap']
        cmin = cmin or vis['cmin']
        cmax = cmax or vis['cmax']

        fig = plt.figure(figsize=(6, 5))
        ax = plt.gca()
        data = image.sel(band=band)
        if cmin is None or cmax is None:
            cmin = float(np.nanpercentile(data, 2))
            cmax = float(np.nanpercentile(data, 98))
            
        print(f'{band.item()} cmin: {cmin}, cmax: {cmax}')
        ax.imshow(data, cmap=cmap, vmin=cmin, vmax=cmax)
        im = ax.get_images()[0]
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel('')
        ax.set_ylabel('')
        fig.tight_layout()
        cax = fig.add_axes([0.06, 0.3, 0.02, 0.15])
        cbar = fig.colorbar(im, cax=cax, orientation='vertical')
        cbar.set_ticks([cmin, cmax])
        cbar.set_ticklabels([f'{cmin:.0f}', f'{cmax:.0f}'])
        cbar.ax.tick_params(size=0, pad=2)
        cbar.outline.set_visible(False)
        figs[band_name] = fig
    return figs


def plot_tiff_image_with_profile(image: xr.DataArray, cmin: int=None, cmax: int=None, cmap: str = None, show_profile: bool = False):
    """
    Plot single-band image on Equal Earth projection with an optional latitudinal mean±sd profile on the left.
    """
    if image.ndim == 2:
        image = image.expand_dims('band')
    figs = {}
    for band in image.band:
        band_name = band.item()
        if band_name in VSM_VIS_PARAMS:
            vis = VSM_VIS_PARAMS[band_name]
        else:
            vis = VSM_VIS_PARAMS['rh98_q1']
            band_name = 'RH98_Q1'

        cmap_ = cmap or vis['cmap']
        cmin_ = cmin or vis['cmin']
        cmax_ = cmax or vis['cmax']

        data = image.sel(band=band)
        if cmin_ is None or cmax_ is None:
            cmin_ = float(np.nanpercentile(data, 2))
            cmax_ = float(np.nanpercentile(data, 98))
        print(f'{band.item()} cmin: {cmin_}, cmax: {cmax_}')
        proj = ccrs.EqualEarth()
        data_crs = ccrs.PlateCarree()

        fig = plt.figure(figsize=(9, 4))

        if show_profile:
            ax_img = fig.add_axes([0.25, 0.05, 0.72, 0.9], projection=proj)
        else:
            ax_img = fig.add_axes([0.05, 0.05, 0.9, 0.9], projection=proj)

        # --- Map ---
        x = data.x.values
        y = data.y.values
        extent = [x.min(), x.max(), y.min(), y.max()]
        if cmap_.lower() == 'mako':
            cmap_ = sns.color_palette("mako", as_cmap=True)
        ax_img.imshow(
            data.values, cmap=cmap_, vmin=cmin_, vmax=cmax_,
            origin='upper', extent=extent,
            transform=data_crs
        )
        ax_img.set_global()
        ax_img.coastlines(linewidth=0.3, color='gray')

        fig.canvas.draw()
        map_pos = ax_img.get_position()

        if show_profile:
            y_map_bot, y_map_top = ax_img.get_ylim()

            ax_prof = fig.add_axes([
                map_pos.x0 - 0.16,
                map_pos.y0,
                0.1,
                map_pos.height
            ])

            data_np = data.values
            if band_name.lower() == 'cr':
                data_np[data_np < 0] = np.nan
            row_mean = np.nanmean(data_np, axis=1)
            row_std = np.nanstd(data_np, axis=1)

            y_projected = np.array([proj.transform_point(0, lat, data_crs)[1] for lat in y])

            ax_prof.fill_betweenx(y_projected, row_mean - row_std, row_mean + row_std,
                                  alpha=0.3, color='gray', label='sd')
            ax_prof.plot(row_mean, y_projected, 'k-', linewidth=0.6, label='mean')
            ax_prof.set_ylim(y_map_bot, y_map_top)

            ax_prof.set_xlabel(BAND_NAMES[band_name.lower()], fontsize=10)
            n_ticks = 2
            xtick_vals = np.linspace(cmin_, cmax_, n_ticks)
            ax_prof.set_xticks(xtick_vals)
            ax_prof.set_xticklabels([f'{v:.0f}' for v in xtick_vals])

            tick_lats = np.arange(-60, 90, 20)
            tick_y_proj = [proj.transform_point(0, lat, data_crs)[1] for lat in tick_lats]
            ax_prof.set_yticks(tick_y_proj)
            ax_prof.set_yticklabels([f'{v:.0f}°' for v in tick_lats])
            ax_prof.set_ylabel('Latitude [°]', fontsize=10)

            ax_prof.legend(loc='lower right', fontsize=7, framealpha=0.7)

        # --- Colorbar inside map ---
        if show_profile:
            cax = fig.add_axes([
                map_pos.x0 - 0.02,
                map_pos.y0 + 0.001,
                map_pos.width * 0.015,
                map_pos.height * 0.25
            ])
        else:
            cax = fig.add_axes([
                map_pos.x0 + 0.14,
                map_pos.y0 + 0.1,
                map_pos.width * 0.015,
                map_pos.height * 0.25
            ])
        im = ax_img.get_images()[0]
        cbar = fig.colorbar(im, cax=cax, orientation='vertical')
        cbar.set_ticks([cmin_, cmax_])
        if band_name.lower() == 'cr':
            cbar.set_ticklabels([f'{cmin_:.2f}', f'{cmax_:.2f}'])
        else:
            cbar.set_ticklabels([f'{cmin_:.0f}', f'{cmax_:.0f}'])
        cbar.ax.tick_params(size=0, pad=3, labelsize=9)
        cbar.outline.set_visible(False)

        figs[band_name] = fig
    return figs
# --------- Main Functions ---------

def make_s2_image_rh_pair_pdf(zarr_dir: str, tile_id: str, pdf_file: Path, year: int = 2020, resolution: int = 100,
        **kwargs):
    """
    Generate a PDF where each page displays the S2 images and RH98 (Q1) metric for a given tile.
    """
    timestamp = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    zarr_dir = Path(zarr_dir).expanduser()
    pdf_file = Path(pdf_file).expanduser()
    pdf_file.parent.mkdir(parents=True, exist_ok=True)
    pdf_file = pdf_file.with_stem(f'{pdf_file.stem}_{timestamp}')
    with PdfPages(pdf_file) as pdf:
        fig_cover = plot_pdf_cover(locals(), timestamp)
        pdf.savefig(fig_cover)
        plt.close(fig_cover)
        ds = read_s2_images_from_zarr(zarr_dir, tile_id, resolution=resolution)
        time_stamps = sorted(ds.time.data)
        for time_stamp in time_stamps:
            image = ds.sel(time=time_stamp)
            fig = plot_xr_rgb(image, title=f'{tile_id} - S2 images at {time_stamp}')
            fig2 = plot_xr_image(image, title=f'{tile_id} - RH98 Q1 at {time_stamp}')
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
    print(f'saved to {pdf_file}')

def make_s2_image_pdf(zarr_dir: str, tile_id: str, pdf_file: Path, year: int = 2020, resolution: int = 100,
        **kwargs):
    """
    Generate a PDF where each page displays the S2 tiles in the given year.
    """
    timestamp = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    zarr_dir = Path(zarr_dir).expanduser()
    pdf_file = Path(pdf_file).expanduser()
    pdf_file.parent.mkdir(parents=True, exist_ok=True)
    pdf_file = pdf_file.with_stem(f'{pdf_file.stem}_{timestamp}')
    with PdfPages(pdf_file) as pdf:
        fig_cover = plot_pdf_cover(locals(), timestamp)
        pdf.savefig(fig_cover)
        plt.close(fig_cover)
        ds = read_s2_images_from_zarr(zarr_dir, tile_id, resolution=resolution)
        time_stamps = sorted(ds.time.data)
        for time_stamp in time_stamps:
            image = ds.sel(time=time_stamp)
            fig = plot_xr_rgb(image, title=f'{tile_id} - S2 images at {time_stamp}')
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
    print(f'saved to {pdf_file}')
    

def make_pred_neighbor_pdf(
        tif_dir: str, tile_id_file: str, s2_grid_file: str, stac_collection_dir: str, year: int = 2020, pdf_file: Path=None, resolution: int = 100,
        **kwargs):
    """
    Generate a PDF where each page displays the RH98 (Q1) metric for an 8-neighbor group of S2 tiles, 
    rendered at a lower resolution. Each page is centered on a target tile (from tile_id_file), 
    surrounded by its neighboring tiles, using the corresponding raster data from tif_dir.
    
    Args:
        tif_dir (str): Directory containing COG GeoTIFFs for each tile, expected under ${tile}/RH98_Q1.tif.
        tile_id_file (str): Path to a file containing a list of tile IDs (one per line).
        s2_grid_file (str): Path to the S2 grid file (parquet format) defining tile geometries.
        stac_collection_dir (str): Directory containing STAC collections (unused in plotting but required).
        year (int): Year for which to generate the images. Used for path/file selection.
        pdf_file (Path, optional): Destination for the output PDF. Timestamp is appended to the filename.
        resolution (int): Resolution (in meters) for downsampling the COG images for display.
        **kwargs: Additional unused keyword arguments.

    Notes:
        - The first page in the PDF summarizes parameters and generation time.
        - Each subsequent page visualizes the RH98 Q1 metric for a tile and its 8 neighbors.
    """
    timestamp = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    tif_dir = Path(tif_dir).expanduser()
    tile_id_file = Path(tile_id_file).expanduser()
    s2_grid_file = Path(s2_grid_file).expanduser()
    stac_collection_dir = Path(stac_collection_dir).expanduser()
    pdf_file = Path(pdf_file).expanduser()
    pdf_file.parent.mkdir(parents=True, exist_ok=True)
    pdf_file = pdf_file.with_stem(f'{pdf_file.stem}_{timestamp}')
    
    my_cmap = copy.copy(plt.get_cmap('inferno'))
    my_cmap.set_bad(color='black')
    tile_ids = np.loadtxt(tile_id_file, dtype=str)
    s2_grid = gpd.read_parquet(s2_grid_file, columns=['Name', 'geometry'])
    with PdfPages(pdf_file) as pdf:
        fig_cover = plot_pdf_cover(locals(), timestamp)
        pdf.savefig(fig_cover)
        plt.close(fig_cover)
        for tile in tile_ids:
            images = read_rh_8neighbors(tile, s2_grid, stac_collection_dir, year, resolution)
            fig = plot_xr_image(images, tile=f'{tile} and its neighboring tiles - RH98 Q1', cmap=my_cmap)
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
    print(f'saved to {pdf_file}')

def make_rh_pair_pdf(
        tif_dir: str, tile_id_file: str, pdf_file: Path, overview_level: int = 0, top_rh: int = 98, low_rh: int = 25, black_background: bool = False,
        **kwargs):
    '''
    Make a PDF file where each page renders a pair of RH metrics in lower resolution for one tile
    '''
    timestamp = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    tif_dir = Path(tif_dir).expanduser()
    tile_id_file = Path(tile_id_file).expanduser()
    pdf_file = Path(pdf_file).expanduser()
    pdf_file.parent.mkdir(parents=True, exist_ok=True)
    pdf_file = pdf_file.with_stem(f'{pdf_file.stem}_{timestamp}')
    my_cmap = copy.copy(plt.get_cmap('inferno'))
    if black_background:
        my_cmap.set_bad(color='black')
    tile_ids = np.loadtxt(tile_id_file, dtype=str)
    with PdfPages(pdf_file) as pdf:
        fig_cover = plot_pdf_cover(locals(), timestamp)
        pdf.savefig(fig_cover)
        plt.close(fig_cover)
        for tile in tile_ids:
            if not (tif_dir / f'{tile}/RH{top_rh}_Q1.tif').exists():
                print(f'{tile} not found in {tif_dir}')
                continue
            rh_top = rio_read(tif_dir / f'{tile}/RH{top_rh}_Q1.tif', overview_level)
            rh_low = rio_read(tif_dir / f'{tile}/RH{low_rh}_Q1.tif', overview_level)
            fig = plot_rh_pair(rh_top, rh_low, tile=tile, top_rh=top_rh, low_rh=low_rh, cmap=my_cmap)
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
    print(f'saved to {pdf_file}')
    
def make_global_mosaic_pdf(mosaic_dir: str, tif_filename_pattern: str = 'global_mosaic_*RH*_Q0-Q2.cog.tif', pdf_file: Path=None, multi_pages: bool = False, cmin: float = None, cmax: float = None, cmap: str = None, show_profile: bool = False, **kwargs):
    '''
    Make a PDF file where each page renders a global mosaic of a TIFF image
    '''
    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    mosaic_dir = Path(mosaic_dir).expanduser()
    tif_files = list(mosaic_dir.glob(tif_filename_pattern))
    tif_files = sorted(tif_files)
    pdf_file = Path(pdf_file).expanduser()
    pdf_file.parent.mkdir(parents=True, exist_ok=True)
    if multi_pages:
        pdf_file = pdf_file.with_stem(f'{pdf_file.stem}_{timestamp}')
        with PdfPages(pdf_file) as pdf:
            fig_cover = plot_pdf_cover(locals(), timestamp)
            pdf.savefig(fig_cover)
            plt.close(fig_cover)
            for tif_file in tif_files:
                image = rio_read(tif_file)
                figs = plot_tiff_image_with_profile(image, cmin=cmin, cmax=cmax, cmap=cmap)
                for band, fig in figs.items():
                    pdf.savefig(fig, bbox_inches='tight', dpi=300)
                    plt.close(fig)
        print(f'saved to {pdf_file}')
    else:
        for tif_file in tif_files:
            image = rio_read(tif_file)
            figs = plot_tiff_image_with_profile(image, cmin=cmin, cmax=cmax, cmap=cmap, show_profile=show_profile)
            for band, fig in figs.items():
                band_name = '' if 'RH' in band else band
                _pdf_file = pdf_file.parent / f'{tif_file.stem}{band_name}.pdf' # band is for multibands tif
                fig.savefig(_pdf_file, bbox_inches='tight', dpi=300)
                fig.savefig(_pdf_file.with_suffix('.png'), bbox_inches='tight', dpi=300)
                plt.close(fig)
                print(f'saved to {_pdf_file}')
    
