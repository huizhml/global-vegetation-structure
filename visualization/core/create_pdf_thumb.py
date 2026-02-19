from pathlib import Path
import pandas as pd
import rasterio
import rioxarray as rio
import h5py
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import Colormap
from typing import Union
import copy
import os
os.environ['HYDRA_FULL_ERROR'] = '1'
import geopandas as gpd
from postprocessing.core.s2_tiling import find_intersecting_s2_tiles
import pystac
import stackstac


# --------- I/O Functions ---------
def rio_read(tif_file: Path, overview_level: int = 0):
    return rio.open_rasterio(tif_file, overview_level=overview_level, masked=True).squeeze()

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
    
    
    
