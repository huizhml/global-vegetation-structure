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

def read_8neighbors(tile_id: str, s2_grid: gpd.GeoDataFrame, stac_collection_dir: Path, year: int = 2020, resolution: int = 100):
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

# --------- Plotting Functions ---------
def plot_tile_group(tile_group, *, tile: str, cmap: Colormap=None):
    '''
    Plot the RH98 (Q1) metric for an 8-neighbor group of tiles in lower resolution for the given tile
    Args:
        tile_group: xarray.DataArray, already stacked image of the RH98 (Q1) metric for the 8-neighbor group of tiles in lower resolution for the given tile
        cmap: matplotlib.colors.Colormap, the colormap to use
    Returns:
        fig: matplotlib.figure.Figure, the figure object
    '''
    fig = plt.figure(figsize=(12, 10))
    tile_group.plot.imshow(ax=fig.gca(), cmap=cmap, vmin=0, vmax=500, add_colorbar=True)
    fig.gca().set_title(f"{tile} and its neighboring tiles - RH98 Q1")
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

def make_tile_group_pdf(
        tif_dir: str, tile_id_file: str, s2_grid_file: str, stac_collection_dir: str, year: int = 2020, pdf_file: Path=None, resolution: int = 100,
        **kwargs):
    '''
    Make a PDF file where each page renders the RH98 (Q1) metric for an 8-neighbor group of tiles in lower resolution for the given tile
    '''
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
            images = read_8neighbors(tile, s2_grid, stac_collection_dir, year, resolution)
            fig = plot_tile_group(images, tile=tile, cmap=my_cmap)
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
    print(f'saved to {pdf_file}')

def make_rh_pair_pdf(
        tif_dir: str, tile_id_file: str, pdf_file: Path, overview_level: int = 0, top_rh: int = 98, low_rh: int = 25, 
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
    
    
    
