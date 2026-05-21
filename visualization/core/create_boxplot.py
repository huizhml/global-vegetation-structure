import h5py
import geopandas as gpd
from postprocessing.core.s2_tiling import find_intersecting_s2_tiles
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

from const import FIGURE_SIZES, set_plot_fonts, fewer_ticks

set_plot_fonts()
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from download.products.sentinel2.inference import query_growing_season_images_by_api, get_top_20_images

def get_cloud_cover(h5_dir: Path, tile: str, s2_grid: gpd.GeoDataFrame, year: int = 2020, max_cloud_cover: int = 90):
    tiles = find_intersecting_s2_tiles(s2_grid, tile)
    cloud_cover_dict = {}
    for tile_id in tiles:
        if (h5_dir / f'{tile_id}.h5').exists():
            cloud_cover = get_cloud_cover_from_h5(h5_dir / f'{tile_id}.h5', tile_id)
            cloud_cover_dict[tile_id] = cloud_cover
        else:
            cloud_cover = get_cloud_cover_from_api(tile_id, s2_grid, year, max_cloud_cover)
            cloud_cover_dict[tile_id] = cloud_cover
    return cloud_cover_dict

def get_cloud_cover_from_h5(h5_file: Path, tile: str):
    with h5py.File(h5_file, 'r') as f:
        cloud_cover = f['eo:cloud_cover'][()]
    return cloud_cover

def get_cloud_cover_from_api(tile_id: str, s2_grid: gpd.GeoDataFrame, year: int = 2020, max_cloud_cover: int = 90):
    df = query_growing_season_images_by_api(tile_id, s2_grid, year, max_cloud_cover)
    df = get_top_20_images(df)
    return df['eo:cloud_cover'].values

def plot_cloud_cover_box(cloud_cover_dict, *, tile: str, **kwargs):
    '''
    Plot the cloud cover in boxplot for the input images for the given tile
    Args:
        cloud_cover: xarray.DataArray, cloud cover box for the given tile
        tile: str, the tile ID
    Returns:
        fig: matplotlib.figure.Figure, the figure object
    '''
    fig, ax = plt.subplots(figsize=kwargs.get('figsize', FIGURE_SIZES['medium']))
    data = list(cloud_cover_dict.values())
    labels = list(cloud_cover_dict.keys())
    colors = plt.cm.tab10.colors  # or plt.cm.Set3.colors
        
    bp = ax.boxplot(data, patch_artist=True, showfliers=False, widths=0.6,
                    medianprops={'color': 'white', 'linewidth': 2, 'zorder': 3},
                    boxprops={'edgecolor': '#333333'},
                    whiskerprops={'color': '#333333'})

    # 2. Style boxes and add jittered points
    for i, (patch, color) in enumerate(zip(bp['boxes'], colors)):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
        
        # Add jittered points to show the actual distribution
        y = data[i]
        x = np.random.normal(i + 1, 0.04, size=len(y))
        ax.scatter(x, y, alpha=0.2, color=color, s=5, zorder=2)

    # 3. Add tile names as X-axis labels instead of floating text
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=45)
    ax.set_ylabel('Cloud Cover (%)')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    fewer_ticks(ax, axis='y')
    plt.tight_layout()
    return fig

# --------- Main Functions ---------
def make_cloud_cover_boxplot(h5_dir: str, tile_id_file: str, pdf_file: Path, s2_grid_file: str, year: int = 2020, max_cloud_cover: int = 90, **kwargs):
    timestamp = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    h5_dir = Path(h5_dir).expanduser()
    tile_id_file = Path(tile_id_file).expanduser()
    pdf_file = Path(pdf_file).expanduser()
    pdf_file.parent.mkdir(parents=True, exist_ok=True)
    pdf_file = pdf_file.with_stem(f'{pdf_file.stem}_{timestamp}')
    s2_grid = gpd.read_parquet(s2_grid_file, columns=['Name', 'geometry', 'growing_months'])
    with PdfPages(pdf_file) as pdf:
        tile_ids = np.loadtxt(tile_id_file, dtype=str)
        for tile in tile_ids:
            cloud_cover_list = get_cloud_cover(h5_dir, tile, s2_grid, year, max_cloud_cover)
            fig = plot_cloud_cover_box(cloud_cover_list, tile=tile)
            pdf.savefig(fig)
            plt.close(fig)
    print(f'saved to {pdf_file}')