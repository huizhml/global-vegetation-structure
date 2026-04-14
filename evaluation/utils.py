import time
import threading
import psutil
import os
from shapely import Point, get_coordinates, points
import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import xarray as xr
from tqdm import tqdm
from concurrent.futures import as_completed
import stackstac
import planetary_computer as pc
from pystac_client import Client
from rasterio.transform import rowcol
import rasterio
from concurrent.futures import ThreadPoolExecutor

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"

NON_VEGETATION_CLASSES = {50, 60, 70, 80}


class ProgressMonitor:
    """Background thread that prints speed and RAM stats."""

    def __init__(self, total_tiles, interval=5.0):
        self.total = total_tiles
        self.done = 0
        self.interval = interval
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._start_time = time.time()
        self._process = psutil.Process(os.getpid())
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()

    def tick(self):
        with self._lock:
            self.done += 1

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=2)

    def _run(self):
        while not self._stop.wait(self.interval):
            self._print_status()
        self._print_status()

    def _print_status(self):
        with self._lock:
            done = self.done
        elapsed = time.time() - self._start_time
        mem = self._process.memory_info()
        rss_gb = mem.rss / (1024 ** 3)
        try:
            for child in self._process.children(recursive=True):
                rss_gb += child.memory_info().rss / (1024 ** 3)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
        rate = done / elapsed if elapsed > 0 else 0
        eta = (self.total - done) / rate if rate > 0 else float('inf')
        pct = 100 * done / self.total if self.total > 0 else 0
        print(
            f"  [{done}/{self.total}] {pct:5.1f}% | "
            f"{rate:.2f} tiles/s | "
            f"elapsed {elapsed:.0f}s | "
            f"ETA {eta:.0f}s | "
            f"RAM {rss_gb:.2f} GB (total)"
        )



def plot_biome_samples(gdf_dissolved, points_gdf, save_path=None, figsize=(16, 10)):
    """Plot biome polygons with sampled points overlaid, no explicit loops."""
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Plot all biome polygons at once
    gdf_dissolved.plot(
        ax=ax, column="BIOME", cmap="tab20", alpha=0.5,
        edgecolor="black", linewidth=0.3, legend=True,
        # legend_kwds={"fontsize": 7, "loc": "lower left", "ncol": 2}
    )

    # Plot all points at once, colored by biome
    points_gdf.plot(
        ax=ax, column="BIOME", cmap="tab20", markersize=12,
        edgecolor="k", linewidth=0.3, zorder=5, alpha=0.8
    )

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect("equal")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    



def _sample_tile(item, xs, ys, indices):
    """Read land cover values for points falling within one tile."""
    href = pc.sign(item.assets["map"].href)
    result = {}
    with rasterio.open(href) as src:
        tb = src.bounds
        in_tile = (
            (xs[indices] >= tb.left) & (xs[indices] <= tb.right) &
            (ys[indices] >= tb.bottom) & (ys[indices] <= tb.top)
        )
        idx = indices[in_tile]
        if len(idx) == 0:
            return result
        rows, cols = rowcol(src.transform, xs[idx], ys[idx])
        rows = np.clip(rows, 0, src.height - 1)
        cols = np.clip(cols, 0, src.width - 1)
        # Windowed read to avoid loading the full tile
        row_min, row_max = int(np.min(rows)), int(np.max(rows))
        col_min, col_max = int(np.min(cols)), int(np.max(cols))
        window = rasterio.windows.Window(
            col_min, row_min,
            col_max - col_min + 1, row_max - row_min + 1,
        )
        data = src.read(1, window=window)
        local_rows = np.array(rows) - row_min
        local_cols = np.array(cols) - col_min
        for i, r, c in zip(idx, local_rows, local_cols):
            result[i] = data[r, c]
    return result


def get_landcover_values(points_gdf, collection="esa-worldcover", year=2021, max_workers=16):
    """Query ESA WorldCover tile-by-tile in parallel via thread pool."""
    catalog = Client.open(STAC_URL, modifier=pc.sign_inplace)
    bbox = [float(b) for b in points_gdf.total_bounds]
    search = catalog.search(
        collections=[collection],
        bbox=bbox,
        datetime=f"{year}-01-01/{year}-12-31",
    )
    items = list(search.items())
    print(f"  Found {len(items)} WorldCover tiles")

    xs = points_gdf.geometry.x.values
    ys = points_gdf.geometry.y.values
    lc_values = np.full(len(points_gdf), -1, dtype=np.int16)
    remaining = np.arange(len(points_gdf))

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_sample_tile, item, xs, ys, remaining): item
            for item in items
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Querying WorldCover tiles"):
            for i, val in future.result().items():
                lc_values[i] = val

    # Update remaining so already-resolved points aren't rechecked
    n_resolved = np.sum(lc_values != -1)
    print(f"  Resolved {n_resolved}/{len(points_gdf)} points")
    return lc_values

def sample_points_by_biome(biome_file: str, n_samples: int, seed: int = 42,
                           save_dir: str = None, plot_points: bool = True,
                           collection: str = "esa-worldcover", **kwargs):
    '''
    Sample vegetation points by biome from the GEDI ecoregions shapefile.
    It took ~2301s to sample 100000 points per biome.
    Parameters:
        biome_file: path to the GEDI ecoregions shapefile
        n_samples: number of samples per biome
        seed: random seed
        save_dir: path to save the sampled points
        plot_points: whether to plot the sampled points
        collection: name of the ESA WorldCover collection
        year: year of the ESA WorldCover data
    Returns:
        points_gdf: GeoDataFrame of the sampled points
    '''
    year = kwargs.get('year', 2021)
    from shapely import prepare, contains

    NON_VEGETATION_CLASSES = {50, 60, 70, 80}
    OVERSAMPLE_FACTOR = 2

    rng = np.random.default_rng(seed)
    save_dir = Path(save_dir).expanduser()
    save_dir.mkdir(parents=True, exist_ok=True)
    biome_file = Path(biome_file).expanduser()
    biome_df = gpd.read_file(biome_file)
    biome_df = biome_df.to_crs(epsg=4326)
    gdf_dissolved = biome_df.dissolve(by="BIOME").reset_index()
    gdf_dissolved = gdf_dissolved[~gdf_dissolved.BIOME.isin({98, 99})]
    for geom in gdf_dissolved.geometry:
        prepare(geom)

    # --- Stage 1: oversample 2x with rejection sampling on geometry only ---
    n_oversample = n_samples * OVERSAMPLE_FACTOR
    points_by_biome = {}
    for biome in gdf_dissolved.BIOME.unique():
        geom = gdf_dissolved[gdf_dissolved.BIOME == biome].geometry.iloc[0]
        minx, miny, maxx, maxy = geom.bounds
        points = []
        while len(points) < n_oversample:
            batch = max((n_oversample - len(points)) * 5, 100)
            xs = rng.uniform(minx, maxx, batch)
            ys = rng.uniform(miny, maxy, batch)
            pts = [Point(x, y) for x, y in zip(xs, ys)]
            mask = contains(geom, pts)
            points.extend(p for p, m in zip(pts, mask) if m)
        points_by_biome[biome] = points[:n_oversample]

    # Build full oversampled GeoDataFrame
    rows = [
        {"BIOME": biome, "geometry": pt}
        for biome, pts in points_by_biome.items()
        for pt in pts
    ]
    all_points = gpd.GeoDataFrame(rows, crs=gdf_dissolved.crs)
    
    lc_values = get_landcover_values(all_points, collection=collection, year=year)
    all_points["lc_class"] = lc_values
    vegetation = all_points[
        (~all_points.lc_class.isin(NON_VEGETATION_CLASSES)) & (all_points.lc_class != -1)
    ].copy()

    # --- Stage 3: take n_samples per biome from the filtered set ---
    points_gdf = (
        vegetation
        .groupby("BIOME", group_keys=False)
        .apply(lambda g: g.sample(n=min(n_samples, len(g)), random_state=seed))
        # .drop(columns="lc_class")
        .reset_index(drop=True)
    )

    # Warn if any biome came up short
    counts = points_gdf.groupby("BIOME").size()
    short = counts[counts < n_samples]
    if not short.empty:
        for biome, count in short.items():
            print(f"  Warning: biome {biome} only got {count}/{n_samples} vegetation points")

    points_gdf.to_parquet(save_dir / f'random_sample_{n_samples}_points_per_biome.parquet')
    print(f"Saved {len(points_gdf)} points")

    if plot_points:
        plot_biome_samples(biome_df, points_gdf,
                           save_path=save_dir / f'random_sample_{n_samples}_points_per_biome.pdf')
        
               
def partition_points_by_tile(gdf_file: str, s2_tile_file: str, save_dir: str, **kwargs):
    '''
    Partition the points by tile.
    Args:
        gdf_file: path to the points GeoDataFrame
        s2_tile_file: path to the S2 tile file
        save_dir: path to save the partitioned points
    Returns:
    '''
    save_dir = Path(save_dir).expanduser()
    save_dir.mkdir(parents=True, exist_ok=True)
    gdf_file = Path(gdf_file).expanduser()
    s2_tile_file = Path(s2_tile_file).expanduser()
    points_gdf = gpd.read_parquet(gdf_file)
    s2_tile_df = gpd.read_parquet(s2_tile_file, columns=['Name', 'geometry', 'covered_by_gedi'])
    s2_tile_df = s2_tile_df.to_crs(epsg=4326)
    points_gdf = points_gdf.to_crs(epsg=4326)
    points_gdf = gpd.sjoin(points_gdf, s2_tile_df, how='left', predicate='intersects')
    points_gdf = points_gdf.drop_duplicates(subset='geometry', keep='first')
    points_gdf = points_gdf.drop(columns=['index_right'])
    for tile in points_gdf.Name.unique():
        points_gdf_tile = points_gdf[points_gdf.Name == tile]
        points_gdf_tile.to_parquet(save_dir / f'{tile}.parquet')
        print(f"Saved {len(points_gdf_tile)} points to {save_dir / f'{tile}.parquet'}")

if __name__ == "__main__":
    sample_points_by_biome(
        biome_file='/projects/dereeco/data/GEDI/ecoregions/wwf_terr_ecos.shp',
        n_samples=100000,
        save_dir='/projects/dereeco/data/gvs/analysis/biome_anlaysis/random_sample_100000_points_per_biome',
        plot_points=True
    )
    # partition_points_by_tile(
    #     gdf_file='/projects/dereeco/data/gvs/analysis/biome_anlaysis/random_sample_100000_points_per_biome.parquet',
    #     s2_tile_file='~/data/gvs/state/s2_tiles_with_growing_months.parquet',
    #     save_dir='/projects/dereeco/data/gvs/analysis/biome_anlaysis/random_sample_100000_points_per_biome_by_tile'
    # )