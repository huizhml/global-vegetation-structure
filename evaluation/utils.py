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

def sample_points_by_biome(biome_file: str, n_samples: int, seed: int = 42, save_dir: str = None, plot_points: bool = True):
    """Vectorized rejection sampling using Shapely 2.0 prepare."""
    
    from shapely import prepare, contains
    rng = np.random.default_rng(seed)
    save_dir = Path(save_dir).expanduser()
    save_dir.mkdir(parents=True, exist_ok=True)
    biome_file = Path(biome_file).expanduser()
    biome_df = gpd.read_file(biome_file)
    biome_df = biome_df.to_crs(epsg=4326)
    gdf_dissolved = biome_df.dissolve(by="BIOME").reset_index()
    for geom in gdf_dissolved.geometry:
        prepare(geom)


    points_by_biome = {}
    for biome in gdf_dissolved.BIOME.unique():
        gdf_biome = gdf_dissolved[gdf_dissolved.BIOME == biome]
        geom = gdf_biome.geometry.iloc[0]  # extract the single geometry
        minx, miny, maxx, maxy = geom.bounds
        points = []
        while len(points) < n_samples:
            batch = max((n_samples - len(points)) * 5, 100)
            xs = rng.uniform(minx, maxx, batch)
            ys = rng.uniform(miny, maxy, batch)
            pts = [Point(x, y) for x, y in zip(xs, ys)]
            mask = contains(geom, pts)  # geometry object, not GeoDataFrame
            points.extend(p for p, m in zip(pts, mask) if m)
        points_by_biome[biome] = points[:n_samples]
    rows = [
        {"BIOME": biome, "geometry": pt}
        for biome, pts in points_by_biome.items()
        for pt in pts
    ]
    points_gdf = gpd.GeoDataFrame(rows, crs=gdf_dissolved.crs)
    
    points_gdf.to_parquet(save_dir / f'random_sample_{n_samples}_points_per_biome.parquet')
    print(f"Saved {len(points_gdf)} points to {save_dir / f'random_sample_{n_samples}_points_per_biome.parquet'}")
    if plot_points:
        points_gdf = points_gdf[~points_gdf.BIOME.isin([98,99])]
        plot_biome_samples(biome_df, points_gdf, save_path=save_dir / f'random_sample_{n_samples}_points_per_biome.pdf')
        
def partition_points_by_tile(gdf_file: str, s2_tile_file: str, save_dir: str):
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
    partition_points_by_tile(
        gdf_file='/projects/dereeco/data/gvs/analysis/biome_anlaysis/random_sample_100000_points_per_biome.parquet',
        s2_tile_file='~/data/gvs/state/s2_tiles_with_growing_months.parquet',
        save_dir='/projects/dereeco/data/gvs/analysis/biome_anlaysis/random_sample_100000_points_per_biome_by_tile'
    )