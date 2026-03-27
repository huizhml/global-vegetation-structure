"""
Compute per-pixel Foliage Height Diversity (Shannon entropy of RH profile)
from 101-band GEDI-like rasters.

Fully vectorized — no for loops, no JAX, no numba.
Uses VRT + rasterio windowed reads for fast I/O and
ProcessPoolExecutor for true parallel CPU compute.

Dependencies: numpy, rasterio, gdal, scipy, psutil
Optional: stackstac, pystac, xarray, rioxarray (for STAC-based workflow)
"""

from osgeo import gdal
import glob
import rasterio
from rasterio.windows import Window
import numpy as np
from pathlib import Path
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import threading
import psutil
import os
import time

MAX_HEIGHT = 1000.0
N_BINS = 20
BIN_WIDTH = MAX_HEIGHT / N_BINS
NODATA_IN = 32767
NODATA_OUT = -9999.0
stac_collection_dir = '~/data/gvs/products/gvsm_stac_catalog/vsm_local'


def pixel_fhd(rhs, interval=50):
    """
    Compute per-pixel FHD using a simple histogram approach.
    """
    rhs_arr = np.asarray(rhs, dtype=np.float32)
    rhs_arr = rhs_arr[np.isfinite(rhs_arr)]
    n_bins = int(MAX_HEIGHT / interval)
    hist, bins = np.histogram(rhs_arr, bins=n_bins, range=(0, MAX_HEIGHT))
    p = hist / hist.sum()
    fhd = -np.sum(p * np.log(p), axis=-1).astype(np.float32)
    return fhd


def pixel_enl1d(rhs, interval=50):
    """
    Compute per-pixel ENL0 using a simple histogram approach.
    """
    rhs_arr = np.asarray(rhs, dtype=np.float32)
    rhs_arr = rhs_arr[np.isfinite(rhs_arr)]
    n_bins = int(MAX_HEIGHT / interval)
    hist, bins = np.histogram(rhs_arr, bins=n_bins, range=(0, MAX_HEIGHT))
    p = hist / hist.sum()
    enl0 = -np.sum(p * np.log(p), axis=-1).astype(np.float32)
    return enl0




def _entropy_chunk(tile, bin_width=50):
    """
    Vectorized Shannon entropy for a single spatial chunk.

    Parameters
    ----------
    tile : ndarray, shape (101, rows, cols)

    Returns
    -------
    out : ndarray, shape (rows, cols), float32
    """
    n_bands, n_rows, n_cols = tile.shape
    n_pixels = n_rows * n_cols
    n_bins = int(MAX_HEIGHT / bin_width)
    valid = np.isfinite(tile) & (tile != NODATA_IN) & (tile > 0)
    nodata_mask = valid.sum(axis=0) == 0

    tile_clean = np.where(valid, tile, 0.0)
    bin_idx = np.clip((tile_clean / bin_width).astype(np.int32), 0, n_bins - 1)
    bin_idx = np.where(valid, bin_idx, -1)

    bin_flat = bin_idx.reshape(n_bands, n_pixels)
    pixel_indices = np.broadcast_to(
        np.arange(n_pixels)[np.newaxis, :], (n_bands, n_pixels)
    )

    hist = np.zeros((n_pixels, n_bins), dtype=np.float32)
    flat_valid = bin_flat != -1
    np.add.at(hist, (pixel_indices[flat_valid], bin_flat[flat_valid]), 1.0)

    total = hist.sum(axis=-1, keepdims=True)
    total = np.where(total == 0, 1.0, total)
    p = hist / total
    log_p = np.where(p > 0, np.log(p), 0.0)
    entropy = -np.sum(p * log_p, axis=-1).astype(np.float32)
    enl1d = np.exp(entropy)
    enl2d = (1/ (p**2).sum(axis=-1)).astype(np.float32) # 2D ENL

    entropy = entropy.reshape(n_rows, n_cols)
    enl1d = enl1d.reshape(n_rows, n_cols)
    enl2d = enl2d.reshape(n_rows, n_cols)

    entropy[nodata_mask] = NODATA_OUT
    enl1d[nodata_mask] = NODATA_OUT
    enl2d[nodata_mask] = NODATA_OUT
    return entropy, enl1d, enl2d


def _process_tile(args):
    """
    Worker function for ProcessPoolExecutor.
    Each worker opens its own file handle (required for multiprocessing).
    Reads all 101 bands for one window in a single call, computes entropy.
    """
    vrt_path, col_off, row_off, w, h, bin_width = args
    with rasterio.open(vrt_path, "r") as src:
        window = Window(col_off, row_off, w, h)
        tile = src.read(window=window).astype(np.float32)  # (101, h, w)
    ent, enl1d, enl2d = _entropy_chunk(tile, bin_width=bin_width)
    return ent, enl1d, enl2d, col_off, row_off, w, h


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


def compute_entropy(output_dir, tile_id, year, vrt_path=None,
                         chunk_size=512, max_workers=8, bin_width=50, **kwargs):
    """
    Compute per-pixel FHD entropy — fast version.

    Two key speedups over the stackstac version:
    1. VRT + rasterio windowed read: single I/O call reads all 101 bands
       for a spatial window (vs stackstac opening 101 separate COGs)
    2. ProcessPoolExecutor: true CPU parallelism for entropy computation
       (vs ThreadPoolExecutor which is GIL-limited for numpy)

    Parameters
    ----------
    output_dir : str
        Output single-band GeoTIFF path.
    tile_id : str
        Tile identifier (e.g. '36NTF').
    year : int
        Year for file lookup.
    vrt_path : str, optional
        Path to 101-band VRT. If None, auto-creates from tile directory.
    chunk_size : int
        Spatial chunk size in pixels.
    max_workers : int
        Number of parallel processes.
    """
    output_dir = Path(output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path_entropy = output_dir / f'{tile_id}_entropy.tif'
    output_path_enl1d = output_dir / f'{tile_id}_enl1d.tif'
    output_path_enl2d = output_dir / f'{tile_id}_enl2d.tif'
    if output_path_entropy.exists() and output_path_enl1d.exists() and output_path_enl2d.exists():
        return

    # Create VRT if not provided
    if vrt_path is None:
        tile_dir = f"~/data/gvs/predictions/{year}/original/tiles/cog/{tile_id}"
        vrt_path = f"~/data/gvs/predictions/{year}/original/vrt/{tile_id}_Q1.vrt"
        create_vrt(tile_dir, vrt_path)

    vrt_path = str(Path(vrt_path).expanduser())

    # Get metadata from VRT
    with rasterio.open(vrt_path, "r") as src:
        ny = src.height
        nx = src.width
        crs = src.crs
        transform = src.transform
        n_bands = src.count
        print(f"VRT: {nx}x{ny}, {n_bands} bands, dtype={src.dtypes[0]}")

    assert n_bands == 100, f"Expected 100 bands, got {n_bands}"

    out_profile = {
        "driver": "GTiff",
        "dtype": "float32",
        "height": ny,
        "width": nx,
        "count": 1,
        "crs": crs,
        "transform": transform,
        "nodata": NODATA_OUT,
        "compress": None,
        "tiled": True,
        "blockxsize": 512,
        "blockysize": 512,
    }

    # Build work items
    work_items = []
    for row_off in range(0, ny, chunk_size):
        for col_off in range(0, nx, chunk_size):
            h = min(chunk_size, ny - row_off)
            w = min(chunk_size, nx - col_off)
            work_items.append((vrt_path, col_off, row_off, w, h, bin_width))

    print(f"Processing {len(work_items)} tiles with {max_workers} processes")
    print(f"Estimated peak RAM: ~{max_workers * 100 * chunk_size**2 * 4 / 1e9:.1f} GB")

    monitor = ProgressMonitor(total_tiles=len(work_items), interval=5.0)

    with (
        rasterio.open(output_path_entropy, "w", **out_profile) as dst_entropy,
        rasterio.open(output_path_enl1d, "w", **out_profile) as dst_enl1d,
        rasterio.open(output_path_enl2d, "w", **out_profile) as dst_enl2d,
    ):
        monitor.start()
        try:
            with ProcessPoolExecutor(max_workers=max_workers) as pool:
                futures = {
                    pool.submit(_process_tile, item): i
                    for i, item in enumerate(work_items)
                }
                for fut in as_completed(futures):
                    ent, enl1d, enl2d, col_off, row_off, w, h = fut.result()
                    win = Window(col_off, row_off, w, h)
                    dst_entropy.write(ent[np.newaxis, :, :], window=win)
                    dst_enl1d.write(enl1d[np.newaxis, :, :], window=win)
                    dst_enl2d.write(enl2d[np.newaxis, :, :], window=win)
                    monitor.tick()
        finally:
            monitor.stop()

    print(f"Done: {output_path_entropy}, {output_path_enl1d}, {output_path_enl2d}")
    return output_path_entropy, output_path_enl1d, output_path_enl2d


def compute_entropy_parallel(output_path, year, tile_id, chunk_size=512, max_workers=4):
    """
    Original stackstac-based version (kept for compatibility).
    For faster processing, use compute_entropy() instead.
    """
    import xarray as xr
    import rioxarray
    import pystac
    import stackstac

    output_path = Path(output_path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    stac_file = f'{stac_collection_dir}/{tile_id}_{year}/{tile_id}_{year}.json'
    stac_file = Path(stac_file).expanduser()
    item = pystac.Item.from_file(stac_file)

    data = stackstac.stack(
        item, chunksize=chunk_size,
        assets=[f'RH{i}_Q1' for i in range(100)],
        resolution=10, rescale=False,
        dtype='float32', fill_value=np.float32(np.nan),
    ).squeeze()

    ny, nx = data.sizes["y"], data.sizes["x"]

    profile = {
        "driver": "GTiff", "dtype": "float32",
        "height": ny, "width": nx, "count": 1,
        "crs": data.rio.crs, "transform": data.rio.transform(),
        "nodata": NODATA_OUT, "compress": "deflate",
        "tiled": True, "blockxsize": 256, "blockysize": 256,
    }

    windows = []
    for row_off in range(0, ny, chunk_size):
        for col_off in range(0, nx, chunk_size):
            h = min(chunk_size, ny - row_off)
            w = min(chunk_size, nx - col_off)
            windows.append((row_off, col_off, h, w))

    print(f"Processing {len(windows)} spatial tiles with {max_workers} workers")

    def fetch_and_compute(row_off, col_off, h, w):
        tile = data[:, row_off:row_off + h, col_off:col_off + w].values
        ent = _entropy_chunk(tile)
        return ent, Window(col_off, row_off, w, h)

    write_lock = threading.Lock()
    monitor = ProgressMonitor(total_tiles=len(windows), interval=5.0)

    with rasterio.open(output_path, "w", **profile) as dst:
        monitor.start()
        try:
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = {
                    pool.submit(fetch_and_compute, *w): i
                    for i, w in enumerate(windows)
                }
                for fut in as_completed(futures):
                    ent, win = fut.result()
                    with write_lock:
                        dst.write(ent[np.newaxis, :, :], window=win)
                    monitor.tick()
        finally:
            monitor.stop()

    print(f"Done: {output_path}")


def create_vrt(tile_dir, vrt_path, q_idx="1"):
    tile_dir = Path(tile_dir).expanduser()
    vrt_path = Path(vrt_path).expanduser()
    vrt_path.parent.mkdir(parents=True, exist_ok=True)

    tile_id = tile_dir.stem
    files = sorted(
        glob.glob(f"{tile_dir}/RH*_Q{q_idx}.tif"),
        key=lambda x: int(x.split("RH")[-1].split("_")[0]),
    )
    files = files[1:] # remove RH0_Q1.tif
    print(f"Creating VRT for {tile_id} with {len(files)} files")
    assert len(files) == 100, f"Expected 100 files, found {len(files)}"

    vrt_options = gdal.BuildVRTOptions(separate=True)
    vrt = gdal.BuildVRT(str(vrt_path), files, options=vrt_options)

    for i in range(1, len(files)):
        band = vrt.GetRasterBand(i)
        band.SetDescription(f"RH{i - 1}")

    vrt.FlushCache()
    vrt = None
    return vrt_path


def vertical_profile(rhs, min_rh=-200, max_rh=500, step=1, window=31):
    rhs_arr = np.asarray(rhs, dtype=np.float32)
    rhs_arr = rhs_arr[np.isfinite(rhs_arr)]
    x = np.arange(min_rh, max_rh + step, step, dtype=np.float32)

    if rhs_arr.size < 3:
        return x, np.zeros_like(x, dtype=np.float32)

    rhs_unique = np.unique(np.sort(rhs_arr))
    if rhs_unique.size < 3:
        return x, np.zeros_like(x, dtype=np.float32)

    ones = np.arange(rhs_unique.size, dtype=np.float32)
    grad = np.gradient(ones, rhs_unique)
    grad = np.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
    grad_inter = interp1d(rhs_unique, grad, kind='linear', fill_value=0, bounds_error=False)
    grad_resampled = grad_inter(x)

    max_window = int(grad_resampled.size) if grad_resampled.size % 2 == 1 else int(grad_resampled.size) - 1
    safe_window = min(window if window % 2 == 1 else window + 1, max_window)
    if safe_window < 3:
        smoothed_grad = grad_resampled
    else:
        smoothed_grad = savgol_filter(grad_resampled, safe_window, 1)
    return x, smoothed_grad.astype(np.float32)


if __name__ == "__main__":
    tile_id = '36NTF'
    year = 2020
    vrt_path = f"~/data/gvs/predictions/{year}/original/vrt/{tile_id}_Q1.vrt"
    output_dir = f"~/data/gvs/products/profile_entropy/{year}/tiles/geotiff"

    # Create VRT if needed
    tile_dir = f"~/data/gvs/predictions/{year}/original/tiles/cog/{tile_id}"
    vrt_resolved = Path(vrt_path).expanduser()
    if not vrt_resolved.exists():
        create_vrt(tile_dir, vrt_path)

    start = time.time()
    compute_entropy(
        output_dir=output_dir,
        tile_id=tile_id,
        year=year,
        vrt_path=vrt_path,
        chunk_size=512,
        max_workers=8,
        bin_width=50,
    )
    elapsed = time.time() - start
    print(f"Time taken: {elapsed:.2f} seconds")