import os
import time
import threading
import psutil
import glob
from osgeo import gdal
from pathlib import Path
import numpy as np
import rasterio
from rasterio.windows import Window
import xarray as xr
import dask
from rio_cogeo.cogeo import cog_translate
from rio_cogeo.profiles import cog_profiles
from concurrent.futures import ProcessPoolExecutor, FIRST_COMPLETED, wait, as_completed
from const import INDICES_NODATA, MAX_HEIGHT_METERS

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


def create_vrt(tile_dir, vrt_path, q_idx="1"):
    tile_dir = Path(tile_dir).expanduser()
    vrt_path = Path(vrt_path).expanduser()
    vrt_path.parent.mkdir(parents=True, exist_ok=True)

    files = sorted(
        glob.glob(f"{tile_dir}/RH*_Q{q_idx}.tif"), 
        key=lambda x: int(x.split("RH")[-1].split("_")[0]),
    )
    files = files[1:]
    print(f"Creating VRT for {len(files)} files")
    assert len(files) == 100, f"Expected 100 files, found {len(files)}"

    vrt_options = gdal.BuildVRTOptions(separate=True)
    vrt = gdal.BuildVRT(str(vrt_path), files, options=vrt_options)

    for i in range(1, len(files) + 1):
        band = vrt.GetRasterBand(i)
        band.SetDescription(f"RH{i - 1}")

    vrt.FlushCache()
    vrt = None
    return vrt_path
    
    
def _chunk_diversity(data, bin_width=5, max_height=MAX_HEIGHT_METERS):
    """
    Vectorized Shannon entropy for a single spatial chunk. data is in decimeters.

    Parameters
    ----------
    tile : ndarray, shape (101, rows, cols)

    Returns
    -------
    out : ndarray, shape (rows, cols), float32
    """
    if len(data.shape) == 3:
        data = data[None, ...]  # add band dimension for consistency

    n_batch, n_bands, n_rows, n_cols = data.shape
    hist, nodata_mask = batch_binning(data, bin_width=bin_width, max_height=max_height)

    # Normalize
    total = hist.sum(axis=-1, keepdims=True)
    total = np.where(total > 0, total, 1.0)
    p = hist / total

    # FHD
    log_p = np.where(p > 0, np.log(p), 0.0)
    fhd = -np.sum(p * log_p, axis=-1).astype(np.float32)

    # ENL1D
    enl1d = np.exp(fhd).astype(np.float32)

    # ENL2D
    sum_p2 = np.sum(np.where(p > 0, p ** 2, 0.0), axis=-1)
    enl2d = np.where(sum_p2 > 0, 1.0 / sum_p2, np.nan).astype(np.float32)

    # CR
    # TODO: check order of RHs? at least rh25 and rh98
    rh25 = np.maximum(data[:, 24, :], 0)
    rh98 = data[:, 97, :]
    cr = np.where(rh98 > 0, (rh98 - rh25) / rh98, np.nan).astype(np.float32)

    fhd = fhd.reshape(n_batch, n_rows, n_cols)
    enl1d = enl1d.reshape(n_batch, n_rows, n_cols)
    enl2d = enl2d.reshape(n_batch, n_rows, n_cols)
    cr = cr.reshape(n_batch, n_rows, n_cols)
    # Apply nodata
    fhd[nodata_mask] = np.nan
    enl1d[nodata_mask] = np.nan
    enl2d[nodata_mask] = np.nan
    cr[nodata_mask] = np.nan
    return fhd, enl1d, enl2d, cr

def _process_tile(args):
    """
    Worker function for ProcessPoolExecutor.
    Each worker opens its own file handle (required for multiprocessing).
    Reads all 101 bands for one window in a single call, computes entropy.
    """
    vrt_path, col_off, row_off, w, h, bin_width, max_height = args
    # Cap per-worker GDAL block cache so 8 workers don't each grab ~5% of RAM.
    gdal.SetCacheMax(256 * 1024 * 1024)  # 256 MB per worker
    with rasterio.open(vrt_path, "r") as src:
        window = Window(col_off, row_off, w, h)
        tile = src.read(window=window)#  # (101, h, w)， keep original dtype for unit check, m or dm
    ent, enl1d, enl2d, cr = _chunk_diversity(tile, bin_width=bin_width, max_height=max_height)
    return ent, enl1d, enl2d, cr, col_off, row_off, w, h


def create_global_diversity_maps(save_dir: Path=None, tif_dir: str=None,
                         chunk_size=512, max_workers=8, bin_width=5, max_height:int=150, **kwargs):
    """
    Compute global 1km diversity indices (entropy, enl1d, enl2d, cr) from the 100 RH layers from the 1km global mosaics.
    NOTE: RH0 is excluded because all values are negative. (acutally RH1 is also all negative, at least for the global 1km mosaic)
    Two key speedups over the stackstac version:
    1. VRT + rasterio windowed read: single I/O call reads all 101 bands
       for a spatial window (vs stackstac opening 101 separate COGs)
    2. ProcessPoolExecutor: true CPU parallelism for entropy computation
       (vs ThreadPoolExecutor which is GIL-limited for numpy)

    Parameters
    ----------
    save_dir : str
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
    save_dir = Path(save_dir).expanduser()
    save_dir.mkdir(parents=True, exist_ok=True)
    gtif_path = save_dir/f'geotiff/diversity_maps_bin{bin_width}_max{max_height}.tif'
    cog_path = save_dir/f'cog/diversity_maps_bin{bin_width}_max{max_height}.cog.tif'
    if cog_path.exists():
        print(f"Output COG already exists: {cog_path}")
        return cog_path
    if gtif_path.exists():
        return to_cog(gtif_path)
    
    tif_dir = Path(tif_dir).expanduser()
    
    # Create VRT if not provided
    vrt_file = list(tif_dir.glob('*.vrt'))
    if len(vrt_file) == 0:
        vrt_file = tif_dir/f'{gtif_path.stem}.vrt'
        create_vrt(tif_dir, vrt_file)
    else:
        vrt_file = vrt_file[0]

    # Get metadata from VRT
    with rasterio.open(vrt_file, "r") as src:
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
        "count": 4,
        "crs": crs,
        "transform": transform,
        "nodata": INDICES_NODATA,
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
            # ent, enl1d, enl2d, cr, col_off, row_off, w, h = _process_tile((vrt_file, col_off, row_off, w, h, bin_width, max_height))
            work_items.append((vrt_file, col_off, row_off, w, h, bin_width, max_height))

    print(f"Processing {len(work_items)} tiles with {max_workers} processes")
    print(f"Estimated peak RAM: ~{max_workers * 100 * chunk_size**2 * 4 / 1e9:.1f} GB")

    monitor = ProgressMonitor(total_tiles=len(work_items), interval=5.0)

    # Bound in-flight futures so worker results don't accumulate in the result
    # queue (and so completed Futures don't pin their numpy arrays in memory
    # until the whole pool exits).
    max_inflight = max_workers * 2

    with (rasterio.open(gtif_path, "w", **out_profile) as dst):
        monitor.start()
        try:
            with ProcessPoolExecutor(max_workers=max_workers) as pool:
                work_iter = iter(work_items)
                inflight = set()
                # Prime the pool.
                for _ in range(max_inflight):
                    try:
                        inflight.add(pool.submit(_process_tile, next(work_iter)))
                    except StopIteration:
                        break

                while inflight:
                    done, inflight = wait(inflight, return_when=FIRST_COMPLETED)
                    for fut in done:
                        ent, enl1d, enl2d, cr, col_off, row_off, w, h = fut.result()
                        win = Window(col_off, row_off, w, h)
                        data = np.stack([ent, enl1d, enl2d, cr], axis=0).squeeze()  # (4, h, w)
                        dst.write(data, window=win)
                        monitor.tick()
                        # Refill so the pool stays saturated.
                        try:
                            inflight.add(pool.submit(_process_tile, next(work_iter)))
                        except StopIteration:
                            pass
        finally:
            dst.set_band_description(1, "fhd")
            dst.set_band_description(2, "enl1d")
            dst.set_band_description(3, "enl2d")
            dst.set_band_description(4, "cr")
            monitor.stop()

    print(f"Done: {gtif_path}")
    to_cog(gtif_path, cog_path)
    return gtif_path


def create_tile_diversity_maps(output_dir, tile_id, year, vrt_path=None,
                         chunk_size=512, max_workers=8, bin_width=5, max_height=MAX_HEIGHT_METERS, **kwargs):
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

    output_path = output_dir / f'{tile_id}.tif'
    if output_path.exists():
        return

    # Create VRT if not provided
    if vrt_path is None:
        tile_dir = f"~/data/gvs/products/vsm/{year}/original/tiles/cog/{tile_id}"
        vrt_path = f"~/data/gvs/products/vsm/{year}/original/vrt/{tile_id}_Q1.vrt"
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
        "count": 4,
        "crs": crs,
        "transform": transform,
        "nodata": INDICES_NODATA,
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
            # ent, enl1d, enl2d, cr = _process_tile((vrt_path, col_off, row_off, w, h, bin_width, max_height))
            work_items.append((vrt_path, col_off, row_off, w, h, bin_width, max_height))

    print(f"Processing {len(work_items)} tiles with {max_workers} processes")
    print(f"Estimated peak RAM: ~{max_workers * 100 * chunk_size**2 * 4 / 1e9:.1f} GB")

    monitor = ProgressMonitor(total_tiles=len(work_items), interval=5.0)

    with (rasterio.open(output_path, "w", **out_profile) as dst):
        monitor.start()
        try:
            with ProcessPoolExecutor(max_workers=max_workers) as pool:
                futures = {
                    pool.submit(_process_tile, item): i
                    for i, item in enumerate(work_items)
                }
                for fut in as_completed(futures):
                    ent, enl1d, enl2d, cr, col_off, row_off, w, h = fut.result()
                    win = Window(col_off, row_off, w, h)
                    dst.write(np.stack([ent, enl1d, enl2d, cr], axis=0), window=win)
                    monitor.tick()
        finally:
            dst.set_band_description(1, "fhd")
            dst.set_band_description(2, "enl1d")
            dst.set_band_description(3, "enl2d")
            dst.set_band_description(4, "cr")
            monitor.stop()

    print(f"Done: {output_path}")
    return output_path


def to_cog(gtif_path: Path, cog_path: Path=None):
    gtif_path = Path(gtif_path).expanduser()
    if cog_path is None:
        cog_path = gtif_path.with_suffix('.cog.tif')
    else:
        cog_path = Path(cog_path).expanduser()
    if cog_path.exists():
        return cog_path
    output_profile = cog_profiles.get("ZSTD")
    output_profile.update(dict(
        BIGTIFF="IF_SAFER",
        ZSTD_LEVEL=1,
        PREDICTOR=2,
        BLOCKYSIZE=1024,
        BLOCKXSIZE=1024,
        MAX_Z_ERROR=0
    ))
    config = dict(
        GDAL_NUM_THREADS="ALL_CPUS",
        GDAL_TIFF_INTERNAL_MASK=True,
        GDAL_TIFF_OVR_BLOCKSIZE="128",
    )
    cog_translate(gtif_path, cog_path, output_profile, config=config, in_memory=False, quiet=True, use_cog_driver=True)
    print(f"✅ {gtif_path} written to {cog_path}")
    return cog_path
    


# ============================================================================
# Hydra entrypoint
# ============================================================================
import hydra
from config.loader import register
from config.runner import run_cli

register(
    Path(__file__).resolve().parents[1] / 'config' / 'eval' / 'config.yaml',
    section='diversity_maps',
    default_run='create_global_diversity_maps',
)


@hydra.main(config_name='no_log', version_base='1.2', config_path='../config/base')
def main(cfg):
    run_cli(cfg)


if __name__ == '__main__':
    main()

    
    
