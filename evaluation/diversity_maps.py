import os
import time
import json
import threading
import psutil
import glob
from datetime import datetime
from osgeo import gdal
from pathlib import Path
import numpy as np
import rasterio
from rasterio.windows import Window
import xarray as xr
import dask
from rio_cogeo.cogeo import cog_translate
from rio_cogeo.profiles import cog_profiles
from concurrent.futures import ProcessPoolExecutor, FIRST_COMPLETED, wait
from evaluation.utils import batch_binning
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
    assert len(files) == 100, f"Expected 100 files, found {len(files)} in {tile_dir}"

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


def _check_or_write_manifest(manifest_path: Path, *, bin_width, max_height,
                             product_version, year) -> dict:
    """
    Version-level provenance: one canonical (bin_width, max_height) per
    `products/diversity_indices/{year}/{product_version}/` tree.

    On first call (no manifest yet): write one. On later calls: assert the
    params match what was written, otherwise refuse — silently mixing two
    parameter sets in the same product tree is the bug we're guarding against.
    """
    manifest_path = Path(manifest_path).expanduser()
    record = {
        'bin_width': bin_width,
        'max_height': max_height,
        'product_version': product_version,
        'year': year,
    }
    if manifest_path.exists():
        saved = json.loads(manifest_path.read_text())
        for k in ('bin_width', 'max_height'):
            if saved.get(k) != record[k]:
                raise RuntimeError(
                    f"Manifest {manifest_path} has {k}={saved.get(k)}, "
                    f"refusing to mix with {k}={record[k]}. "
                    f"Pick a different save_dir for the new params."
                )
        return saved
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    record['created_at'] = datetime.now().isoformat(timespec='seconds')
    manifest_path.write_text(json.dumps(record, indent=2))
    print(f"Wrote manifest: {manifest_path}")
    return record


def _write_diversity_gtiff(vrt_path, gtif_path, *, chunk_size, max_workers,
                           bin_width, max_height):
    """
    Core compute loop: 100-band RH VRT → 4-band float32 GeoTIFF
    (fhd, enl1d, enl2d, cr). Uses a bounded-inflight ProcessPoolExecutor so
    results don't queue up and pin worker memory.
    """
    vrt_path = str(Path(vrt_path).expanduser())
    gtif_path = Path(gtif_path).expanduser()
    gtif_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(vrt_path, "r") as src:
        ny, nx = src.height, src.width
        crs, transform = src.crs, src.transform
        n_bands = src.count
        print(f"VRT: {nx}x{ny}, {n_bands} bands, dtype={src.dtypes[0]}")
    assert n_bands == 100, f"Expected 100 bands, got {n_bands}"

    out_profile = {
        "driver": "GTiff", "dtype": "float32",
        "height": ny, "width": nx, "count": 4,
        "crs": crs, "transform": transform,
        "nodata": INDICES_NODATA, "compress": None,
        "tiled": True, "blockxsize": 512, "blockysize": 512,
    }

    work_items = []
    for row_off in range(0, ny, chunk_size):
        for col_off in range(0, nx, chunk_size):
            h = min(chunk_size, ny - row_off)
            w = min(chunk_size, nx - col_off)
            work_items.append((vrt_path, col_off, row_off, w, h, bin_width, max_height))

    print(f"Processing {len(work_items)} chunks with {max_workers} processes")
    print(f"Estimated peak RAM: ~{max_workers * 100 * chunk_size**2 * 4 / 1e9:.1f} GB")

    monitor = ProgressMonitor(total_tiles=len(work_items), interval=5.0)
    max_inflight = max_workers * 2

    with rasterio.open(gtif_path, "w", **out_profile) as dst:
        monitor.start()
        try:
            with ProcessPoolExecutor(max_workers=max_workers) as pool:
                work_iter = iter(work_items)
                inflight = set()
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
    return gtif_path


def _compute_diversity_maps(vrt_path, save_dir, name, *, chunk_size, max_workers,
                            bin_width, max_height, keep_geotiff=False) -> Path:
    """
    Write {save_dir}/geotiff/{name}.tif then {save_dir}/cog/{name}.cog.tif.
    Returns the COG path. Resume-friendly:
      - COG already exists           → no-op
      - only GTiff exists            → only run cog_translate
      - neither exists               → compute everything

    After a successful cog_translate, the intermediate GeoTIFF is removed
    (keep_geotiff=True opts out — useful if you want to re-run cog_translate
    with different settings without recomputing).
    """
    save_dir = Path(save_dir).expanduser()
    gtif_path = save_dir / 'geotiff' / f'{name}.tif'
    cog_path  = save_dir / 'cog'     / f'{name}.tif'
    cog_path.parent.mkdir(parents=True, exist_ok=True)

    if cog_path.exists():
        print(f"Already done: {cog_path}")
        return cog_path
    if not gtif_path.exists():
        _write_diversity_gtiff(
            vrt_path, gtif_path,
            chunk_size=chunk_size, max_workers=max_workers,
            bin_width=bin_width, max_height=max_height,
        )
    to_cog(gtif_path, cog_path)
    if not keep_geotiff:
        gtif_path.unlink(missing_ok=True)
        print(f"Removed intermediate: {gtif_path}")
    return cog_path


def create_tile_diversity_maps(save_dir, tile_id, year,
                               product_version='bias_corrected',
                               product_format='geotiff',
                               vsm_tile_root=None, q_idx=1, vrt_path=None,
                               chunk_size=512, max_workers=8,
                               bin_width=5, max_height=MAX_HEIGHT_METERS,
                               keep_geotiff=False, **kwargs):
    """
    Per-tile diversity indices (fhd / enl1d / enl2d / cr).

    Output layout:
        {save_dir}/geotiff/{tile_id}.tif       intermediate (removed after COG by default)
        {save_dir}/cog/{tile_id}.cog.tif       final (returned)

    VRT inputs (only used if vrt_path is None):
        {vsm_tile_root}/cog/{tile_id}/RH*_Q{q_idx}.tif      source bands
        {vsm_tile_root}/vrt/{tile_id}_Q{q_idx}.vrt          generated VRT

    Manifest at {save_dir}/../PARAMS.json pins (bin_width, max_height) for the
    version — re-running with mismatched params raises rather than silently
    overwriting.
    """
    save_dir = Path(save_dir).expanduser()
    save_dir.mkdir(parents=True, exist_ok=True)

    _check_or_write_manifest(
        save_dir.parent / 'PARAMS.json',
        bin_width=bin_width, max_height=max_height,
        product_version=product_version, year=year,
    )

    if vrt_path is None:
        vsm_tile_root = Path(vsm_tile_root).expanduser()
        tile_dir = vsm_tile_root / product_format / tile_id
        vrt_path = vsm_tile_root / 'vrt' / f'{tile_id}_Q{q_idx}.vrt'
        create_vrt(tile_dir, vrt_path)

    return _compute_diversity_maps(
        vrt_path, save_dir, name=tile_id,
        chunk_size=chunk_size, max_workers=max_workers,
        bin_width=bin_width, max_height=max_height,
        keep_geotiff=keep_geotiff,
    )


def create_global_diversity_maps(save_dir, tif_dir=None, vrt_path=None,
                                 chunk_size=512, max_workers=8,
                                 bin_width=5, max_height=150,
                                 keep_geotiff=False, **kwargs):
    """
    Global 1 km diversity-indices mosaic. Filename encodes parameters so
    multiple (bin, max) variants can coexist for comparison work.

    Output layout:
        {save_dir}/geotiff/diversity_maps_bin{B}_max{H}.tif
        {save_dir}/cog/diversity_maps_bin{B}_max{H}.cog.tif      (returned)
    """
    save_dir = Path(save_dir).expanduser()
    save_dir.mkdir(parents=True, exist_ok=True)
    name = f'diversity_maps_bin{bin_width}_max{max_height}'

    if vrt_path is None:
        tif_dir = Path(tif_dir).expanduser()
        existing = list(tif_dir.glob('*.vrt'))
        vrt_path = existing[0] if existing else tif_dir / f'{name}.vrt'
        if not existing:
            create_vrt(tif_dir, vrt_path)

    return _compute_diversity_maps(
        vrt_path, save_dir, name=name,
        chunk_size=chunk_size, max_workers=max_workers,
        bin_width=bin_width, max_height=max_height,
        keep_geotiff=keep_geotiff,
    )


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

    
    
