"""Build a stable-forest mask per S2 tile from JRC TMF for evaluation prep.

Use case: comparing LVIS Gabon 2016 against VSM-from-GEDI 2020 (or any two
acquisitions across a year range) only makes sense on pixels whose canopy
was structurally undisturbed across both. JRC Tropical Moist Forest
(AnnualChanges) gives a per-year class 1..6 at ~30 m; stable = class 1
(undisturbed TMF) every Dec from `year_start` through `year_end`.

Per tile in the meta CSV:
  1. Pull the stable-forest image clipped to the tile bbox via GEE
     getDownloadURL at native scale, EPSG:4326 (one small request).
  2. Reproject (nearest-neighbour) to the matching VSM RH98_Q1.tif grid
     (10 m, local UTM) so the mask is a drop-in for sampling in on_lvis
     and on_gedi. Falls back to native if the VSM ref tile is missing.
  3. Write a uint8 GeoTIFF: 1 = stable forest, 0 = not, 255 = nodata.

Limitation: JRC TMF only covers the pan-tropical moist belt. Outside it
(boreal / temperate / tropical dry), every pixel returns 0 — use a
different dataset (e.g. Hansen GFC) for those biomes.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import time

import ee
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import requests
from rasterio.warp import Resampling, reproject

from download.core.utils import authenticate


def build_stable_forest_image(
        year_start: int,
        year_end: int,
        tmf_asset: str,
        undisturbed_class: int) -> ee.Image:
    """JRC TMF stable-forest image: pixel = 1 iff class == undisturbed_class
    every Dec in `[year_start, year_end]`.

    AnnualChanges is a per-geographic-tile mosaic with one band per year
    (`Dec1990`..`Dec<last>`). `.mosaic()` flattens it so a single bbox
    query covers everything.
    """
    coll = ee.ImageCollection(tmf_asset).mosaic()
    stable = coll.select(f'Dec{year_start}').eq(undisturbed_class)
    for y in range(year_start + 1, year_end + 1):
        stable = stable.And(coll.select(f'Dec{y}').eq(undisturbed_class))
    return stable.rename('stable_forest').toUint8()


def _download_tile_native(img: ee.Image, tile_geom_4326,
                          scale_m: int, retries: int,
                          retry_backoff_s: int) -> bytes:
    """getDownloadURL for the image clipped to the tile bbox at native
    resolution in EPSG:4326. Returns the raw GeoTIFF bytes."""
    minx, miny, maxx, maxy = tile_geom_4326.bounds
    region = ee.Geometry.Rectangle(
        [minx, miny, maxx, maxy], proj='EPSG:4326', geodesic=False)
    url = img.getDownloadURL({
        'region': region,
        'scale': scale_m,
        'crs': 'EPSG:4326',
        'format': 'GEO_TIFF',
    })
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(url, stream=True, timeout=300)
            r.raise_for_status()
            return r.content
        except Exception as e:
            last_err = e
            if attempt < retries:
                time.sleep(retry_backoff_s * attempt)
    raise RuntimeError(f'GEE fetch failed after {retries} tries: {last_err}')


def _align_to_vsm_grid(src_bytes: bytes, ref_tif: Path,
                       out_path: Path, band_desc: str) -> None:
    """Reproject the EPSG:4326 native-res mask onto the VSM tile's 10 m
    UTM grid with nearest-neighbour. uint8, 0/1 with 255 as nodata."""
    tmp_path = out_path.with_suffix('.src.tif')
    tmp_path.write_bytes(src_bytes)
    try:
        with rasterio.open(tmp_path) as src, rasterio.open(ref_tif) as ref:
            dst_arr = np.full((ref.height, ref.width), 255, dtype=np.uint8)
            reproject(
                source=rasterio.band(src, 1),
                destination=dst_arr,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=ref.transform,
                dst_crs=ref.crs,
                resampling=Resampling.nearest,
                dst_nodata=255,
            )
        profile = {
            'driver': 'GTiff',
            'height': dst_arr.shape[0],
            'width': dst_arr.shape[1],
            'count': 1,
            'dtype': 'uint8',
            'crs': ref.crs,
            'transform': ref.transform,
            'nodata': 255,
            'compress': 'deflate',
            'tiled': True,
            'blockxsize': 512,
            'blockysize': 512,
        }
        with rasterio.open(out_path, 'w', **profile) as dst:
            dst.write(dst_arr, 1)
            dst.set_band_description(1, band_desc)
    finally:
        tmp_path.unlink(missing_ok=True)


def build_stable_forest_mask(
        meta_file: str,
        s2_grid_file: str,
        vsm_ref_tile_dir: str,
        save_dir: str,
        year_start: int = 2016,
        year_end: int = 2020,
        tmf_asset: str = 'projects/JRC/TMF/v1_2023/AnnualChanges',
        tmf_undisturbed_class: int = 1,
        native_scale_m: int = 30,
        vsm_ref_filename: str = 'RH98_Q1.tif',
        max_workers: int = 4,
        retries: int = 3,
        retry_backoff_s: int = 5,
        rewrite: bool = False,
        **kwargs) -> None:
    """Build a 0/1 stable-forest GeoTIFF per S2 tile listed in `meta_file`.

    Args:
        meta_file: CSV with a `Tile name` column (e.g. meta_lvis_profile.csv).
        s2_grid_file: parquet with `Name` + `geometry` columns covering all
            tiles in the meta CSV (EPSG:4326, or with a recoverable CRS).
        vsm_ref_tile_dir: dir with one subdir per S2 tile holding the VSM
            RH98 GeoTIFF (its CRS/transform/dims drive the output grid).
            Tiles whose ref TIF is missing fall back to the native 30 m
            EPSG:4326 raster so the run isn't blocked.
        save_dir: output dir for `<tile>.tif` masks.
        year_start, year_end: inclusive year range. JRC TMF AnnualChanges
            covers 1990..(asset year - 1); pick a range covered by the
            asset id.
        tmf_asset: AnnualChanges GEE collection. Bump the v_<year> suffix
            to roll forward to a newer release.
        tmf_undisturbed_class: JRC TMF class id for undisturbed TMF (1 in
            the v1 schema). Pass a different int if a future schema
            renumbers.
        native_scale_m: GEE `scale` argument for the per-tile fetch. JRC
            TMF is 30 m native; smaller values upsample server-side and
            inflate the response.
        vsm_ref_filename: filename inside each `vsm_ref_tile_dir/<tile>/`
            to use as the grid reference. Defaults to `RH98_Q1.tif`; any
            VSM band works since they share the per-tile grid.
        max_workers: parallel per-tile fetches. GEE getDownloadURL is
            rate-limited; 4 is a safe default.
        retries, retry_backoff_s: per-tile fetch retry policy with linear
            backoff.
        rewrite: redo tiles whose output already exists.
    """
    meta_file = Path(meta_file).expanduser()
    s2_grid_file = Path(s2_grid_file).expanduser()
    vsm_ref_tile_dir = Path(vsm_ref_tile_dir).expanduser()
    save_dir = Path(save_dir).expanduser()
    save_dir.mkdir(parents=True, exist_ok=True)

    meta = pd.read_csv(meta_file)
    tiles = meta['Tile name'].astype(str).tolist()
    print(f'Loaded {len(tiles)} tiles from {meta_file}')

    s2 = gpd.read_parquet(s2_grid_file, columns=['Name', 'geometry']).to_crs(4326)
    s2 = s2[s2['Name'].isin(tiles)].drop_duplicates(subset='Name')
    missing = sorted(set(tiles) - set(s2['Name']))
    if missing:
        print(f'WARN: {len(missing)} tiles not in S2 grid: {missing[:5]}...')
    tile_geoms = dict(zip(s2['Name'], s2.geometry))

    authenticate()
    stable_img = build_stable_forest_image(
        year_start, year_end, tmf_asset, tmf_undisturbed_class)
    band_desc = f'stable_TMF_class{tmf_undisturbed_class}_{year_start}_{year_end}'

    def _process(tile_name: str) -> str:
        out_path = save_dir / f'{tile_name}.tif'
        if out_path.exists() and not rewrite:
            return f'{tile_name}: exists, skipped'
        src_bytes = _download_tile_native(
            stable_img, tile_geoms[tile_name],
            native_scale_m, retries, retry_backoff_s)
        ref_tif = vsm_ref_tile_dir / tile_name / vsm_ref_filename
        if ref_tif.exists():
            _align_to_vsm_grid(src_bytes, ref_tif, out_path, band_desc)
            return f'{tile_name}: aligned to VSM 10 m grid -> {out_path.name}'
        out_path.write_bytes(src_bytes)
        return (f'{tile_name}: VSM ref missing ({ref_tif}), wrote native '
                f'{native_scale_m} m EPSG:4326 -> {out_path.name}')

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(_process, n): n for n in tile_geoms.keys()}
        for fut in as_completed(futs):
            name = futs[fut]
            try:
                print(fut.result())
            except Exception as e:
                print(f'{name}: FAILED -> {e}')

    print(f'\nDone. Masks in {save_dir}')


# ============================================================================
# Hydra entrypoint
# ============================================================================
import hydra
from config.loader import register
from config.runner import run_cli

register(
    Path(__file__).resolve().parents[1] / 'config' / 'eval' / 'config.yaml',
    section='forest_mask',
    default_run='build_stable_forest_mask',
)


@hydra.main(config_name='no_log', version_base='1.2', config_path='../config/base')
def main(cfg):
    run_cli(cfg)


if __name__ == '__main__':
    main()
