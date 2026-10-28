"""Monthly per-pixel median of the per-image VSM time series of one tile-year.

Input: the output of `run_deploy.sh predict_timeseries` for one tile and year,

    <tile_dir>/RH25_Q1.cog.tif, RH98_Q1.cog.tif   one band per image (or the
                                                  uncompressed RH*.tif)
    <tile_dir>/images.csv                         band -> date, s2_id

The same acquisition can be in the store twice, once per processing run (the
2024 reprocessing of the archive, e.g. ..._20240810T232727 next to
..._20230115T141952). Such duplicates are dropped first, keeping the newest
processing, so a date is not counted twice.

This is the yearly product's aggregation (nanmedian over cloud/snow-masked
images) applied per calendar month. Output, next to the input, as COGs:

    <var>_monthly_median.cog.tif   12 bands, band m = month m (description
                                   YYYY-MM), dm int16, same nodata; months
                                   without a valid image are nodata
    monthly_valid_images.cog.tif   12 bands, uint8: images with a valid value
                                   per pixel and month

Reads one 512 x 512 block of all images at a time, so memory stays small.

    python -m tools.run run=vsm_monthly_median run.year=2023 run.tile=48PWV
"""
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from osgeo import gdal
from rasterio.windows import Window

gdal.UseExceptions()

COG_OPTIONS = ['COMPRESS=ZSTD', 'PREDICTOR=YES', 'BLOCKSIZE=512', 'OVERVIEWS=AUTO',
               'RESAMPLING=AVERAGE', 'NUM_THREADS=ALL_CPUS', 'BIGTIFF=IF_SAFER']


def unique_acquisitions(images: pd.DataFrame) -> pd.DataFrame:
    """Keep one band per acquisition: the newest processing of each S2 id.

    s2_id is <mission>_MSIL2A_<sensing time>_R<orbit>_T<tile>_<processing time>.
    Without ids (older h5 files) every band is kept.
    """
    if images['s2_id'].isna().all() or (images['s2_id'].astype(str) == '').all():
        return images
    parts = images['s2_id'].str.rsplit('_', n=1, expand=True)
    images = images.assign(acquisition=parts[0], processed=parts[1])
    keep = images.sort_values('processed').groupby('acquisition').tail(1)
    return keep.sort_values('band').drop(columns=['acquisition', 'processed'])


def _input_path(tile_dir: Path, var: str) -> Path:
    for name in (f'{var}.cog.tif', f'{var}.tif'):
        if (tile_dir / name).exists():
            return tile_dir / name
    raise FileNotFoundError(f'no {var}.cog.tif or {var}.tif in {tile_dir}')


def _to_cog(src: Path, dst: Path):
    gdal.Translate(str(dst), str(src), format='COG', creationOptions=COG_OPTIONS)
    src.unlink()


def _monthly_median(in_fp: Path, out_fp: Path, count_fp: Path, months: dict, year: int,
                    block: int = 512):
    """months: {month: [1-based band numbers]}. Writes count_fp too when it is not None."""
    with rasterio.open(in_fp) as src:
        nodata = src.nodata
        profile = dict(driver='GTiff', width=src.width, height=src.height, count=12,
                       crs=src.crs, transform=src.transform, tiled=True,
                       blockxsize=block, blockysize=block, compress='zstd', predictor=2,
                       BIGTIFF='IF_SAFER')
        tmp_fp = out_fp.with_suffix('.tmp.tif')
        dst = rasterio.open(tmp_fp, 'w', dtype='int16', nodata=nodata, **profile)
        if count_fp is not None:
            tmp_count_fp = count_fp.with_suffix('.tmp.tif')
            cnt = rasterio.open(tmp_count_fp, 'w', dtype='uint8', nodata=None, **profile)
        for m in range(1, 13):
            dst.set_band_description(m, f'{year}-{m:02d}')
            if count_fp is not None:
                cnt.set_band_description(m, f'{year}-{m:02d}')

        bands = sorted({b for bs in months.values() for b in bs})
        pos = {b: i for i, b in enumerate(bands)}
        for row in range(0, src.height, block):
            for col in range(0, src.width, block):
                win = Window(col, row, min(block, src.width - col), min(block, src.height - row))
                data = src.read(bands, window=win).astype(np.float32)
                data[data == nodata] = np.nan
                out = np.full((12, win.height, win.width), nodata, dtype=np.int16)
                n_valid = np.zeros((12, win.height, win.width), dtype=np.uint8)
                for m, bs in months.items():
                    sel = data[[pos[b] for b in bs]]
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore', RuntimeWarning)  # all-NaN pixels
                        med = np.nanmedian(sel, axis=0)
                    ok = np.isfinite(med)
                    out[m - 1][ok] = np.round(med[ok]).astype(np.int16)
                    n_valid[m - 1] = np.isfinite(sel).sum(axis=0)
                dst.write(out, window=win)
                if count_fp is not None:
                    cnt.write(n_valid, window=win)
        dst.close()
        _to_cog(tmp_fp, out_fp)
        if count_fp is not None:
            cnt.close()
            _to_cog(tmp_count_fp, count_fp)


def vsm_monthly_median(tile_dir: str, year: int, variables=('RH25_Q1', 'RH98_Q1'), **kwargs):
    tile_dir = Path(tile_dir).expanduser()
    images = pd.read_csv(tile_dir / 'images.csv', dtype={'s2_id': str})
    unique = unique_acquisitions(images)
    dates = pd.to_datetime(unique['date'])
    unique = unique[dates.dt.year == int(year)]
    months = {m: g['band'].tolist() for m, g in unique.groupby(pd.to_datetime(unique['date']).dt.month)}
    print(f'{tile_dir}: {len(images)} bands, {len(unique)} unique acquisitions in {year}; '
          f'images per month: { {m: len(b) for m, b in months.items()} }')

    for i, var in enumerate(variables):
        t0 = time.time()
        out_fp = tile_dir / f'{var}_monthly_median.cog.tif'
        count_fp = tile_dir / 'monthly_valid_images.cog.tif' if i == 0 else None  # same masks for every variable
        _monthly_median(_input_path(tile_dir, var), out_fp, count_fp, months, int(year))
        print(f'{out_fp.name}: {time.time() - t0:.0f} s')
