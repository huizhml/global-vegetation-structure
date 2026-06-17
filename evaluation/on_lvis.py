"""
Partition raw LVIS L2 Geolocated Surface Elevation TXT files into one
geoparquet per Sentinel-2 tile so we can validate the VSM RH metrics
against airborne lidar at the footprint level.

Each LVIS TXT has a `#`-prefixed header block followed by whitespace-separated
columns. The relevant fields are GLON/GLAT (ground centroid, used as the shot
geometry) and RH10..RH100. HLON/HLAT/ZH and CHANNEL_ZG carry the -999 sentinel
when the highest-mode return wasn't detected; we coerce those to NaN.
"""
import json
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import dask
import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import rasterio
from dask.distributed import Client, LocalCluster
from scipy.spatial import cKDTree
from scipy.stats import pearsonr
from sklearn.metrics import r2_score

from const import FIGURE_SIZES, FONT_SIZES
import math

import matplotlib.pyplot as plt
import seaborn as sns
from evaluation.plots import hexbin_density_grid, hexbin_density_plot, sci_notation
from evaluation.structure_partial_correlation import partial_correlation


# Reference only: the R1808 column layout (42 cols). NOT used for parsing —
# `_parse_lvis_txt` reads the actual column names from each file's `#` header
# instead, since later releases (e.g. 2023 Gabon) append trailing columns.
LVIS_COLUMNS = [
    'LFID', 'SHOTNUMBER', 'TIME',
    'GLON', 'GLAT', 'ZG',
    'HLON', 'HLAT', 'ZH',
    'TLON', 'TLAT', 'ZT',
    'RH10', 'RH15', 'RH20', 'RH25', 'RH30', 'RH35', 'RH40', 'RH45',
    'RH50', 'RH55', 'RH60', 'RH65', 'RH70', 'RH75', 'RH80', 'RH85',
    'RH90', 'RH95', 'RH96', 'RH97', 'RH98', 'RH99', 'RH100',
    'AZIMUTH', 'INCIDENTANGLE', 'RANGE', 'COMPLEXITY',
    'CHANNEL_L1B', 'CHANNEL_ZG', 'CHANNEL_RH',
]
# Fields that use -999 as the "not detected" sentinel.
NODATA_COLUMNS = ['HLON', 'HLAT', 'ZH', 'CHANNEL_ZG']

# RH metrics LVIS reports (matches the VSM RH<NN>_Q1.tif naming on disk).
RH_METRICS = [
    'RH10', 'RH15', 'RH20', 'RH25', 'RH30', 'RH35', 'RH40', 'RH45',
    'RH50', 'RH55', 'RH60', 'RH65', 'RH70', 'RH75', 'RH80', 'RH85',
    'RH90', 'RH95', 'RH96', 'RH97', 'RH98', 'RH99', 'RH100',
]


# LVIS L2A "Footprint Cover and Vertical Profile Metrics" CSV columns
# (AfriSAR_LVIS_Footprint_Cover_v0100 release, ORNL DAAC). The header line
# in the CSV has mixed case and a stray leading space before err_cov; we
# normalize to lowercase here and strip whitespace on read.
LVIS_METRICS_COLUMNS = [
    'lfid', 'shotnumber', 'glat', 'glon',
    'totwave', 'groundtot',
    'lai', 'ccover',
    'vfp00', 'vfp10', 'vfp20', 'vfp30',
    'fhd', 'err_rg', 'err_cov',
]
# Per-footprint columns we carry through the partition. Geometry is built
# from glon/glat, so they get dropped from the stored columns.
LVIS_METRICS_KEEP = ['shotnumber', 'lai', 'ccover',
                     'vfp00', 'vfp10', 'vfp20', 'vfp30', 'fhd']


def _read_lvis_header(path: Path) -> list:
    """Return the column names from the LVIS TXT header.

    LVIS ships the column list as the *last* `#`-prefixed comment line
    before the data block. Parse it instead of hardcoding: releases vary
    in their trailing columns (the 2023 Gabon files append `SENSITIVITY`
    and `CHANNEL_ZT` after `COMPLEXITY`, giving 43 columns vs the 42 in
    the R1808 layout). Passing a too-short `names=` to read_csv makes
    pandas silently consume the leftmost column(s) as the index, shifting
    every name one slot right — which is exactly what put GLAT into GLON
    and ZG into GLAT and scattered Gabon shots across the globe.
    """
    last = None
    with open(path) as fh:
        for line in fh:
            if line.startswith('#'):
                last = line
            else:
                break
    if last is None:
        raise ValueError(f'{path.name}: no `#` header line found')
    return last.lstrip('#').split()


def _parse_lvis_txt(path: Path) -> pd.DataFrame:
    cols = _read_lvis_header(path)
    df = pd.read_csv(path, comment='#', sep=r'\s+', header=None, names=cols)
    for col in NODATA_COLUMNS:
        if col in df.columns:
            df[col] = df[col].mask(df[col] == -999)
    df['source_file'] = path.name
    return df


@dask.delayed
def _partition_one_txt(path: Path, s2_grid: gpd.GeoDataFrame,
                       staging_dir: Path, rewrite: bool) -> dict:
    """Parse one TXT, sjoin to the S2 grid, write one staged parquet per
    tile that the file contributes shots to. Returns {tile: staged_path}."""
    done_marker = staging_dir / '_done' / f'{path.stem}.json'
    if done_marker.exists() and not rewrite:
        return json.loads(done_marker.read_text())

    df = _parse_lvis_txt(path)
    gdf = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df['GLON'], df['GLAT']), crs='EPSG:4326')
    joined = gpd.sjoin(gdf, s2_grid, how='inner', predicate='intersects')
    joined = joined.drop(columns=['index_right'])
    # A footprint can fall on the seam between two MGRS tiles; keep only one
    # assignment so we don't double-count shots in downstream stats.
    joined = joined.drop_duplicates(subset='SHOTNUMBER', keep='first')

    written = {}
    for tile, sub in joined.groupby('Name', sort=False):
        out = staging_dir / tile / f'{path.stem}.parquet'
        out.parent.mkdir(parents=True, exist_ok=True)
        sub.to_parquet(out)
        written[tile] = str(out)

    done_marker.parent.mkdir(parents=True, exist_ok=True)
    done_marker.write_text(json.dumps(written))
    return written


@dask.delayed
def _merge_tile(tile: str, staged_files: list, save_dir: Path,
                rewrite: bool) -> str:
    out = save_dir / f'{tile}.parquet'
    if out.exists() and not rewrite:
        return f'{tile}: exists, skipped'
    # Stream-write: open ParquetWriter from the first shard's schema (which
    # carries the geoparquet kv-metadata that lets gpd.read_parquet recover
    # the geometry column), then append the remaining shards one-by-one.
    # Avoids materializing every shard into RAM the way pd.concat does.
    first = pq.read_table(staged_files[0])
    n_rows = first.num_rows
    with pq.ParquetWriter(out, first.schema) as writer:
        writer.write_table(first)
        del first
        for p in staged_files[1:]:
            t = pq.read_table(p)
            n_rows += t.num_rows
            writer.write_table(t)
            del t
    return f'{tile}: {n_rows} shots -> {out.name}'


def partition_lvis_by_s2_tile(
        txt_dir: str,
        s2_grid_file: str,
        save_dir: str,
        n_workers: int = 8,
        rewrite: bool = False,
        **kwargs):
    """Convert raw LVIS L2 TXT files into one geoparquet per Sentinel-2 tile.

    Args:
        txt_dir: directory holding LVIS2_*.TXT granules.
        s2_grid_file: Sentinel-2 grid parquet with `Name` + `geometry` columns.
        save_dir: destination for `<tile>.parquet` outputs.
        n_workers: Dask LocalCluster worker count for parallel TXT parsing.
        rewrite: re-emit per-tile parquet even if it already exists.
    """
    txt_dir = Path(txt_dir).expanduser()
    save_dir = Path(save_dir).expanduser()
    save_dir.mkdir(parents=True, exist_ok=True)
    staging_dir = save_dir / '_staging'
    staging_dir.mkdir(exist_ok=True)

    txt_files = sorted(txt_dir.glob('*.TXT'))
    if not txt_files:
        raise FileNotFoundError(f'No .TXT files under {txt_dir}')
    print(f'Found {len(txt_files)} LVIS TXT files in {txt_dir}')

    s2_grid = gpd.read_parquet(
        Path(s2_grid_file).expanduser(), columns=['Name', 'geometry'])
    s2_grid = s2_grid.to_crs('EPSG:4326')
    print(f'Loaded {len(s2_grid)} S2 tiles from {s2_grid_file}')

    with Client(LocalCluster(n_workers=n_workers, threads_per_worker=1)) as client:
        print(client.dashboard_link)
        # Phase 1: parse + sjoin + stage per (txt, tile). Skips TXTs whose
        # _done marker already exists so a rerun after a crash only redoes
        # missing granules.
        partition_tasks = [
            _partition_one_txt(f, s2_grid, staging_dir, rewrite)
            for f in txt_files]
        per_file_results = dask.compute(*partition_tasks)

        # Group staged parquet paths by tile.
        per_tile = defaultdict(list)
        for written in per_file_results:
            for tile, path in written.items():
                per_tile[tile].append(path)
        print(f'Touched {len(per_tile)} S2 tiles across all granules')

        # Phase 2: concat staged chunks into one geoparquet per tile.
        merge_tasks = [_merge_tile(tile, paths, save_dir, rewrite)
                       for tile, paths in per_tile.items()]
        for msg in dask.compute(*merge_tasks):
            print(msg)

    print(f'Done. Per-tile geoparquets in {save_dir}')


def _parse_lvis_metrics_csv(path: Path) -> pd.DataFrame:
    # The CSV header has a stray leading space (" err_cov") and uppercase
    # LAI; skipinitialspace + lowercase normalize both. We don't pass `names`
    # because the file ships its own header line.
    df = pd.read_csv(path, skipinitialspace=True)
    df.columns = df.columns.str.strip().str.lower()
    missing = set(LVIS_METRICS_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f'{path.name}: missing expected columns {sorted(missing)}')
    df['source_file'] = path.name
    return df


@dask.delayed
def _partition_one_metrics_csv(path: Path, s2_grid: gpd.GeoDataFrame,
                               staging_dir: Path, rewrite: bool) -> dict:
    """Parse one l2a_metrics CSV, sjoin to the S2 grid, write one staged
    parquet per tile the file contributes shots to. Mirrors
    `_partition_one_txt` for the AfriSAR Cover product."""
    done_marker = staging_dir / '_done' / f'{path.stem}.json'
    if done_marker.exists() and not rewrite:
        return json.loads(done_marker.read_text())

    df = _parse_lvis_metrics_csv(path)
    gdf = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df['glon'], df['glat']), crs='EPSG:4326')
    joined = gpd.sjoin(gdf, s2_grid, how='inner', predicate='intersects')
    joined = joined.drop(columns=['index_right'])
    # Same de-dup logic as the RH partitioner — a shot on the tile seam
    # would otherwise count in both tiles' stats.
    joined = joined.drop_duplicates(subset='shotnumber', keep='first')

    written = {}
    for tile, sub in joined.groupby('Name', sort=False):
        out = staging_dir / tile / f'{path.stem}.parquet'
        out.parent.mkdir(parents=True, exist_ok=True)
        sub.to_parquet(out)
        written[tile] = str(out)

    done_marker.parent.mkdir(parents=True, exist_ok=True)
    done_marker.write_text(json.dumps(written))
    return written


def partition_lvis_metrics_by_s2_tile(
        csv_dir: str,
        s2_grid_file: str,
        save_dir: str,
        n_workers: int = 8,
        rewrite: bool = False,
        **kwargs):
    """Convert LVIS L2A footprint-metrics CSVs into one geoparquet per S2 tile.

    Same flow as `partition_lvis_by_s2_tile` but for the AfriSAR
    `*_l2a_metrics_*.csv` files (carrying `fhd`, `lai`, `ccover`, `vfp*`).
    A separate partition run is cleaner than joining onto the RH parquets
    on `shotnumber` because the two products live in distinct DAACs (NSIDC
    L2 vs ORNL DAAC L2A) and may not be a perfect 1:1 shot match.

    Args:
        csv_dir: directory holding `*_l2a_metrics_*.csv` files.
        s2_grid_file: Sentinel-2 grid parquet with `Name` + `geometry`.
        save_dir: destination for `<tile>.parquet` outputs.
        n_workers: Dask LocalCluster worker count.
        rewrite: re-emit per-tile parquet even if it already exists.
    """
    csv_dir = Path(csv_dir).expanduser()
    save_dir = Path(save_dir).expanduser()
    save_dir.mkdir(parents=True, exist_ok=True)
    staging_dir = save_dir / '_staging'
    staging_dir.mkdir(exist_ok=True)

    csv_files = sorted(csv_dir.glob('*_l2a_metrics_*.csv'))
    if not csv_files:
        raise FileNotFoundError(f'No *_l2a_metrics_*.csv files under {csv_dir}')
    print(f'Found {len(csv_files)} LVIS l2a_metrics CSVs in {csv_dir}')

    s2_grid = gpd.read_parquet(
        Path(s2_grid_file).expanduser(), columns=['Name', 'geometry'])
    s2_grid = s2_grid.to_crs('EPSG:4326')
    print(f'Loaded {len(s2_grid)} S2 tiles from {s2_grid_file}')

    with Client(LocalCluster(n_workers=n_workers, threads_per_worker=1)) as client:
        print(client.dashboard_link)
        partition_tasks = [
            _partition_one_metrics_csv(f, s2_grid, staging_dir, rewrite)
            for f in csv_files]
        per_file_results = dask.compute(*partition_tasks)

        per_tile = defaultdict(list)
        for written in per_file_results:
            for tile, path in written.items():
                per_tile[tile].append(path)
        print(f'Touched {len(per_tile)} S2 tiles across all CSVs')

        merge_tasks = [_merge_tile(tile, paths, save_dir, rewrite)
                       for tile, paths in per_tile.items()]
        for msg in dask.compute(*merge_tasks):
            print(msg)

    print(f'Done. Per-tile geoparquets in {save_dir}')


def _row_col_for_points(tif_path: Path, gdf_in_raster_crs: gpd.GeoDataFrame):
    """Project the points into one raster's (row, col) grid. All VSM RH
    rasters in a tile share the same grid, so we do this once per tile."""
    with rasterio.open(tif_path) as src:
        xs = gdf_in_raster_crs.geometry.x.to_numpy()
        ys = gdf_in_raster_crs.geometry.y.to_numpy()
        rows, cols = rasterio.transform.rowcol(src.transform, xs, ys)
        h, w = src.height, src.width
    rows = np.asarray(rows)
    cols = np.asarray(cols)
    valid = (rows >= 0) & (rows < h) & (cols >= 0) & (cols < w)
    return rows, cols, valid


def _sample_raster_at_rowcol(tif_path: Path, rows: np.ndarray,
                             cols: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Read one RH raster and return its values at the given (row, col)
    indices. NaN where the index is out of bounds or the pixel is nodata.
    VSM stores height in decimeters; we divide by 10 to return meters."""
    with rasterio.open(tif_path) as src:
        arr = src.read(1, masked=True).astype(np.float32).filled(np.nan)
    out = np.full(len(rows), np.nan, dtype=np.float32)
    out[valid] = arr[rows[valid], cols[valid]]
    return out / 10.0


def _per_metric_agg(mode: str, rh_metrics: list) -> dict:
    """Map each RH metric to its within-pixel aggregator. `mean_max` splits
    on RH95: upper percentiles use `max` (closer to "tallest return in this
    pixel"), lower percentiles use `mean`."""
    if mode == 'mean':
        return {rh: 'mean' for rh in rh_metrics}
    if mode == 'max':
        return {rh: 'max' for rh in rh_metrics}
    if mode == 'mean_max':
        return {rh: ('max' if int(rh[2:]) >= 95 else 'mean') for rh in rh_metrics}
    raise ValueError(f'Unknown agg mode {mode!r}')


def extract_lvis_vsm_pairs(
        lvis_dir: str,
        vsm_tile_dir: str,
        save_dir: str,
        rh_metrics: list = None,
        aggregate_within_pixel: str = 'mean',
        rewrite: bool = False,
        **kwargs):
    """For each tile, sample each VSM RH raster at the LVIS shot locations
    and write a per-tile parquet of paired heights.

    Why aggregate? LVIS shot centers are spaced ~10m along-track and the
    same area is often re-flown, so several shots typically fall in the
    same 10m VSM pixel. Leaving them as raw pairs pseudo-replicates the VSM
    value (one prediction repeated against varying LVIS) and inflates `n`.

    Note that LVIS RH metrics are *percentile* heights of a single shot's
    waveform; the mean of two shots' RH98 is not the RH98 of the combined
    footprints. So `mean` is a pragmatic aggregator (common in the
    LVIS/GEDI-vs-CHM literature) but not strictly physical, especially for
    upper-canopy percentiles where `max` is more defensible.

    Args:
        lvis_dir: dir containing per-tile LVIS parquets (`<tile>.parquet`),
            produced by `partition_lvis_by_s2_tile`.
        vsm_tile_dir: VSM tile root, e.g.
            `/projects/dereeco/data/gvs/products/vsm/2017/original/tiles/geotiff`.
            Each tile has its own subdir with `RH<NN>_Q1.tif` rasters inside.
        save_dir: where to write `eval_lvis_<tile>.parquet` outputs.
        rh_metrics: subset of RH metrics to evaluate; defaults to all.
        aggregate_within_pixel: how to collapse multi-shot pixels.
            - `'mean'`:    LVIS = mean of shots in pixel (default, common).
            - `'max'`:     LVIS = max of shots — closer to "tallest return
                            in this pixel"; physically defensible for RH100.
            - `'mean_max'`: split treatment — RH95..RH100 use max
                            (top-of-canopy), RH10..RH90 use mean (lower
                            structure). Most physically motivated.
            - `'none'`:    don't aggregate; emit shot-level pairs.
        rewrite: redo tiles whose output already exists.

    Output columns (wide):
        `tile_id`, `row`, `col`, `n_shots`, `geometry`,
        `lvis_RH10..RH100` (meters),
        `vsm_RH10..RH100`  (meters; NaN where the raster is nodata).
        For `aggregate_within_pixel != 'none'`, geometry is the pixel
        center in the raster CRS; otherwise it's the original LVIS GLON/GLAT
        in EPSG:4326.
    """
    valid_modes = {'mean', 'max', 'mean_max', 'none'}
    if aggregate_within_pixel not in valid_modes:
        raise ValueError(
            f'aggregate_within_pixel must be one of {valid_modes}, '
            f'got {aggregate_within_pixel!r}')
    lvis_dir = Path(lvis_dir).expanduser()
    vsm_tile_dir = Path(vsm_tile_dir).expanduser()
    save_dir = Path(save_dir).expanduser()
    save_dir.mkdir(parents=True, exist_ok=True)
    rh_metrics = rh_metrics or RH_METRICS

    for parq in sorted(lvis_dir.glob('*.parquet')):
        tile_id = parq.stem
        out_file = save_dir / f'eval_lvis_{tile_id}.parquet'
        if out_file.exists() and not rewrite:
            print(f'{tile_id}: exists, skipping')
            continue
        tile_dir = vsm_tile_dir / tile_id
        if not tile_dir.exists():
            print(f'{tile_id}: VSM tile dir {tile_dir} missing, skipping')
            continue

        gdf = gpd.read_parquet(parq, columns=['geometry', *rh_metrics])
        # All RH rasters in a tile share the same grid; reproject + index once.
        first_tif = next((tile_dir / f'{rh}_Q1.tif' for rh in rh_metrics
                          if (tile_dir / f'{rh}_Q1.tif').exists()), None)
        if first_tif is None:
            print(f'{tile_id}: no VSM RH rasters found, skipping')
            continue
        with rasterio.open(first_tif) as src:
            tif_crs = src.crs
            tif_transform = src.transform
        gdf_proj = gdf.to_crs(tif_crs)
        rows, cols, valid = _row_col_for_points(first_tif, gdf_proj)

        out = pd.DataFrame({'tile_id': tile_id, 'row': rows, 'col': cols})
        for rh in rh_metrics:
            out[f'lvis_{rh}'] = gdf[rh].values
            tif = tile_dir / f'{rh}_Q1.tif'
            if not tif.exists():
                print(f'{tile_id}: {tif.name} missing, leaving vsm_{rh}=NaN')
                out[f'vsm_{rh}'] = np.nan
            else:
                out[f'vsm_{rh}'] = _sample_raster_at_rowcol(tif, rows, cols, valid)
        # Drop shots that fell outside the raster bounds.
        out = out[valid].reset_index(drop=True)

        if aggregate_within_pixel == 'none':
            out['n_shots'] = 1
            gdf_out = gpd.GeoDataFrame(
                out, geometry=gdf.geometry.values[valid], crs=gdf.crs)
        else:
            rh_agg_func = _per_metric_agg(aggregate_within_pixel, rh_metrics)
            agg = {f'lvis_{rh}': fn for rh, fn in rh_agg_func.items()}
            # VSM is constant within a (row, col); `first` skips redundant
            # averaging and handles NaN-only groups correctly.
            agg.update({f'vsm_{rh}': 'first' for rh in rh_metrics})
            grouped = out.groupby(['tile_id', 'row', 'col'], as_index=False)
            n_shots = grouped.size().rename(columns={'size': 'n_shots'})
            out = grouped.agg(agg).merge(n_shots, on=['tile_id', 'row', 'col'])
            # Geometry of an aggregated row = pixel center in the raster CRS.
            xs_c, ys_c = rasterio.transform.xy(
                tif_transform, out['row'].to_numpy(), out['col'].to_numpy())
            out_geom = gpd.points_from_xy(xs_c, ys_c)
            gdf_out = gpd.GeoDataFrame(out, geometry=out_geom, crs=tif_crs)

        gdf_out.to_parquet(out_file)
        kind = 'shots' if aggregate_within_pixel == 'none' else 'pixels'
        print(f'{tile_id}: {len(gdf_out)} {kind} paired across '
              f'{len(rh_metrics)} RH metrics ({aggregate_within_pixel})')


def extract_lvis_vsm_fhd_pairs(
        lvis_metrics_dir: str,
        vsm_fhd_tile_dir: str,
        save_dir: str,
        fhd_band: int = 1,
        aggregate_within_pixel: str = 'mean',
        carry_metrics: list = None,
        rewrite: bool = False,
        **kwargs):
    """For each S2 tile, sample the VSM diversity raster's FHD band at every
    LVIS shot location and write a per-tile parquet of paired FHD values.

    Counterpart of `extract_lvis_vsm_pairs` but for the single-metric FHD
    comparison. The VSM diversity product (see `evaluation/diversity_maps.py`)
    is a 4-band raster per tile (band 1 = fhd, 2 = enl1d, 3 = enl2d, 4 = cr);
    we only read band `fhd_band` here.

    Args:
        lvis_metrics_dir: dir of per-tile LVIS metrics parquets
            (`<tile>.parquet`), produced by
            `partition_lvis_metrics_by_s2_tile`.
        vsm_fhd_tile_dir: dir with per-tile diversity rasters
            (`<tile_id>.tif`, default 4-band layout). Defaults to the
            geotiff output of `create_tile_diversity_maps`.
        save_dir: where to write `eval_lvis_fhd_<tile>.parquet`.
        fhd_band: rasterio 1-based band index for FHD (default 1, matches
            `dst.set_band_description(1, "fhd")` in diversity_maps.py).
        aggregate_within_pixel:
            - `'mean'`: LVIS FHD = mean of shots in pixel (default).
            - `'none'`: don't aggregate; emit shot-level pairs (the VSM
              column will have duplicates within each pixel).
            (`max` is omitted: max diversity within a pixel doesn't have a
             clean physical interpretation the way max RH100 does.)
        carry_metrics: optional list of extra LVIS metric columns
            (e.g. ['ccover', 'lai']) to copy through to the pair parquet
            without aggregation logic of their own — aggregated with `mean`
            when pixel-binning. Lets you compute follow-up controls (FHD
            partial r controlling for ccover) without re-running this step.
        rewrite: redo tiles whose output already exists.

    Output columns:
        `tile_id`, `row`, `col`, `n_shots`, `geometry`,
        `lvis_fhd`, `vsm_fhd`,
        plus `lvis_<m>` for each `m` in `carry_metrics`.
    """
    valid_modes = {'mean', 'none'}
    if aggregate_within_pixel not in valid_modes:
        raise ValueError(
            f'aggregate_within_pixel must be one of {valid_modes}, '
            f'got {aggregate_within_pixel!r}')
    lvis_metrics_dir = Path(lvis_metrics_dir).expanduser()
    vsm_fhd_tile_dir = Path(vsm_fhd_tile_dir).expanduser()
    save_dir = Path(save_dir).expanduser()
    save_dir.mkdir(parents=True, exist_ok=True)
    carry_metrics = list(carry_metrics or [])

    needed_cols = ['geometry', 'fhd', *carry_metrics]

    for parq in sorted(lvis_metrics_dir.glob('*.parquet')):
        tile_id = parq.stem
        out_file = save_dir / f'eval_lvis_fhd_{tile_id}.parquet'
        if out_file.exists() and not rewrite:
            print(f'{tile_id}: exists, skipping')
            continue
        tif = vsm_fhd_tile_dir / f'{tile_id}.tif'
        if not tif.exists():
            print(f'{tile_id}: VSM FHD tile {tif} missing, skipping')
            continue

        gdf = gpd.read_parquet(parq, columns=needed_cols)
        with rasterio.open(tif) as src:
            tif_crs = src.crs
            tif_transform = src.transform
            # Read only the FHD band; convert masked nodata to NaN so the
            # downstream R2/RMSE filter on np.isfinite catches it.
            fhd_arr = src.read(fhd_band, masked=True).astype(np.float32).filled(np.nan)
            h, w = src.height, src.width

        gdf_proj = gdf.to_crs(tif_crs)
        xs = gdf_proj.geometry.x.to_numpy()
        ys = gdf_proj.geometry.y.to_numpy()
        rows, cols = rasterio.transform.rowcol(tif_transform, xs, ys)
        rows = np.asarray(rows)
        cols = np.asarray(cols)
        valid = (rows >= 0) & (rows < h) & (cols >= 0) & (cols < w)

        vsm_fhd = np.full(len(rows), np.nan, dtype=np.float32)
        vsm_fhd[valid] = fhd_arr[rows[valid], cols[valid]]

        out = pd.DataFrame({
            'tile_id': tile_id,
            'row': rows,
            'col': cols,
            'lvis_fhd': gdf['fhd'].to_numpy(),
            'vsm_fhd': vsm_fhd,
        })
        for m in carry_metrics:
            out[f'lvis_{m}'] = gdf[m].to_numpy()
        out = out[valid].reset_index(drop=True)

        if aggregate_within_pixel == 'none':
            out['n_shots'] = 1
            gdf_out = gpd.GeoDataFrame(
                out, geometry=gdf.geometry.values[valid], crs=gdf.crs)
        else:
            agg = {'lvis_fhd': 'mean', 'vsm_fhd': 'first'}
            for m in carry_metrics:
                agg[f'lvis_{m}'] = 'mean'
            grouped = out.groupby(['tile_id', 'row', 'col'], as_index=False)
            n_shots = grouped.size().rename(columns={'size': 'n_shots'})
            out = grouped.agg(agg).merge(n_shots, on=['tile_id', 'row', 'col'])
            xs_c, ys_c = rasterio.transform.xy(
                tif_transform, out['row'].to_numpy(), out['col'].to_numpy())
            out_geom = gpd.points_from_xy(xs_c, ys_c)
            gdf_out = gpd.GeoDataFrame(out, geometry=out_geom, crs=tif_crs)

        gdf_out.to_parquet(out_file)
        kind = 'shots' if aggregate_within_pixel == 'none' else 'pixels'
        print(f'{tile_id}: {len(gdf_out)} {kind} paired ({aggregate_within_pixel})')


def _stats(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """R2 / Pearson r / RMSE / MAE / ME on already-filtered finite arrays.

    R2 is sklearn's `1 - SS_res/SS_tot` — sensitive to bias and slope (a
    pure shift drops R2 even if pearson_r == 1). Pearson r is the unbiased
    linear-association strength, directly comparable to the `partial_r`
    column. Reporting both lets you see when VSM tracks LVIS well in shape
    (high r) but is off in level (low R2)."""
    n = int(len(y_true))
    residual = y_pred - y_true
    out = {
        'n': n,
        'r2': float(r2_score(y_true, y_pred)) if n > 1 else np.nan,
        'rmse': float(np.sqrt((residual ** 2).mean())),
        'mae': float(np.abs(residual).mean()),
        'me': float(residual.mean()),
    }
    if n > 1:
        r, p = pearsonr(y_true, y_pred)
        out['pearson_r'] = float(r)
        out['pearson_p'] = float(p)
    else:
        out['pearson_r'] = np.nan
        out['pearson_p'] = np.nan
    return out


def evaluate_on_lvis_pairs(
        pairs_dir: str,
        save_dir: str,
        pred_pattern: str,
        pred_label: str,
        output_basename: str,
        pred_color: str = 'C1',
        pairs_glob: str = '*.parquet',
        rh_metrics: list = None,
        per_tile: bool = True,
        control_rh: str = 'RH98',
        control_rh_min: float = None,
        control_rh_max: float = None,
        max_distance_m: float = None,
        make_plot: bool = True,
        plot_metrics: list = None,
        nrows: int = 4,
        ncols: int = 6,
        panel_size: float = 3.0,
        gridsize: int = 50,
        extent: tuple = None,
        annot_fontsize: float = 7,
        make_violin: bool = True,
        violin_figsize: tuple = None,
        violin_sample_size: int = 50_000,
        violin_yscale: str = 'linear',
        violin_random_state: int = 0,
        **kwargs):
    """Pool per-tile pair parquets and compute R2 / RMSE / MAE / ME for
    each RH metric, plus a partial correlation column controlling for TRUE
    LVIS `control_rh` (default RH98). Sensor-agnostic — the caller picks
    which `pred_*` columns to read via `pred_pattern`.

    Why the partial column? Plain R^2 between (pred RH25, LVIS RH25) is
    inflated by the shared canopy-top signal: tall sites have high RH25
    AND high RH98, so a predictor that only nailed RH98 still looks great
    on RH25. Partialing TRUE LVIS RH98 out of both sides isolates "pred
    picks up structure beyond canopy top height" — same logic as
    `structure_partial_correlation`. The control metric (`control_rh`
    itself) is skipped since `partial_r` is undefined there.

    Args:
        pairs_dir: directory of pair parquets.
        save_dir: where to write the stats CSVs and plots.
        pred_pattern: column-name template for the predicted side, with
            `{lvl}` substituted by the LVIS percentile digits.
            For LVIS-vs-VSM:  `"vsm_RH{lvl}"`  -> LVIS RH98 paired with `vsm_RH98`.
            For LVIS-vs-GEDI: `"gedi_rh{lvl}"` -> LVIS RH98 paired with `gedi_rh98`.
        pred_label: short display name for the predicted side ('VSM',
            'GEDI', ...). Used in plot axis labels and violin legend.
        output_basename: filename stem for CSVs and plots, e.g.
            `lvis_vsm` or `lvis_gedi`. Produces
            `{basename}_overall_stats.csv`, `{basename}_per_tile_stats.csv`,
            `scatter_grid_{basename}.pdf`, `violin_{basename}.pdf`.
        pred_color: matplotlib color for the predicted half of the split
            violin. Defaults to 'C1' (matches the original VSM figure);
            use 'C2' for GEDI to keep cross-figure comparability.
        pairs_glob: glob pattern under `pairs_dir`. Defaults to
            `*.parquet`; the LVIS-VSM pair extractor writes
            `eval_lvis_<tile>.parquet` so pass `*.parquet`
            there.
        rh_metrics: subset of LVIS RH metrics to score; defaults to every
            LVIS percentile (RH10..RH100).
        per_tile: also emit per-(tile, RH) stats.
        control_rh: LVIS metric used as the height control for the
            partial correlation. Default `'RH98'` (matches GEDI convention).
        control_rh_min, control_rh_max: optional bounds on the TRUE LVIS
            `control_rh` value applied before any stat is computed. Use
            e.g. `control_rh_min=20` to keep only tall-canopy pixels.
            The filename suffix (`_filter_<control>_gt<min>_lt<max>`) keeps
            multiple bounds runnable into the same `save_dir`.
        max_distance_m: post-extraction tightening — drop pairs whose
            `match_distance_m` column exceeds this. Only relevant for the
            footprint-pairing pipeline (LVIS-vs-GEDI); silently ignored
            if the column isn't present (e.g. pixel-aggregated VSM pairs).
            `None` keeps every pair.
        make_plot: write a `nrows x ncols` grid of log-normed hexbin
            density scatters (one panel per RH metric) with shared axes
            and a single colorbar.
        plot_metrics: subset of RH metrics to plot. Defaults to every
            metric that produced a row in the overall stats.
        nrows, ncols: grid layout. With 23 LVIS percentiles, 4x6 leaves
            one empty cell (hidden by `hexbin_density_grid`).
        panel_size, gridsize, extent: per-panel size, hexbin gridsize,
            and shared axis extent. `extent=None` auto-fits, rounded up
            to the next 5 m.
        annot_fontsize: font size of the per-panel stats box (R^2, r,
            RMSE, ME, partial_r). Defaults to 7 — five lines fit in a
            small panel; bump to 8-9 for wall posters, drop to 6 for
            dense 6x4 grids.
        make_violin: also write a two-sided split violin (LVIS left half,
            `pred_label` right half). Honors `plot_metrics`.
        violin_figsize: `None` -> `FIGURE_SIZES['wide']`.
        violin_sample_size: per-(metric, source) subsample size before
            the seaborn KDE; 50k gives a tight estimate in seconds on
            millions of pairs. `None` uses every observation.
        violin_yscale: `'linear'` or `'symlog'` (use symlog if any
            heights dip negative for the low percentiles).
        violin_random_state: RNG seed for the subsample.
    """
    pairs_dir = Path(pairs_dir).expanduser()
    save_dir = Path(save_dir).expanduser()
    save_dir.mkdir(parents=True, exist_ok=True)
    rh_metrics = rh_metrics or RH_METRICS

    files = sorted(pairs_dir.glob(pairs_glob))
    if not files:
        raise FileNotFoundError(
            f'No files matching {pairs_glob!r} under {pairs_dir}')

    def _pred_col(rh: str) -> str:
        # rh is 'RH98' / 'RH100'; lvl is the digits after 'RH'.
        return pred_pattern.format(lvl=rh[2:])

    # Geometry isn't needed for the stats; reading via pandas keeps RAM bounded.
    # The control column may be in `rh_metrics` already; `set()` keeps it unique.
    needed_metrics = set(rh_metrics) | {control_rh}
    cols_needed = ['tile_id']
    cols_needed += [f'lvis_{r}' for r in needed_metrics]
    cols_needed += [_pred_col(r) for r in needed_metrics]
    # Some pair parquets carry match_distance_m (footprint pairing); peek
    # the schema of the first file and pull it through if present so the
    # post-extraction distance filter can use it.
    import pyarrow.parquet as pq
    first_schema = set(pq.ParquetFile(files[0]).schema.names)
    has_distance = 'match_distance_m' in first_schema
    if has_distance:
        cols_needed.append('match_distance_m')

    df = pd.concat([pd.read_parquet(f, columns=cols_needed) for f in files],
                   ignore_index=True)
    print(f'Loaded {len(df)} paired shots from {len(files)} tiles')

    ctrl_col = f'lvis_{control_rh}'
    if ctrl_col not in df.columns:
        raise KeyError(f'Control column {ctrl_col} missing from pair parquets')

    suffix_parts = []
    if max_distance_m is not None:
        if not has_distance:
            print(f'max_distance_m={max_distance_m} ignored: no '
                  f'match_distance_m column in {files[0].name}')
        else:
            n_before = len(df)
            df = df[df['match_distance_m'] <= max_distance_m]
            suffix_parts.append(f'dlt{max_distance_m:g}m')
            print(f'Filter match_distance_m <= {max_distance_m}: '
                  f'{n_before} -> {len(df)} rows')
    if control_rh_min is not None:
        n_before = len(df)
        df = df[df[ctrl_col] >= control_rh_min]
        suffix_parts.append(f'{control_rh}gt{control_rh_min:g}')
        print(f'Filter LVIS {control_rh} >= {control_rh_min}: '
              f'{n_before} -> {len(df)} rows')
    if control_rh_max is not None:
        n_before = len(df)
        df = df[df[ctrl_col] <= control_rh_max]
        suffix_parts.append(f'{control_rh}lt{control_rh_max:g}')
        print(f'Filter LVIS {control_rh} <= {control_rh_max}: '
              f'{n_before} -> {len(df)} rows')
    suffix = f'_filter_{"_".join(suffix_parts)}' if suffix_parts else ''

    overall_rows = []
    panel_data: dict = {}  # rh -> (x, y, stats) for the optional grid plot
    for rh in rh_metrics:
        lvis_col, pred_col = f'lvis_{rh}', _pred_col(rh)
        if lvis_col not in df.columns or pred_col not in df.columns:
            continue
        mask = np.isfinite(df[lvis_col].values) & np.isfinite(df[pred_col].values)
        if mask.sum() < 2:
            print(f'{rh}: only {mask.sum()} finite pairs, skipping')
            continue
        x_arr = df.loc[mask, lvis_col].to_numpy()
        y_arr = df.loc[mask, pred_col].to_numpy()
        s = _stats(x_arr, y_arr)
        # Partial r of (pred_rh, lvis_rh) controlling for lvis_ctrl.
        # Undefined when `rh == control_rh` (perfect collinearity with z).
        if rh == control_rh:
            s['partial_r'] = np.nan
            s['partial_p'] = np.nan
        else:
            pr, pp, _ = partial_correlation(
                df[pred_col].to_numpy(),
                df[lvis_col].to_numpy(),
                df[ctrl_col].to_numpy(),
            )
            s['partial_r'] = pr
            s['partial_p'] = pp
        s['rh'] = rh
        overall_rows.append(s)
        panel_data[rh] = (x_arr, y_arr, s)
    overall = pd.DataFrame(overall_rows).set_index('rh')[
        ['n', 'r2', 'pearson_r', 'pearson_p', 'rmse', 'mae', 'me',
         'partial_r', 'partial_p']]
    overall_path = save_dir / f'{output_basename}_overall_stats{suffix}.csv'
    overall.to_csv(overall_path)
    print(f'\nOverall stats per RH metric (partial_r controls for LVIS {control_rh}):'
          f'\n{overall}\n-> {overall_path}')

    to_plot = ([rh for rh in (plot_metrics or rh_metrics) if rh in panel_data]
               if panel_data else [])

    if make_plot and to_plot:
        if extent is None:
            hi_data = max(max(float(np.max(x)), float(np.max(y)))
                          for rh in to_plot for x, y, _ in [panel_data[rh]])
            hi = int(math.ceil(hi_data / 5.0) * 5)
            extent_eff = (0.0, float(hi), 0.0, float(hi))
        else:
            extent_eff = tuple(extent)
        panels = [
            {
                'x': panel_data[rh][0],
                'y': panel_data[rh][1],
                'annotation': _rh_panel_annotation(panel_data[rh][2]),
                'title': rh,
            }
            for rh in to_plot
        ]
        grid_path = save_dir / f'scatter_grid_{output_basename}{suffix}.pdf'
        hexbin_density_grid(
            panels,
            save_path=grid_path,
            nrows=nrows,
            ncols=ncols,
            panel_size=panel_size,
            gridsize=gridsize,
            extent=extent_eff,
            refline='identity',
            equal_aspect=True,
            annot_fontsize=annot_fontsize,
            x_label='LVIS RH (m)',
            y_label=f'{pred_label} RH (m)',
        )
        print(f'RH grid scatter -> {grid_path}')

    if make_violin and to_plot:
        violin_path = save_dir / f'violin_{output_basename}{suffix}.pdf'
        _make_lvis_split_violin(
            panel_data, to_plot, violin_path,
            figsize=tuple(violin_figsize) if violin_figsize else FIGURE_SIZES['wide'],
            sample_size=violin_sample_size,
            yscale=violin_yscale,
            random_state=violin_random_state,
            right_label=pred_label,
            right_color=pred_color,
        )
        print(f'RH violin -> {violin_path}')

    if per_tile:
        per_tile_rows = []
        for tile_id, sub in df.groupby('tile_id', sort=True):
            for rh in rh_metrics:
                lvis_col, pred_col = f'lvis_{rh}', _pred_col(rh)
                if lvis_col not in sub.columns or pred_col not in sub.columns:
                    continue
                mask = np.isfinite(sub[lvis_col].values) & np.isfinite(sub[pred_col].values)
                if mask.sum() < 2:
                    continue
                s = _stats(sub.loc[mask, lvis_col].to_numpy(),
                           sub.loc[mask, pred_col].to_numpy())
                if rh == control_rh:
                    s['partial_r'] = np.nan
                    s['partial_p'] = np.nan
                else:
                    pr, pp, _ = partial_correlation(
                        sub[pred_col].to_numpy(),
                        sub[lvis_col].to_numpy(),
                        sub[ctrl_col].to_numpy(),
                    )
                    s['partial_r'] = pr
                    s['partial_p'] = pp
                s['tile_id'] = tile_id
                s['rh'] = rh
                per_tile_rows.append(s)
        per_tile_df = pd.DataFrame(per_tile_rows)[
            ['tile_id', 'rh', 'n', 'r2', 'pearson_r', 'pearson_p',
             'rmse', 'mae', 'me', 'partial_r', 'partial_p']]
        per_tile_path = save_dir / f'{output_basename}_per_tile_stats{suffix}.csv'
        per_tile_df.to_csv(per_tile_path, index=False)
        print(f'Per-tile stats -> {per_tile_path}')
    return overall


def _make_lvis_split_violin(
        panel_data: dict, to_plot: list, save_path: Path,
        figsize: tuple, sample_size: int, yscale: str,
        random_state: int,
        right_label: str = 'VSM', right_color: str = 'C1') -> None:
    """Two-sided violin: left half = LVIS, right half = `right_label`, one
    violin per RH metric, all metrics on one axes. Models the on_gedi
    `make_violin_plot` (sns.violinplot split=True, hue='source').

    `panel_data[rh]` is `(x_arr, y_arr, stats)` from the stats loop in
    `evaluate_on_lvis_pairs`; both arrays are already paired and finite,
    so we just have to long-format them. With 23 metrics * millions of
    pairs, a no-subsample call would feed seaborn a giant KDE; `sample_size`
    subsamples each (metric, source) independently to keep the violin fast.

    `right_label` controls both the hue category and the figure title
    (`LVIS vs <right_label> RH distributions`); use 'VSM' for the
    VSM-vs-LVIS comparison and 'GEDI' for the GEDI-vs-LVIS comparison.
    `right_color` controls the right-half violin color; defaults to
    'C1' (VSM convention) — pass 'C2' for GEDI so the two figures stay
    visually distinguishable when shown side-by-side.
    """
    rng = np.random.default_rng(random_state)

    def _maybe_sample(arr):
        if sample_size is None or len(arr) <= sample_size:
            return arr
        return rng.choice(arr, size=sample_size, replace=False)

    rh_chunks, height_chunks, source_chunks = [], [], []
    for rh in to_plot:
        x_arr, y_arr, _ = panel_data[rh]
        x_sub = _maybe_sample(x_arr)
        y_sub = _maybe_sample(y_arr)
        rh_chunks.append(np.full(len(x_sub) + len(y_sub), rh))
        height_chunks.append(np.concatenate([x_sub, y_sub]))
        source_chunks.append(np.concatenate([
            np.full(len(x_sub), 'LVIS'),
            np.full(len(y_sub), right_label),
        ]))
    long_df = pd.DataFrame({
        'rh_metric': np.concatenate(rh_chunks),
        'height': np.concatenate(height_chunks),
        'source': np.concatenate(source_chunks),
    })

    fig, ax = plt.subplots(figsize=figsize)
    sns.violinplot(
        data=long_df, x='rh_metric', y='height', hue='source',
        split=True, inner='quartile', order=to_plot,
        palette={'LVIS': 'C0', right_label: right_color}, ax=ax,
    )
    if yscale == 'symlog':
        # Symlog keeps the near-zero region linear; useful if there are
        # negative heights from low-percentile RH residuals.
        ax.set_yscale('symlog', linthresh=1)
    ax.set_xlabel('LVIS RH metric', fontsize=FONT_SIZES['label'])
    ax.set_ylabel('Height (m)', fontsize=FONT_SIZES['label'])
    ax.set_title(f'LVIS vs {right_label} RH distributions',
                 fontsize=FONT_SIZES['title'])
    ax.grid(True, axis='y', linestyle='--', alpha=0.5)
    ax.set_axisbelow(True)
    if ax.get_legend() is not None:
        ax.get_legend().set_title('')
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    plt.tight_layout()
    fig.savefig(save_path, bbox_inches='tight')
    plt.close(fig)


def _rh_panel_annotation(stats: dict) -> str:
    """Compact per-panel stats text for the RH grid. Tighter than the
    single-panel FHD annotation because every cell carries its own copy,
    and the panels are small. ME (mean error) sits next to RMSE so bias
    and spread are visible at a glance."""
    lines = [
        f"$R^2$={stats['r2']:.2f}",
        f"r={stats['pearson_r']:.2f}",
        f"RMSE={stats['rmse']:.1f}",
        f"ME={stats['me']:.2f}",
    ]
    if np.isfinite(stats.get('partial_r', np.nan)):
        lines.append(f"pr={stats['partial_r']:.2f}")
    return '\n'.join(lines)


def _fhd_stats_annotation(stats: dict) -> str:
    """Boxed stats text for the FHD hexbin. Same style as on_als'
    `_stats_annotation` but uses Pearson r (the canonical line for diversity
    indices) instead of avg height (which is a CHM-specific summary)."""
    lines = [
        f"$R^2$ = {stats['r2']:.2f}",
        f"r = {stats['pearson_r']:.2f}",
        f"RMSE = {stats['rmse']:.2f}",
        f"ME = {stats['me']:.2f}",
        f"N = ${sci_notation(stats['n'])}$",
    ]
    if 'partial_r' in stats and np.isfinite(stats.get('partial_r', np.nan)):
        lines.insert(2, f"partial r = {stats['partial_r']:.2f}")
    return '\n'.join(lines)


def evaluate_vsm_on_lvis_fhd(
        pairs_dir: str,
        save_dir: str,
        per_tile: bool = True,
        control_metric: str = None,
        control_min: float = None,
        control_max: float = None,
        make_plot: bool = True,
        figsize: tuple = None,
        gridsize: int = 60,
        extent: tuple = None,
        **kwargs):
    """Pool all per-tile FHD pair parquets and compute R2 / Pearson r /
    RMSE / MAE / ME between LVIS FHD and VSM FHD.

    Args:
        pairs_dir: directory of `eval_lvis_fhd_<tile>.parquet` produced by
            `extract_lvis_vsm_fhd_pairs`.
        save_dir: where to write the stats CSVs.
        per_tile: also emit per-tile stats.
        control_metric: optional carried LVIS column (e.g. `'ccover'`) to
            partial out from both sides — answers "does VSM FHD pick up
            structure beyond canopy cover?". Only meaningful if
            `extract_lvis_vsm_fhd_pairs(... carry_metrics=['ccover', ...])`
            was run so the column exists in the pair parquets.
        control_min, control_max: optional bounds on the carried
            `control_metric` applied before stats (e.g.
            `control_metric='ccover', control_min=0.5` to restrict to
            covered pixels). The filename gains a
            `_filter_<metric>_gt<min>_lt<max>` suffix so multiple runs
            don't clobber each other.
        make_plot: also write a log-normed hexbin density scatter of
            (LVIS FHD, VSM FHD) with stats annotation, using the project's
            shared `hexbin_density_plot` style.
        figsize, gridsize, extent: passthroughs to `hexbin_density_plot`.
            `extent=None` lets the plot auto-fit the data range; note that
            LVIS L2A FHD and VSM FHD use *different* foliage-bin schemes
            (LVIS bins the waveform finely, VSM uses bin_width=5 m), so
            the two are not on the same numeric scale — the plot will
            surface that as an off-diagonal point cloud rather than
            quietly bias the R2.
    """
    pairs_dir = Path(pairs_dir).expanduser()
    save_dir = Path(save_dir).expanduser()
    save_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(pairs_dir.glob('eval_lvis_fhd_*.parquet'))
    if not files:
        raise FileNotFoundError(f'No eval_lvis_fhd_*.parquet under {pairs_dir}')

    needed = ['tile_id', 'lvis_fhd', 'vsm_fhd']
    if control_metric is not None:
        needed.append(f'lvis_{control_metric}')
    df = pd.concat([pd.read_parquet(f, columns=needed) for f in files],
                   ignore_index=True)
    print(f'Loaded {len(df)} paired FHD samples from {len(files)} tiles')

    suffix_parts = []
    if control_metric is not None and (control_min is not None or control_max is not None):
        ctrl_col = f'lvis_{control_metric}'
        if control_min is not None:
            n_before = len(df)
            df = df[df[ctrl_col] >= control_min]
            suffix_parts.append(f'gt{control_min:g}')
            print(f'Filter LVIS {control_metric} >= {control_min}: '
                  f'{n_before} -> {len(df)} rows')
        if control_max is not None:
            n_before = len(df)
            df = df[df[ctrl_col] <= control_max]
            suffix_parts.append(f'lt{control_max:g}')
            print(f'Filter LVIS {control_metric} <= {control_max}: '
                  f'{n_before} -> {len(df)} rows')
    suffix = (f'_filter_{control_metric}_{"_".join(suffix_parts)}'
              if suffix_parts else '')

    def _row(sub):
        mask = np.isfinite(sub['lvis_fhd'].values) & np.isfinite(sub['vsm_fhd'].values)
        if mask.sum() < 2:
            return None
        s = _stats(sub.loc[mask, 'lvis_fhd'].to_numpy(),
                   sub.loc[mask, 'vsm_fhd'].to_numpy())
        if control_metric is not None:
            ctrl_col = f'lvis_{control_metric}'
            cmask = mask & np.isfinite(sub[ctrl_col].values)
            if cmask.sum() > 2:
                pr, pp, _ = partial_correlation(
                    sub.loc[cmask, 'vsm_fhd'].to_numpy(),
                    sub.loc[cmask, 'lvis_fhd'].to_numpy(),
                    sub.loc[cmask, ctrl_col].to_numpy(),
                )
                s['partial_r'] = pr
                s['partial_p'] = pp
            else:
                s['partial_r'] = np.nan
                s['partial_p'] = np.nan
        return s

    overall_row = _row(df)
    if overall_row is None:
        raise RuntimeError('No finite (lvis_fhd, vsm_fhd) pairs after filtering')
    overall = pd.DataFrame([overall_row])
    cols = ['n', 'r2', 'pearson_r', 'pearson_p', 'rmse', 'mae', 'me']
    if control_metric is not None:
        cols += ['partial_r', 'partial_p']
    overall = overall[cols]
    overall_path = save_dir / f'lvis_fhd_overall_stats{suffix}.csv'
    overall.to_csv(overall_path, index=False)
    ctrl_msg = (f' (partial_r controls for LVIS {control_metric})'
                if control_metric is not None else '')
    print(f'\nOverall FHD stats{ctrl_msg}:\n{overall}\n-> {overall_path}')

    if make_plot:
        # Log-normed hexbin density scatter on the pooled (finite) pairs.
        # Identity refline + equal aspect because the two should be on the
        # same FHD scale in principle (Shannon entropy of foliage bins);
        # if they aren't, that's exactly what we want the plot to show.
        m = (np.isfinite(df['lvis_fhd'].values) &
             np.isfinite(df['vsm_fhd'].values))
        x_arr = df.loc[m, 'lvis_fhd'].to_numpy()
        y_arr = df.loc[m, 'vsm_fhd'].to_numpy()
        # Symmetric default extent so the data range is the same on x and y
        # — combined with the box_aspect lock below, the identity diagonal
        # spans corner-to-corner and any LVIS/VSM bin-scheme mismatch shows
        # as off-diagonal mass rather than being absorbed into an axis rescale.
        if extent is None:
            lo = float(min(np.min(x_arr), np.min(y_arr)))
            hi = float(max(np.max(x_arr), np.max(y_arr)))
            extent_eff = (lo, hi, lo, hi)
        else:
            extent_eff = tuple(extent)
        scatter_path = save_dir / f'scatter_lvis_vsm_fhd{suffix}.pdf'
        # Create the fig+ax ourselves so we can force the plotting box to a
        # 1:1 aspect *after* the colorbar divider has carved out its slot.
        # hexbin_density_plot's own `equal_aspect=True` calls set_aspect('equal'),
        # which under make_axes_locatable can fall back to adjustable='datalim'
        # and silently expand the data range instead of squaring the box.
        # set_box_aspect bypasses that path: it operates on the box's
        # height/width ratio directly and is colorbar-divider safe.
        fig, ax = plt.subplots(figsize=figsize or FIGURE_SIZES['square'])
        hexbin_density_plot(
            x_arr, y_arr,
            ax=ax,
            save_path=None,
            gridsize=gridsize,
            extent=extent_eff,
            refline='identity',
            equal_aspect=False,
            annotation=_fhd_stats_annotation(overall_row),
            annot_corner='upper left',
            annot_fontsize=12,
            x_label='LVIS FHD (L2A footprint metrics)',
            y_label='VSM FHD (diversity_indices band 1)',
            title='LVIS vs VSM Foliage Height Diversity',
        )
        ax.set_box_aspect(1)
        fig.savefig(scatter_path, bbox_inches='tight')
        plt.close(fig)
        print(f'FHD scatter -> {scatter_path}')

    if per_tile:
        per_tile_rows = []
        for tile_id, sub in df.groupby('tile_id', sort=True):
            s = _row(sub)
            if s is None:
                continue
            s['tile_id'] = tile_id
            per_tile_rows.append(s)
        per_tile_df = pd.DataFrame(per_tile_rows)
        per_tile_df = per_tile_df[['tile_id', *cols]]
        per_tile_path = save_dir / f'lvis_fhd_per_tile_stats{suffix}.csv'
        per_tile_df.to_csv(per_tile_path, index=False)
        print(f'Per-tile FHD stats -> {per_tile_path}')
    return overall


# ============================================================================
# Augment LVIS-VSM RH pair parquets with the LVIS L2A `fhd` column
# ============================================================================
def attach_lvis_fhd_to_rh_pairs(
        rh_pairs_dir: str,
        fhd_pairs_dir: str,
        save_dir: str,
        rh_filename_pattern: str = '*.parquet',
        rewrite: bool = False,
        **kwargs) -> None:
    """Merge `lvis_fhd` onto the LVIS-VSM RH pair parquets by
    (tile_id, row, col).

    Both pair products are keyed by the same pixel index — RH pairs from
    `extract_lvis_vsm_pairs` and FHD pairs from `extract_lvis_vsm_fhd_pairs`
    aggregate LVIS shots into the same VSM 10 m grid. The merge is a
    pure pandas left-join on (tile_id, row, col); no resampling.

    Only `lvis_fhd` is copied through. `vsm_fhd` is intentionally
    skipped — the diversity raster is already evaluated separately.

    Requires both `extract_lvis_vsm_pairs` and `extract_lvis_vsm_fhd_pairs`
    to have run first.

    Args:
        rh_pairs_dir: dir of RH pair parquets from
            `extract_lvis_vsm_pairs`. Default filename pattern is
            `eval_lvis_<tile>.parquet`.
        fhd_pairs_dir: dir of FHD pair parquets from
            `extract_lvis_vsm_fhd_pairs`
            (`eval_lvis_fhd_<tile>.parquet`).
        save_dir: dir for the augmented `eval_lvis_<tile>.parquet`.
            Same filename pattern as the RH input so downstream ops
            (e.g. `evaluate_on_lvis_pairs`) work unchanged.
        rh_filename_pattern: glob for the RH input. The tile id is
            inferred by stripping `eval_lvis_` from the stem.
        rewrite: redo tiles whose augmented file already exists.
    """
    rh_pairs_dir = Path(rh_pairs_dir).expanduser()
    fhd_pairs_dir = Path(fhd_pairs_dir).expanduser()
    save_dir = Path(save_dir).expanduser()
    save_dir.mkdir(parents=True, exist_ok=True)

    rh_files = sorted(rh_pairs_dir.glob(rh_filename_pattern))
    if not rh_files:
        raise FileNotFoundError(
            f'No files matching {rh_filename_pattern!r} under {rh_pairs_dir}')

    for rh_file in rh_files:
        tile_id = rh_file.stem.replace('eval_lvis_', '')
        out_file = save_dir / rh_file.name
        if out_file.exists() and not rewrite:
            print(f'{tile_id}: exists, skipped')
            continue
        fhd_file = fhd_pairs_dir / f'eval_lvis_fhd_{tile_id}.parquet'
        if not fhd_file.exists():
            print(f'{tile_id}: FHD pair {fhd_file.name} missing, skipped')
            continue

        rh_gdf = gpd.read_parquet(rh_file)
        fhd_df = pd.read_parquet(
            fhd_file, columns=['tile_id', 'row', 'col', 'lvis_fhd'])
        merged = rh_gdf.merge(fhd_df, on=['tile_id', 'row', 'col'], how='left')
        # gpd.GeoDataFrame.merge preserves the active geometry; recast
        # defensively for older geopandas versions.
        if not isinstance(merged, gpd.GeoDataFrame):
            merged = gpd.GeoDataFrame(merged, geometry=rh_gdf.geometry.name,
                                      crs=rh_gdf.crs)
        merged.to_parquet(out_file)
        n_total = len(merged)
        n_with_fhd = int(np.isfinite(merged['lvis_fhd']).sum())
        print(f'{tile_id}: {n_with_fhd}/{n_total} rows got lvis_fhd '
              f'-> {out_file.name}')

    print(f'\nDone. Augmented RH+FHD pairs in {save_dir}')


# ============================================================================
# Cross-sensor pairing: GEDI 2020 vs LVIS 2016 inside stable forest
# ============================================================================
# For every GEDI shot in stable forest, attach the single nearest LVIS shot
# within `max_radius_m` (KDTree in UTM), writing one per-tile parquet of paired
# `gedi_rh{lvl}` + `lvis_RH{lvl}` columns. Downstream extractors then sample VSM
# at the pair locations and attach LVIS L2A `fhd`, so all three sensors can be
# scored on the same shot population via `evaluate_on_lvis_pairs`.
#
# Caveats:
#   - GEDI RH levels are dense 0..100; LVIS levels are sparse (10,15,..,100).
#     Output keeps both naming schemes distinct (`gedi_rh`/`lvis_RH`).
#   - Nearest-neighbour at a small radius pairs one LVIS point inside the GEDI
#     footprint disk, not an integrating estimate.

# LVIS RH percentiles (uppercase) match the partitioned per-tile parquets and
# equal RH_METRICS. GEDI RH percentiles are dense 0..100.
LVIS_RH_COLUMNS = RH_METRICS
GEDI_RH_COLUMNS = [f'rh{i}' for i in range(101)]
# Optional QA columns passed through to the pair parquet when present in the
# source GEDI file. Missing ones are silently dropped — different vintages of
# the GEDI prep carry different metadata.
GEDI_OPTIONAL_COLUMNS = ['slope', 'quality_flag', 'sensitivity', 'BIOME']


def _sample_mask_at_points(gdf: gpd.GeoDataFrame,
                           mask_src: rasterio.io.DatasetReader) -> np.ndarray:
    """Sample a 1-band raster at each point. Out-of-bounds and nodata
    return 0 (treated as not-stable-forest)."""
    xs = gdf.geometry.x.to_numpy()
    ys = gdf.geometry.y.to_numpy()
    rows, cols = rasterio.transform.rowcol(mask_src.transform, xs, ys)
    rows = np.asarray(rows)
    cols = np.asarray(cols)
    valid = ((rows >= 0) & (rows < mask_src.height) &
             (cols >= 0) & (cols < mask_src.width))
    out = np.zeros(len(gdf), dtype=np.uint8)
    if valid.any():
        arr = mask_src.read(1)
        nodata = mask_src.nodata
        sampled = arr[rows[valid], cols[valid]]
        if nodata is not None:
            sampled = np.where(sampled == nodata, 0, sampled)
        out[valid] = sampled
    return out


def _load_filter_to_mask(parq_path: Path, mask_src: rasterio.io.DatasetReader,
                         keep_cols: list) -> gpd.GeoDataFrame:
    """Read a per-tile parquet (only the columns we need), reproject to
    the mask CRS, drop shots outside stable forest. Returns an empty GDF
    if no rows survive (so downstream pairing skips the tile cleanly)."""
    available = set(pq_columns(parq_path))
    cols = ['geometry'] + [c for c in keep_cols if c in available]
    missing = [c for c in keep_cols if c not in available]
    if missing:
        print(f'  {parq_path.name}: missing columns (will skip): {missing[:5]}'
              f'{"..." if len(missing) > 5 else ""}')
    gdf = gpd.read_parquet(parq_path, columns=cols)
    if gdf.empty:
        return gdf
    gdf = gdf.to_crs(mask_src.crs)
    mask_vals = _sample_mask_at_points(gdf, mask_src)
    return gdf.loc[mask_vals == 1].reset_index(drop=True)


def pq_columns(parq_path: Path) -> list:
    """Lightweight schema peek — avoids loading the full file just to
    know which optional columns exist. pyarrow opens only the footer."""
    return pq.ParquetFile(parq_path).schema.names


def _pair_nearest(gedi_gdf: gpd.GeoDataFrame, lvis_gdf: gpd.GeoDataFrame,
                  max_radius_m: float) -> pd.DataFrame:
    """For each GEDI shot, attach the single nearest LVIS shot within
    `max_radius_m`. cKDTree expects projected coords (mask CRS is UTM)."""
    if gedi_gdf.empty or lvis_gdf.empty:
        return pd.DataFrame()
    lvis_xy = np.column_stack([lvis_gdf.geometry.x.to_numpy(),
                               lvis_gdf.geometry.y.to_numpy()])
    gedi_xy = np.column_stack([gedi_gdf.geometry.x.to_numpy(),
                               gedi_gdf.geometry.y.to_numpy()])
    tree = cKDTree(lvis_xy)
    dist, idx = tree.query(gedi_xy, k=1, distance_upper_bound=max_radius_m)
    matched = np.isfinite(dist)
    if not matched.any():
        return pd.DataFrame()

    gedi_kept = gedi_gdf.loc[matched].reset_index(drop=True)
    lvis_kept = lvis_gdf.iloc[idx[matched]].reset_index(drop=True)

    # GEDI columns get a `gedi_` prefix, LVIS gets `lvis_` — keeps the
    # uppercase/lowercase distinction visible AND lets
    # structure_partial_correlation use `gedi_rh{lvl}` / `lvis_RH{lvl}`
    # patterns directly. Assemble all columns at once to avoid the
    # repeated-insert fragmentation warning.
    cols = {f'gedi_{col}': gedi_kept[col].values
            for col in gedi_kept.columns if col != 'geometry'}
    cols.update({f'lvis_{col}': lvis_kept[col].values
                 for col in lvis_kept.columns if col != 'geometry'})
    cols['match_distance_m'] = dist[matched].astype(np.float32)
    out = pd.DataFrame(cols)
    # Output geometry = GEDI shot center in mask CRS (the integration
    # reference). Drop the LVIS coordinate since match_distance_m
    # encodes the offset.
    out = gpd.GeoDataFrame(out, geometry=gedi_kept.geometry.values,
                           crs=gedi_gdf.crs)
    return out


def _process_one_tile(tile_id: str, gedi_dir: Path, lvis_dir: Path,
                      forest_mask_dir: Path, save_dir: Path,
                      max_radius_m: float, gedi_keep_cols: list,
                      lvis_keep_cols: list, rewrite: bool) -> str:
    out_file = save_dir / f'{tile_id}.parquet'
    if out_file.exists() and not rewrite:
        return f'{tile_id}: exists, skipped'

    gedi_parq = gedi_dir / f'{tile_id}.parquet'
    lvis_parq = lvis_dir / f'{tile_id}.parquet'
    mask_tif = forest_mask_dir / f'{tile_id}.tif'
    for label, p in [('GEDI', gedi_parq), ('LVIS', lvis_parq),
                     ('mask', mask_tif)]:
        if not p.exists():
            return f'{tile_id}: {label} missing ({p}), skipped'

    with rasterio.open(mask_tif) as mask_src:
        gedi_gdf = _load_filter_to_mask(gedi_parq, mask_src, gedi_keep_cols)
        lvis_gdf = _load_filter_to_mask(lvis_parq, mask_src, lvis_keep_cols)

    if gedi_gdf.empty or lvis_gdf.empty:
        return (f'{tile_id}: no stable-forest shots '
                f'(GEDI {len(gedi_gdf)}, LVIS {len(lvis_gdf)}), skipped')

    pairs = _pair_nearest(gedi_gdf, lvis_gdf, max_radius_m)
    if pairs.empty:
        return (f'{tile_id}: no pairs within {max_radius_m} m '
                f'(GEDI {len(gedi_gdf)}, LVIS {len(lvis_gdf)})')

    pairs['tile_id'] = tile_id
    pairs.to_parquet(out_file)
    return (f'{tile_id}: {len(pairs)} pairs '
            f'(GEDI in mask={len(gedi_gdf)}, LVIS in mask={len(lvis_gdf)}, '
            f'median dist={np.median(pairs["match_distance_m"]):.1f} m) '
            f'-> {out_file.name}')


def extract_lvis_gedi_pairs(
        meta_file: str,
        gedi_dir: str,
        lvis_dir: str,
        forest_mask_dir: str,
        save_dir: str,
        max_radius_m: float = 12.5,
        gedi_optional_columns: list = None,
        max_workers: int = 4,
        rewrite: bool = False,
        **kwargs) -> None:
    """Build per-tile GEDI–LVIS pair parquets restricted to stable forest.

    Args:
        meta_file: CSV with a `Tile name` column listing S2 tiles to
            process (e.g. meta_lvis2016_profile.csv).
        gedi_dir: dir of per-tile GEDI parquets (`<tile>.parquet`), each
            carrying `rh0..rh100`, `slope`, geometry.
        lvis_dir: dir of per-tile LVIS parquets from
            `partition_lvis_by_s2_tile` (`RH10..RH100`, geometry).
        forest_mask_dir: dir of per-tile stable-forest GeoTIFFs from
            `evaluation/forest_mask.py` (10 m UTM, 1 = stable forest).
        save_dir: output dir for `<tile>.parquet` pair files.
        max_radius_m: max GEDI->nearest-LVIS distance to count as a
            pair. ~12.5 m matches the GEDI footprint radius; tighten to
            5 m for stricter overlap, loosen with caution (LVIS spacing
            varies with overflight pattern).
        gedi_optional_columns: extra GEDI columns to carry through if
            present (slope, quality_flag, sensitivity, biome).
            `None` -> `GEDI_OPTIONAL_COLUMNS`.
        max_workers: tile-level ProcessPoolExecutor parallelism. One
            tile fits comfortably in one worker; per-tile cost is
            dominated by GEDI parquet read.
        rewrite: redo tiles whose pair parquet already exists.
    """
    meta_file = Path(meta_file).expanduser()
    gedi_dir = Path(gedi_dir).expanduser()
    lvis_dir = Path(lvis_dir).expanduser()
    forest_mask_dir = Path(forest_mask_dir).expanduser()
    save_dir = Path(save_dir).expanduser()
    save_dir.mkdir(parents=True, exist_ok=True)

    gedi_optional = list(gedi_optional_columns
                         if gedi_optional_columns is not None
                         else GEDI_OPTIONAL_COLUMNS)
    gedi_keep = GEDI_RH_COLUMNS + gedi_optional
    lvis_keep = LVIS_RH_COLUMNS

    tiles = pd.read_csv(meta_file)['Tile name'].astype(str).tolist()
    print(f'Loaded {len(tiles)} tiles from {meta_file}')

    # Processes (not threads): the RH-column dataframe ops + KDTree build
    # hit the GIL hard, and the rasterio handle is opened per-worker so
    # cross-process file descriptor sharing isn't a concern.
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futs = {
            ex.submit(_process_one_tile, t, gedi_dir, lvis_dir,
                      forest_mask_dir, save_dir, max_radius_m,
                      gedi_keep, lvis_keep, rewrite): t
            for t in tiles
        }
        for fut in as_completed(futs):
            tile = futs[fut]
            try:
                print(fut.result())
            except Exception as e:
                print(f'{tile}: FAILED -> {e}')

    print(f'\nDone. Pairs in {save_dir}')


# ============================================================================
# Augment pair parquets with VSM RH columns sampled at the GEDI center
# ============================================================================
def _sample_vsm_on_one_tile(parq_path: Path, vsm_tile_dir: Path,
                            save_dir: Path, rh_metrics: list,
                            rewrite: bool) -> str:
    """Per-tile worker: read the pair parquet, sample each VSM RH raster
    at the pair geometry, append `vsm_RH<NN>` columns, write augmented
    parquet. Reuses the partition sampling helpers so VSM is converted from
    decimetres to metres exactly the same way as extract_lvis_vsm_pairs.
    """
    tile_id = parq_path.stem
    out_file = save_dir / f'{tile_id}.parquet'
    if out_file.exists() and not rewrite:
        return f'{tile_id}: exists, skipped'

    tile_dir = vsm_tile_dir / tile_id
    if not tile_dir.exists():
        return f'{tile_id}: VSM tile dir {tile_dir} missing, skipped'

    gdf = gpd.read_parquet(parq_path)
    if gdf.empty:
        return f'{tile_id}: pair parquet empty, skipped'

    # All RH rasters in a tile share the grid; resolve row/col once.
    first_tif = next(
        (tile_dir / f'{rh}_Q1.tif' for rh in rh_metrics
         if (tile_dir / f'{rh}_Q1.tif').exists()),
        None,
    )
    if first_tif is None:
        return f'{tile_id}: no VSM RH<NN>_Q1.tif under {tile_dir}, skipped'

    with rasterio.open(first_tif) as src:
        tif_crs = src.crs
    gdf_proj = gdf.to_crs(tif_crs)
    rows, cols, valid = _row_col_for_points(first_tif, gdf_proj)

    for rh in rh_metrics:
        tif = tile_dir / f'{rh}_Q1.tif'
        if not tif.exists():
            gdf[f'vsm_{rh}'] = np.nan
        else:
            gdf[f'vsm_{rh}'] = _sample_raster_at_rowcol(tif, rows, cols, valid)

    gdf.to_parquet(out_file)
    n_invalid = int((~valid).sum())
    msg = f'{tile_id}: {len(gdf)} pairs augmented'
    if n_invalid:
        # Shouldn't normally happen — the pair geometry was already inside
        # the stable-forest mask, which is per-tile — but log if any land
        # outside the VSM raster footprint (e.g. tile-edge slop when the
        # mask fell back to the native 30 m EPSG:4326 raster).
        msg += f' ({n_invalid} outside VSM raster, vsm_RH=NaN)'
    return msg + f' -> {out_file.name}'


def extract_vsm_on_pair_locations(
        pairs_dir: str,
        vsm_tile_dir: str,
        save_dir: str,
        rh_metrics: list = None,
        max_workers: int = 4,
        rewrite: bool = False,
        **kwargs) -> None:
    """Append `vsm_RH<NN>` columns to each LVIS-GEDI pair parquet so VSM
    can be evaluated against LVIS on the same shot population GEDI was
    evaluated against.

    Sampling is one VSM pixel at the GEDI shot center (the pair's
    geometry, already in the mask's UTM CRS for tiles where the mask was
    aligned to the VSM grid). VSM RH<NN>_Q1.tif stores heights in
    decimetres; the sampling helper divides by 10 to land back in metres.

    Args:
        pairs_dir: dir of LVIS-GEDI pair parquets from
            `extract_lvis_gedi_pairs` (`<tile>.parquet` each carrying
            `lvis_RH<NN>`, `gedi_rh<NN>`, geometry).
        vsm_tile_dir: VSM tile root for the year you want to sample,
            e.g. `~/data/gvs/products/vsm/2020/original/tiles/geotiff`.
            Each tile has its own subdir with `RH<NN>_Q1.tif` rasters.
        save_dir: output dir for augmented parquets. Run twice (once per
            VSM year) into distinct dirs so you can compare 2017 vs 2020
            on the same pair locations.
        rh_metrics: subset of LVIS RH percentiles to sample from VSM;
            defaults to every LVIS percentile (RH10..RH100).
        max_workers: parallel per-tile workers (process pool). I/O-bound
            but parquet writes benefit from true parallelism.
        rewrite: redo tiles whose augmented parquet already exists.
    """
    pairs_dir = Path(pairs_dir).expanduser()
    vsm_tile_dir = Path(vsm_tile_dir).expanduser()
    save_dir = Path(save_dir).expanduser()
    save_dir.mkdir(parents=True, exist_ok=True)
    rh_metrics = rh_metrics or RH_METRICS

    parq_files = sorted(pairs_dir.glob('*.parquet'))
    if not parq_files:
        raise FileNotFoundError(f'No <tile>.parquet under {pairs_dir}')
    print(f'Augmenting {len(parq_files)} pair parquets with VSM RH from '
          f'{vsm_tile_dir}')

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futs = {
            ex.submit(_sample_vsm_on_one_tile, p, vsm_tile_dir, save_dir,
                      rh_metrics, rewrite): p.stem
            for p in parq_files
        }
        for fut in as_completed(futs):
            tile = futs[fut]
            try:
                print(fut.result())
            except Exception as e:
                print(f'{tile}: FAILED -> {e}')

    print(f'\nDone. Augmented pairs in {save_dir}')


# ============================================================================
# Attach LVIS L2A `fhd` to each LVIS-GEDI pair via spatial join
# ============================================================================
def _attach_lvis_fhd_one_tile(parq_path: Path, lvis_metrics_dir: Path,
                              save_dir: Path, max_radius_m: float,
                              rewrite: bool) -> str:
    """Per-tile worker: KDTree the LVIS L2A shots, find the nearest one
    within max_radius_m of each pair's geometry, attach its `fhd` value.
    Pairs with no within-radius L2A shot get NaN."""
    tile_id = parq_path.stem
    out_file = save_dir / f'{tile_id}.parquet'
    if out_file.exists() and not rewrite:
        return f'{tile_id}: exists, skipped'

    lvis_metrics_parq = lvis_metrics_dir / f'{tile_id}.parquet'
    if not lvis_metrics_parq.exists():
        return f'{tile_id}: L2A metrics {lvis_metrics_parq} missing, skipped'

    pair_gdf = gpd.read_parquet(parq_path)
    if pair_gdf.empty:
        return f'{tile_id}: pair parquet empty, skipped'

    # `fhd` is the only column we need from L2A. shotnumber would let
    # us join by id instead of distance, but extract_lvis_gedi_pairs
    # doesn't carry SHOTNUMBER through right now.
    lvis_metrics = gpd.read_parquet(
        lvis_metrics_parq, columns=['geometry', 'fhd'])
    if lvis_metrics.empty:
        return f'{tile_id}: L2A metrics parquet empty, skipped'

    # Reproject L2A into the pair geometry CRS (UTM) so distances are
    # in metres and matchable against max_radius_m.
    lvis_metrics = lvis_metrics.to_crs(pair_gdf.crs)

    pair_xy = np.column_stack([
        pair_gdf.geometry.x.to_numpy(), pair_gdf.geometry.y.to_numpy()])
    l2a_xy = np.column_stack([
        lvis_metrics.geometry.x.to_numpy(),
        lvis_metrics.geometry.y.to_numpy()])
    tree = cKDTree(l2a_xy)
    dist, idx = tree.query(pair_xy, k=1, distance_upper_bound=max_radius_m)
    matched = np.isfinite(dist)

    fhd = np.full(len(pair_gdf), np.nan, dtype=np.float32)
    fhd[matched] = lvis_metrics['fhd'].iloc[idx[matched]].to_numpy()
    fhd_dist = np.where(matched, dist.astype(np.float32), np.nan)
    pair_gdf['lvis_fhd'] = fhd
    pair_gdf['lvis_fhd_match_distance_m'] = fhd_dist

    pair_gdf.to_parquet(out_file)
    n_total = len(pair_gdf)
    n_matched = int(matched.sum())
    return (f'{tile_id}: {n_matched}/{n_total} pairs got lvis_fhd '
            f'(median match dist={np.nanmedian(fhd_dist):.1f} m) '
            f'-> {out_file.name}')


def attach_lvis_fhd_to_lvis_gedi_pairs(
        pairs_dir: str,
        lvis_metrics_dir: str,
        save_dir: str,
        max_radius_m: float = 12.5,
        max_workers: int = 4,
        rewrite: bool = False,
        **kwargs) -> None:
    """Append `lvis_fhd` (and `lvis_fhd_match_distance_m`) columns to each
    LVIS-GEDI pair parquet by spatial-joining to the LVIS L2A footprint
    Cover product.

    The RH pair extractor uses LVIS L2 (Geolocated Surface Elevation,
    `RH10..RH100`); FHD lives in the separate L2A Cover product with
    its own per-tile parquet. SHOTNUMBER would let us join exactly, but
    extract_lvis_gedi_pairs doesn't carry SHOTNUMBER through, so we
    nearest-neighbour the L2A shot within `max_radius_m` of the pair's
    geometry. L2A and L2 shots are co-located by design — distances
    should be near zero in practice. `lvis_fhd_match_distance_m` is
    written alongside so you can see when that assumption breaks.

    Args:
        pairs_dir: dir of LVIS-GEDI pair parquets from
            `extract_lvis_gedi_pairs`.
        lvis_metrics_dir: dir of LVIS L2A per-tile parquets from
            `partition_lvis_metrics_by_s2_tile` (`<tile>.parquet` each
            carrying `fhd` + geometry).
        save_dir: output dir for augmented `<tile>.parquet` (same
            filename convention as the input — drop-in for downstream).
        max_radius_m: max pair->nearest-L2A distance to count as a
            match. ~12.5 m is the same default as the LVIS-GEDI pairing
            (GEDI footprint radius); L2/L2A pairs should be ~0 m apart
            so any large match distance flags an L2A coverage gap.
        max_workers: tile-level process-pool parallelism.
        rewrite: redo tiles whose augmented parquet already exists.
    """
    pairs_dir = Path(pairs_dir).expanduser()
    lvis_metrics_dir = Path(lvis_metrics_dir).expanduser()
    save_dir = Path(save_dir).expanduser()
    save_dir.mkdir(parents=True, exist_ok=True)

    parq_files = sorted(pairs_dir.glob('*.parquet'))
    if not parq_files:
        raise FileNotFoundError(f'No <tile>.parquet under {pairs_dir}')
    print(f'Attaching LVIS L2A fhd to {len(parq_files)} pair parquets '
          f'from {lvis_metrics_dir}')

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futs = {
            ex.submit(_attach_lvis_fhd_one_tile, p, lvis_metrics_dir,
                      save_dir, max_radius_m, rewrite): p.stem
            for p in parq_files
        }
        for fut in as_completed(futs):
            tile = futs[fut]
            try:
                print(fut.result())
            except Exception as e:
                print(f'{tile}: FAILED -> {e}')

    print(f'\nDone. Augmented pairs in {save_dir}')


# ============================================================================
# Hydra entrypoint
# ============================================================================
import hydra
from config.loader import register
from config.runner import run_cli

register(
    Path(__file__).resolve().parents[1] / 'config' / 'eval' / 'config.yaml',
    section='on_lvis',
    default_run='partition_lvis_by_s2_tile',
)


@hydra.main(config_name='no_log', version_base='1.2', config_path='../config/base')
def main(cfg):
    run_cli(cfg)


if __name__ == '__main__':
    main()
