"""Match GEDI 2020 and LVIS Gabon 2016 footprints inside stable forest.

Goal: how much sub-canopy structure does GEDI actually resolve compared to
LVIS, on pixels whose canopy class didn't change between the two
acquisitions (stable forest mask from `evaluation/forest_mask.py`).

Per S2 tile in `meta_file`:
  1. Load the GEDI per-tile parquet (`rh0..rh100`, `slope`, geometry) and
     the LVIS per-tile parquet (`RH10..RH100`, geometry).
  2. Reproject both to the forest mask's CRS, sample the mask at each
     shot center, and keep only shots on stable-forest pixels.
  3. For every GEDI shot, find the nearest LVIS shot within
     `max_radius_m` (KDTree in UTM). 1:1 pairing; GEDI shots without a
     within-radius LVIS shot are dropped.
  4. Write one per-tile parquet of paired RH columns + match distance,
     suitable as input to `structure_partial_correlation` (using
     `true_pattern="lvis_RH{lvl}"`, `pred_pattern="gedi_rh{lvl}"`).

Caveats:
  - GEDI RH levels are 0..100 (integer); LVIS levels are
    10,15,...,90,95,96,97,98,99,100. Output keeps both naming schemes
    distinct.
  - Nearest-neighbour at a small radius (a few m) means most GEDI shots
    are *inside* a GEDI footprint disk full of LVIS shots; the paired
    LVIS sample is one specific point inside that disk, not an
    integrating estimate. Use the disk-aggregation extractor if you need
    physically aligned area integration instead.
"""
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from scipy.spatial import cKDTree


# LVIS RH percentiles emitted by the on_lvis partition. Uppercase to match
# the column names already in the LVIS per-tile parquets.
LVIS_RH_COLUMNS = [
    'RH10', 'RH15', 'RH20', 'RH25', 'RH30', 'RH35', 'RH40', 'RH45',
    'RH50', 'RH55', 'RH60', 'RH65', 'RH70', 'RH75', 'RH80', 'RH85',
    'RH90', 'RH95', 'RH96', 'RH97', 'RH98', 'RH99', 'RH100',
]
# GEDI RH percentiles are dense 0..100.
GEDI_RH_COLUMNS = [f'rh{i}' for i in range(101)]
# Optional QA columns passed through to the pair parquet when present in
# the source GEDI file. Missing ones are silently dropped — different
# vintages of the GEDI prep carry different metadata.
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
    import pyarrow.parquet as pq
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
            process (e.g. meta_lvis_profile.csv).
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
    parquet. Reuses on_lvis sampling helpers so VSM is converted from
    decimetres to metres exactly the same way as extract_lvis_vsm_pairs.
    """
    from evaluation.on_lvis import (
        _row_col_for_points, _sample_raster_at_rowcol)

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
    decimetres; the on_lvis helper divides by 10 to land back in metres.

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
    from evaluation.on_lvis import RH_METRICS

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
    section='lvis_vs_gedi',
    default_run='extract_lvis_gedi_pairs',
)


@hydra.main(config_name='no_log', version_base='1.2', config_path='../config/base')
def main(cfg):
    run_cli(cfg)


if __name__ == '__main__':
    main()
